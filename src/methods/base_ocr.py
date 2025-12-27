import ast
import json
import os
from abc import ABC, abstractmethod

import pandas as pd
from tqdm import tqdm

from general_config import images_path, output_path
from src.utils import (
    compute_cer,
    compute_jaro_winkler_distance,
    compute_wer,
    exctract_images,
    unify_string_format,
)


class BaseOCR(ABC):
    def __init__(self) -> None:
        self.data_folder = images_path
        self.output_path = output_path
        self.model_name = None  # Must be set by subclasses


    @abstractmethod
    def run_method(self, image_path):
        """Run inference on one image, and outputs a string corresponding to the text extracted"""
        pass


    def inference_tsv(self, tsv_path, debug_mode=False):
        df = pd.read_csv(tsv_path, delimiter='\t')
        dataset = os.path.basename(tsv_path).split('.')[0]
        images_folder = os.path.join(self.data_folder, dataset+'/')
        output_csv = f"{self.output_path}/{self.model_name}/{dataset}/{self.model_name}_{dataset}.csv"
        if debug_mode:
            output_csv = f"{self.output_path}/{self.model_name}/{dataset}_debug/{self.model_name}_{dataset}.csv"
        if os.path.exists(output_csv) and not debug_mode:
            print(f"the results of model {self.model_name} on dataset {dataset} is already Done!")
            return output_csv
        if not os.path.exists(images_folder):
            print("Extracting Images!")
            exctract_images(tsv_path, images_folder)
            # exctract_images(tsv_path, images_folder)
        else:
            print("IMAGES FOLDER FOUND!")
        results = []
        for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Processing rows"):
            if index > 5 and debug_mode: # in debug mode, inference only first 5
                break
            image_path = os.path.join(images_folder, str(row['index'])+'.png')
            try:
                ocr_res = self.run_method(image_path)
            except:
                print(f"ERROR happened when inferencing image {index}")
                ocr_res = "ERROR IN PARSING!!"
            results.append({
                'index': row['index'],
                'answer': row['answer'],
                'prediction': ocr_res
            })
        
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        results_df = pd.DataFrame(results)
        results_df.to_csv(output_csv, index=False, encoding='utf-8-sig')
        print(f"OCR results saved to {output_csv}")
        return output_csv


    def eval_results(self, csv_path: str, dataset: str, debug_mode=False):
        """
        Evaluate OCR results from a CSV with 'answer' and 'prediction' columns.
        Handles both single answers and lists of acceptable answers.
        For each sample, metrics are computed against all reference answers,
        and the best result (e.g., lowest CER, highest Jaro-Winkler, or exact match) is kept.

        Files are saved in: os.path.join(self.output_path, self.model_name)
        """
        if self.model_name is None:
            raise ValueError("model_name must be set before calling eval_results.")

        # Read CSV
        df = pd.read_csv(csv_path)

        if 'answer' not in df.columns or 'prediction' not in df.columns:
            raise ValueError("CSV must contain 'answer' and 'prediction' columns.")

        # Helper: Parse and normalize answer into a list of cleaned strings
        def parse_and_clean_answers(answer_val):
            # Handle string vs list representation
            if isinstance(answer_val, str):
                try:
                    # Try to parse as literal list
                    parsed = ast.literal_eval(answer_val)
                    if not isinstance(parsed, list):
                        parsed = [parsed]
                except (ValueError, SyntaxError):
                    # Not a list — treat as single answer
                    parsed = [answer_val]
            elif isinstance(answer_val, list):
                parsed = answer_val
            else:
                # Handle NaN, None, etc.
                parsed = [str(answer_val) if pd.notna(answer_val) else ""]

            # Clean each candidate answer
            return [unify_string_format(str(a).lower()) for a in parsed]

        # Clean prediction
        df['pred_clean'] = df['prediction'].astype(str).str.lower().map(unify_string_format)

        # Parse and clean all answers into list of strings per row
        df['answer_clean_list'] = df['answer'].apply(parse_and_clean_answers)

        # Now compute best metrics per row by comparing pred_clean to all answer_clean candidates
        def compute_best_metrics(row):
            pred = row['pred_clean']
            refs = row['answer_clean_list']

            # Handle empty refs (fallback)
            if not refs:
                refs = [""]

            # Exact match: True if any ref matches pred exactly
            exact = any(pred == ref for ref in refs)

            # Compute CER, WER, Jaro-Winkler for all refs and take best
            cers = [compute_cer(ref, pred) for ref in refs]
            wers = [compute_wer(ref, pred) for ref in refs]
            jaros = [compute_jaro_winkler_distance(ref, pred) for ref in refs]

            best_cer = min(cers)          # lower is better
            best_wer = min(wers)
            best_jaro = max(jaros)        # higher is better

            return pd.Series({
                'correct': exact,
                'cer': best_cer,
                'wer': best_wer,
                'jaro_winkler': best_jaro,
                'num_references': len(refs)
            })

        # Apply best-metric computation
        metric_cols = df.apply(compute_best_metrics, axis=1)
        df = pd.concat([df, metric_cols], axis=1)

        # Aggregate metrics
        total = len(df)
        correct = int(df['correct'].sum())
        accuracy = round(correct / total if total > 0 else 0.0, 4)
        avg_cer = round(df['cer'].mean(), 4)
        avg_cer_filtered = round(df[df['cer'] < 1]['cer'].mean(), 4) if not df[df['cer'] < 1].empty else 0.0
        avg_wer = round(df['wer'].mean(), 4)
        avg_wer_filtered = round(df[df['wer'] < 1]['wer'].mean(), 4) if not df[df['wer'] < 1].empty else 0.0
        median_cer = round(df['cer'].median(), 4)
        avg_jaro = round(df['jaro_winkler'].mean(), 4)

        # Prepare output directory
        output_dir = os.path.join(self.output_path, self.model_name, dataset)
        if debug_mode:
            output_dir = os.path.join(output_dir, "debug")
        os.makedirs(output_dir, exist_ok=True)

        # Generate output filenames
        base_name = os.path.splitext(os.path.basename(csv_path))[0]
        json_output_path = os.path.join(output_dir, f"{base_name}_summary.json")
        csv_output_path = os.path.join(output_dir, f"{base_name}_evaluated.csv")

        # Save JSON result
        results = {
            "dataset": dataset,
            "total_samples": total,
            "exact_matches": correct,
            "accuracy": accuracy,
            "avg_cer": avg_cer,
            "avg_wer": avg_wer,
            "avg_cer_filtered": avg_cer_filtered,
            "avg_wer_filtered": avg_wer_filtered,
            "median_cer": median_cer,
            "avg_jaro": avg_jaro
        }
        with open(json_output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)

        # Optionally drop helper columns before saving CSV
        df_to_save = df.copy()
        # Keep original 'answer' and 'prediction', drop internal clean columns if desired
        # But you might want to keep 'answer_clean_list' for debugging
        df_to_save.to_csv(csv_output_path, index=False, encoding='utf-8-sig')

        print(f"Evaluation results saved to:\n  {json_output_path}\n  {csv_output_path}")