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
            ocr_res = self.run_method(image_path)
            results.append({
                'index': row['index'],
                'answer': row['answer'],
                'prediction': ocr_res
            })
        
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        results_df = pd.DataFrame(results)
        results_df.to_csv(output_csv, index=False)
        print(f"OCR results saved to {output_csv}")
        return output_csv


    def eval_results(self, csv_path: str, dataset: str, debug_mode=False):
        """
        Evaluate OCR results from a CSV with 'answer' and 'prediction' columns.
        Outputs a JSON summary with multiple metrics and an extended CSV with correctness flag.

        Files are saved in: os.path.join(self.output_path, self.model_name)
        Filenames are based on the input CSV, with '_res' added before '.csv'.

        Args:
            csv_path (str): Path to input CSV with 'answer' and 'prediction' columns.
            dataset (str): Name of dataset (used in output path).
            debug_mode (bool): If True, saves results in a debug subdirectory.
        """
        if self.model_name is None:
            raise ValueError("model_name must be set before calling eval_results.")

        # Read CSV
        df = pd.read_csv(csv_path)

        # Validate required columns
        if 'answer' not in df.columns or 'prediction' not in df.columns:
            raise ValueError("CSV must contain 'answer' and 'prediction' columns.")

        # Clean and normalize text
        df['answer_clean'] = df['answer'].astype(str).str.lower().map(unify_string_format)
        df['pred_clean'] = df['prediction'].astype(str).str.lower().map(unify_string_format)

        # Exact match accuracy
        df['correct'] = df['answer_clean'] == df['pred_clean']

        # Compute per-sample CER and WER
        df['cer'] = df.apply(lambda row: compute_cer(row['answer_clean'], row['pred_clean']), axis=1)
        df['wer'] = df.apply(lambda row: compute_wer(row['answer_clean'], row['pred_clean']), axis=1)
        df['jaro_winkler'] = df.apply(lambda row: compute_jaro_winkler_distance(row['answer_clean'], row['pred_clean']), axis=1)

        # Aggregate metrics
        total = len(df)
        correct = int(df['correct'].sum())
        accuracy = round(correct / total if total > 0 else 0.0, 4)
        avg_cer = round(df['cer'].mean(), 4)
        avg_wer = round(df['wer'].mean(), 4)
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
            "median_cer": median_cer,
            "avg_jaro": avg_jaro
        }
        with open(json_output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)

        # Save detailed CSV (without helper clean columns)
        save_df = df.drop(columns=['answer_clean', 'pred_clean'])
        save_df.to_csv(csv_output_path, index=False)

        print(f"Evaluation results saved to:\n  {json_output_path}\n  {csv_output_path}")