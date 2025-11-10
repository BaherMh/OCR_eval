import base64
import os
from difflib import SequenceMatcher

import jellyfish
import numpy as np
import pandas as pd
from jiwer import cer, wer


def exctract_images(tsv_data, save_dir):

    df = pd.read_csv(tsv_data, sep='\t')

    # Create a directory to save images (optional)
    os.makedirs(save_dir, exist_ok=True)

    # Iterate over each row
    for _, row in df.iterrows():
        index = row['index']
        image_base64 = row['image']
        
        # Skip if image data is missing
        if pd.isna(image_base64) or image_base64.strip() == '':
            continue
        
        # Decode base64 image
        try:
            image_data = base64.b64decode(image_base64)
        except Exception as e:
            print(f"Failed to decode image for index {index}: {e}")
            continue
        
        # Save image
        image_path = f"{save_dir}/{index}.png"  # assuming PNG; adjust extension if needed
        with open(image_path, 'wb') as f:
            f.write(image_data)

def unify_string_format(text):
    return text.strip().lower().replace('\n',' ').replace(' ', '')



def compute_cer(answer: str, prediction: str) -> float:
    """
    Compute Character Error Rate (CER) using jiwer.
    CER = (S + D + I) / N_characters
    """
    return cer(answer, prediction)

def compute_wer(answer: str, prediction: str) -> float:
    """
    Compute the Word Error Rate (WER) between a reference (answer) and a hypothesis (prediction).
    WER = (S + D + I) / N
    where:
      S = number of substitutions
      D = number of deletions
      I = number of insertions
      N = number of words in the reference (answer)
    """
    return wer(answer, prediction)

def compute_jaro_winkler_distance(answer: str, prediction: str) -> float:
    """
    Compute the Jaro-Winkler distance between two strings.
    The distance is (1 - similarity), so 0 = identical, 1 = completely different.
    """
    similarity = jellyfish.jaro_winkler_similarity(answer, prediction)
    distance = 1.0 - similarity
    return distance
