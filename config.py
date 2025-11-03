from functools import partial

from src.methods.model_easy_ocr import ModelEasyOCR
from src.methods.model_paddle_ocr import ModelPaddleOCR
from src.methods.model_tesseract import ModelTesseract

dataset_paths = {
    "RusTwit": "C:/Users/baher/OneDrive/Desktop/masters/masters thesis/master_code/data/RusTwit.tsv",
    "RusTwit_real": "C:/Users/baher/OneDrive/Desktop/masters/masters thesis/master_code/data/RusTwit_real.tsv"
}

# Store classes, not instances
models = {
    "PaddleOCR": ModelPaddleOCR,
    "EasyOCR": ModelEasyOCR,
    "ppocr_v4": partial(ModelPaddleOCR, model_name="PP-OCRv4_mobile_rec"),
    "tesseract": ModelTesseract
}