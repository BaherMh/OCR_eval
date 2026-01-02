from functools import partial

from src.methods.model_easy_ocr import ModelEasyOCR
from src.methods.model_paddle_ocr import ModelPaddleOCR
from src.methods.model_tesseract import ModelTesseract

dataset_paths = {
    "RusTwit": "../data/RusTwit.tsv",
    "RusTwit_real": "../data/RusTwit_real.tsv",
    "OCRBench": "../data/OCRBench.tsv",
    "Simple": "../data/Simple.tsv"
}

# Store classes, not instances
models = {
    "ppocr_v5_mobile_ru": partial(ModelPaddleOCR, model_name="ppocr_v5_mobile_ru"),
    "ppocr_v5_mobile_ru_finetuned": partial(ModelPaddleOCR, model_name="ppocr_v5_mobile_ru_finetuned", rec_model_dir="C:/Users/baher/OneDrive/Desktop/masters/masters_thesis/paddleOCR/PaddleOCR/inference/eslav_rec_ppocr_v5_exp1"),
    "ppocr_v5_mobile_ru_finetuned_exp2": partial(ModelPaddleOCR, model_name="ppocr_v5_mobile_ru_finetuned_exp2", rec_model_dir="C:/Users/baher/OneDrive/Desktop/masters/masters_thesis/paddleOCR/PaddleOCR/inference/eslav_rec_ppocr_v5_exp2"),
    "ppocr_v5_server": partial(ModelPaddleOCR, model_name = "PP-OCRv5_server_rec"),
    "EasyOCR": ModelEasyOCR,
    "EasyOCR_en": partial(ModelEasyOCR, lang='en'),
    "ppocr_v4_mobile": partial(ModelPaddleOCR, model_name="PP-OCRv4_mobile_rec"),
    "ppocr_v4_server": partial(ModelPaddleOCR, model_name="PP-OCRv4_server_rec"),
    "tesseract": ModelTesseract,
    "tesseract_en": partial(ModelTesseract, lang='eng'),
}