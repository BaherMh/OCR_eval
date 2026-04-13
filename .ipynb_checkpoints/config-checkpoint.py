from functools import partial

from src.methods.model_easy_ocr import ModelEasyOCR
from src.methods.model_paddle_ocr import ModelPaddleOCR
from src.methods.model_tesseract import ModelTesseract

dataset_paths = {
    "RDIOD": "../data/RDIOD.tsv",
    "Hand": "../data/Hand.tsv",
    "RusTwit": "../data/RusTwit.tsv",
    "RusTwit_real": "../data/RusTwit_real.tsv",
    "OCRBench": "../data/OCRBench.tsv",
    "Simple": "../data/Simple.tsv"
}

# Store classes, not instances
models = {
    "ppocr_v5_mobile_ru": partial(ModelPaddleOCR, model_name="ppocr_v5_mobile_ru"),
    "activations_64": partial(ModelPaddleOCR, model_name="activations_64", rec_model_dir="/home/bm_user/masters/models/inference/activations/"),
    "activations_256": partial(ModelPaddleOCR, model_name="activations_256", rec_model_dir="/home/bm_user/masters/models/inference/activations_256/"),
    "slerp": partial(ModelPaddleOCR, model_name="slerp", rec_model_dir="/home/bm_user/masters/models/inference/slerp/"),
    "tuned": partial(ModelPaddleOCR, model_name="tuned", rec_model_dir="/home/bm_user/masters/models/inference/tuned/"),
    "decoupled_slerp": partial(ModelPaddleOCR, model_name="decoupled_slerp", rec_model_dir="/home/bm_user/masters/models/inference/decoupled_slerp/"),
    
    
    
    
    # "EasyOCR": ModelEasyOCR,
    # "EasyOCR_en": partial(ModelEasyOCR, lang='en'),
    # "ppocr_v4_mobile": partial(ModelPaddleOCR, model_name="PP-OCRv4_mobile_rec"),
    # "ppocr_v4_server": partial(ModelPaddleOCR, model_name="PP-OCRv4_server_rec"),
    # "tesseract": ModelTesseract,
    # "tesseract_en": partial(ModelTesseract, lang='eng'),
}