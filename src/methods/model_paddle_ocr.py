import os

import pandas as pd
from paddleocr import PaddleOCR

from src.methods.base_ocr import BaseOCR


class ModelPaddleOCR(BaseOCR):
    def __init__(self, lang='ru', model_name="PP-OCRv5_mobile_rec", rec_model_dir=None) -> None:
        super().__init__()
        if rec_model_dir is None:
            self.model = PaddleOCR(
            # text_recognition_model_name=model_name,
            lang=lang, 
            use_doc_orientation_classify=False, # Disable document orientation classification model
            use_doc_unwarping=False, # Disable text image unwarping model
            use_textline_orientation=False, # Disable text line orientation classification model
            )
        else:
            print("LOADING FINETUNED MODEL")
            self.model = PaddleOCR(
                rec_model_dir=rec_model_dir ,
                lang=lang,
                use_doc_orientation_classify=False, # Disable document orientation classification model
                use_doc_unwarping=False, # Disable text image unwarping model
                use_textline_orientation=False, # Disable text line orientation classification model
                )

        self.model_name = model_name

    def run_method(self, image_path):
        result = self.model.predict(image_path)
        full_text = " ".join(result[0]['rec_texts']) # to be checked why index 0
        return full_text

