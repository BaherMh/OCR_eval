import os

import easyocr
import pandas as pd
import pytesseract
from paddleocr import PaddleOCR
from PIL import Image

from src.methods.base_ocr import BaseOCR


class ModelTesseract(BaseOCR):
    def __init__(self, lang='rus') -> None:
        super().__init__()
        self.lang = lang

        self.model_name = "Tesseract"

    def run_method(self, image_path):
        img = Image.open(image_path)

        # Perform OCR
        text = pytesseract.image_to_string(img, lang=self.lang)
        return text