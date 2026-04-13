from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from src.methods.base_ocr import BaseOCR


class ModelQwen(BaseOCR):
    def __init__(self, lang='ru') -> None:
        super().__init__()
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen3-VL-2B-Instruct", dtype="auto", device_map="auto"
)
        self.processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-2B-Instruct")
        self.model_name = "Qwen3-VL"

    def run_method(self, image_path):
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": "/home/bm_user/masters/eval/data/RDIOD/0.jpg",
                    },
                    {"type": "text", "text": """"Извлеки весь текст с изображения дословно. 
        Требования к выводу:
        - Только сырой текст, без форматирования
        - Без объяснений, комментариев или вступлений
        - Без markdown, кавычек или служебных символов
        - Сохраняй оригинальные переносы строк и абзацы
        - Не исправляй опечатки, не суммируй, не интерпретируй
        - Если текст нечитаем — оставь как есть или используй [?]

        Начинай вывод сразу с первого символа текста."""},
                ],
            }
        ]

        # Preparation for inference
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        )
        inputs = inputs.to(self.model.device)

        # Inference: Generation of the output
        generated_ids = self.model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        return output_text[0]

