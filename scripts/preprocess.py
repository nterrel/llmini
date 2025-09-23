# preprocess.py
from pathlib import Path


def preprocess(input_path, output_path):
    text = Path(input_path).read_text(encoding="utf-8")
    processed_text = text.lower()  # Example preprocessing
    Path(output_path).write_text(processed_text)
