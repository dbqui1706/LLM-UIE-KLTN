import json
from datasets import Dataset
from typing import Dict, Text, List


def load_dataset_manually(file_path: str):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    return Dataset.from_list(data)


def load_json(file_path: Text) -> List[Dict]:
    with open(file_path, 'r') as f:
        config = json.load(f)

    return config


def write_json(data, file_path, encoding='utf-8'):
    with open(file_path, 'w', encoding=encoding) as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
