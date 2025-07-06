import string

from tqdm import tqdm
from glob import glob

from utils.io import load_json, write_json
from utils.preprocessing_text import preprocessing_text
from utils.util import remove_duplicate


def combine_data(folder_dir: str, mode: str = "train"):
    folders = glob(f'{folder_dir}/*')

    dataset = []

    for folder in tqdm(folders):
        folder_name = folder.split('/')[-1]
        files = glob(f'{folder}/*.json')

        labels = load_json(f'{folder}/labels.json')

        for file in tqdm(files):
            if mode in file:
                data = load_json(file)

                for metadata in data:
                    query = metadata['sentence']
                    entities = metadata['entities']

                    answer = []

                    if len(entities) == 0:
                        continue
                    else:
                        for entity in entities:
                            name = entity['name']
                            type = entity['type']

                            answer.append(f'{type}: {name}')

                        answer = remove_duplicate(answer)
                        answer = ' | '.join(answer)
                        answer = answer.strip().strip(string.punctuation).strip()

                    result = {
                        'task': 'NER',
                        'query': preprocessing_text(query),
                        'answer': preprocessing_text(answer),
                        'labels': labels,
                        'dataset_name': folder_name.lower().replace(' ', '_')
                    }

                    dataset.append(result)

    print(f'{mode} set has {len(dataset)} examples')

    if mode == "train":
        write_json(dataset, 'data/train/ner_data.json')
    elif mode == "dev":
        write_json(dataset, 'data/valid/ner_data.json')
    elif mode == "test":
        write_json(dataset, 'data/benchmark/ner_data.json')
    else:
        raise ValueError(f'Invalid mode {mode}. Choose from "train", "dev", "test".')


if __name__ == '__main__':
    folder_dir = 'data/raw/NER'

    combine_data(folder_dir, "train")
    combine_data(folder_dir, "test")
    combine_data(folder_dir, "dev")
