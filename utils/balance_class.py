from collections import defaultdict
from tqdm import tqdm

from utils.io import load_json, write_json


def balance_single_task(input_path: str, output_path: str):
    final_dataset = []

    dataset = load_json(input_path)

    data_per_dataset = defaultdict(list)
    dataset_counter = defaultdict(int)

    for metadata in dataset:
        dataset_counter[metadata['dataset_name']] += 1
        data_per_dataset[metadata['dataset_name']].append(metadata)

    max_count = max(dataset_counter.values())

    for dataset_name in data_per_dataset.keys():
        scale_data = max_count // dataset_counter[dataset_name]
        cur_dataset = data_per_dataset[dataset_name]

        for _ in range(scale_data):
            data_per_dataset[dataset_name] += cur_dataset
            dataset_counter[dataset_name] += len(cur_dataset)

    dataset_names = list(dataset_counter.keys())
    min_count = min(dataset_counter.values())
    min_batch_size = len(dataset_names)

    for iterator in tqdm(range(min_count)):
        for dataset_name in dataset_names:
            final_dataset.append(data_per_dataset[dataset_name][iterator])

    print(f'Total train dataset: {len(final_dataset)}')
    print(f'Must choose batch size divided by {min_batch_size}')

    write_json(final_dataset, file_path=output_path)

    return min_batch_size


def balance_multi_task(output_path: str):
    # Balance single tasks
    min_batch_ee = balance_single_task(
        input_path='data/train/ee_data.json',
        output_path='data/train/ee_data.json'
    )
    min_batch_re = balance_single_task(
        input_path='data/train/re_data.json',
        output_path='data/train/re_data.json'
    )
    min_batch_ner = balance_single_task(
        input_path='data/train/ner_data.json',
        output_path='data/train/ner_data.json'
    )

    ee_data = load_json('data/train/ee_data.json')
    re_data = load_json('data/train/re_data.json')
    ner_data = load_json('data/train/ner_data.json')

    start_id_ee, start_id_re, start_id_ner = 0, 0, 0

    min_length = min(len(ee_data), len(re_data), len(ner_data))
    min_batch_size = min_batch_ee + min_batch_re + min_batch_ner

    final_dataset = []

    for _ in tqdm(range(min_length)):
        end_id_ee = start_id_ee + min_batch_ee
        end_id_re = start_id_re + min_batch_re
        end_id_ner = start_id_ner + min_batch_ner

        for i in range(start_id_ee, end_id_ee):
            final_dataset.append(ee_data[i])
        for i in range(start_id_re, end_id_re):
            final_dataset.append(re_data[i])
        for i in range(start_id_ner, end_id_ner):
            final_dataset.append(ner_data[i])

        start_id_ee = end_id_ee
        start_id_re = end_id_re
        start_id_ner = end_id_ner

    print(f'Total train dataset: {len(final_dataset)}')
    print(f'Must choose batch size divided by {min_batch_size}')

    write_json(final_dataset, file_path=output_path)


if __name__ == '__main__':
    balance_multi_task('data/train/data.json')
