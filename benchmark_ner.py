import time
from glob import glob
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils.io import load_json
from utils.metrics import recall
from utils.preprocessing_text import preprocessing_text


def inference(user_prompt, ner_labels, query):
    user_prompt = user_prompt.format(ner_labels=ner_labels, text=query)
    messages = [
        {
            "role": "system",
            "content": "You are an expert in Named Entity Recognition (NER) task."
        },
        {
            "role": "user",
            "content": user_prompt
        }
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer(text, return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=512,
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    predict_list = response.split('|')
    predict_list = [preprocessing_text(x).lower() for x in predict_list]

    return predict_list


if __name__ == '__main__':
    model_name_or_path = "llama_checkpoint"

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype="auto",
        device_map="cuda",
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id

    folders = glob('data/raw/NER/*')

    for folder in tqdm(folders):
        data_name = folder.split('/')[-1].replace(' ', '_').lower()
        test_file = f'{folder}/test.json'
        label_file = f'{folder}/labels.json'

        ner_labels = load_json(label_file)
        ner_labels = [x.strip() for x in ner_labels if len(x) >= 2]

        user_prompt = """
            Extract entities from the text **strictly using ONLY the provided Entity List** below and **MUST** strictly adhere to the output format.
            Format output as '<entity tag>: <entity name>' and separated multiple entities by '|'. Return 'None' if no entities are identified.
            Entity List: {ner_labels}
            Text: {text}
        """

        benchmark_data = load_json(test_file)

        accuracy = 0.0
        valid_sample = 0

        print(f'Running on {data_name}')

        for metadata in tqdm(benchmark_data):
            query = preprocessing_text(metadata["sentence"])
            entities = metadata['entities']

            ground_truth_entity = []

            for entity in entities:
                name = entity['name']
                type = entity['type']

                answer = type + ': ' + name

                ground_truth_entity.append(preprocessing_text(answer).lower())

            if len(ground_truth_entity) == 0:
                continue

            predict_entity = inference(user_prompt, ner_labels, query)

            score = recall(ground_truth_entity, predict_entity)

            print()
            print(score)
            print(predict_entity)
            print(ground_truth_entity)
            print()

            accuracy += score
            valid_sample += 1

        accuracy = accuracy / valid_sample

        print(f'{data_name}: Accuracy: {accuracy * 100:.2f}%\n')

        with open('results/result_NER.txt', 'a') as f:
            f.write(f"{data_name}: {accuracy * 100:.2f}%\n")

        time.sleep(10)
