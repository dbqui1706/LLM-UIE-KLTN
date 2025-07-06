import time
from glob import glob

from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils.io import load_json
from utils.metrics import recall
from utils.preprocessing_text import preprocessing_text


def inference(user_prompt, ee_labels, query):
    user_prompt = user_prompt.format(ee_labels=ee_labels, text=query)
    messages = [
        {
            "role": "system",
            "content": "You are an expert in Event Extraction (EE) task."
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

    folders = glob('data/raw/EE/*')

    for folder in tqdm(folders):
        data_name = folder.split('/')[-1].replace(' ', '_').lower()
        test_file = f'{folder}/test.json'
        label_file = f'{folder}/labels.json'

        ee_labels = load_json(label_file)
        ee_labels = [x.strip() for x in ee_labels if len(x) >= 2]

        user_prompt = """
            Extract events and their components from text **strictly using ONLY the provided Event List** below and **MUST** strictly adhere to the output format.
            Format output as '<event_type>: <trigger_word> | <role1>: <argument1> | <role2>: <argument2>' and separate multiple events with '|'. Return 'None' if no events are identified.
            Event List: {ee_labels}
            Text: {text}
        """

        benchmark_data = load_json(test_file)

        accuracy = 0.0
        valid_sample = 0

        print(f'Running on {data_name}')

        for metadata in tqdm(benchmark_data):
            query = preprocessing_text(metadata["sentence"])
            events = metadata['events']

            if len(events) == 0:
                continue

            ground_truth_entity = []

            for entity in events:
                check_arguments = 1
                trigger = entity['trigger']
                typing = entity['type']

                ground_truth_entity.append(f"{typing}: {trigger}".lower())

                arguments = entity['arguments']

                if len(arguments) == 0:
                    check_arguments = 0

                if check_arguments == 0:
                    continue

                for arg in arguments:
                    name = arg['name']
                    role = arg['role']

                    ground_truth_entity.append(f"{role}: {name}".lower())

            if len(ground_truth_entity) == 0:
                continue

            ground_truth_entity = list(map(preprocessing_text, ground_truth_entity))
            predict_entity = inference(user_prompt, ee_labels, query)

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

        with open('results/result_EE.txt', 'a') as f:
            f.write(f"{data_name}: {accuracy * 100:.2f}%\n")

        time.sleep(10)
