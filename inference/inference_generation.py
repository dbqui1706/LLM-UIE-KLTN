from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from utils.io import load_json
from utils.system_prompt import ner_prompt

model_name_or_path = "fluently/checkpoint"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path,
    torch_dtype="auto",
    device_map="auto",
    quantization_config=bnb_config
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id

ner_labels = load_json('data/raw/NER/WikiANN en/labels.json')
ner_labels = [x.strip() for x in ner_labels if len(x) >= 2]

prompt = ner_prompt.format(
    ner_labels=ner_labels,
)

question = "At the end of November, it became part of the 5th Army."

messages = [
    {
        "role": "system",
        "content": prompt
    },
    {
        "role": "user",
        "content": question
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

print(response)
