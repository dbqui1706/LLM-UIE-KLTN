import torch
from transformers import BitsAndBytesConfig, AutoModelForSeq2SeqLM, AutoTokenizer

from utils.io import load_json
from utils.system_prompt import ee_cls_prompt, ner_cls_prompt, re_cls_prompt

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype="float16",
)

tokenizer = AutoTokenizer.from_pretrained("qwen2.5_7B_cls_checkpoint")

model = AutoModelForSeq2SeqLM.from_pretrained(
    "qwen2.5_7B_cls_checkpoint",
    torch_dtype=torch.bfloat16,
    device_map="auto",
    quantization_config=bnb_config
)

ner_labels = load_json('data/raw/NER/ACE 2004/labels.json')
ner_labels = [x.strip() for x in ner_labels if len(x) >= 2]

query = "Xinhua News Agency, Urumchi, September 1st, by reporters Shengjiang Li and Jian ' gang Ding"

prompt = ner_cls_prompt.format(ner_labels=ner_labels, query=query)

input_ids = tokenizer(prompt, return_tensors="pt").to("cuda")

output = model.generate(**input_ids)
print(tokenizer.decode(output[0], skip_special_tokens=True))
