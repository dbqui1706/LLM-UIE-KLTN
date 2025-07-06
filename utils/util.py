from unsloth import FastLanguageModel
from huggingface_hub import HfApi
from transformers import AutoModel, AutoTokenizer

MODEL_CLASSES = {
    "qwen2.5_7B": FastLanguageModel,
    "qwen2.5_14B": FastLanguageModel,
    "qwen3_8B": FastLanguageModel,
    "qwen3_14B": FastLanguageModel,
    "qwen3_32B": FastLanguageModel,
    "llama3.1_8B": FastLanguageModel,
}

MODEL_PATHS = {
    "qwen2.5_7B": "unsloth/Qwen2.5-7B-Instruct",
    "qwen2.5_14B": "unsloth/Qwen2.5-14B-Instruct",
    "qwen3_8B": "unsloth/Qwen3-8B",
    "qwen3_14B": "unsloth/Qwen3-14B",
    "qwen3_32B": "unsloth/Qwen3-32B",
    "llama3.1_8B": "unsloth/Meta-Llama-3.1-8B-Instruct",
}


def count_prompt_length(tokenizer, prompt):
    messages = [
        {
            "role": "system",
            "content": prompt,
        },
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    tokenized = tokenizer(
        prompt,
        return_tensors="pt",
    )

    print(f'System Prompt has {tokenized["input_ids"].size()[1]} tokens')


def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        num_params = param.numel()

        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel

        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


def remove_duplicate(seq):
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]


def push_model_to_hub(folder_path: str):
    tokenizer = AutoTokenizer.from_pretrained(folder_path)
    model = AutoModel.from_pretrained(folder_path, device_map='auto')

    model.push_to_hub("namdp-ptit/LLama-8B-Instruct-MultiTask")
    tokenizer.push_to_hub("namdp-ptit/LLama-8B-Instruct-MultiTask")

    api = HfApi()
    api.upload_folder(
        folder_path=folder_path,
        repo_id="namdp-ptit/LLama-8B-Instruct-MultiTask",
        repo_type="model",
        commit_message="Upload model",
    )
