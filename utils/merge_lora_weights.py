import torch
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM


def merge_lora_weights(
        model_base_name_or_path,
        lora_path,
        save_path
):
    tokenizer = AutoTokenizer.from_pretrained(model_base_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_base_name_or_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id

    model = PeftModel.from_pretrained(model, lora_path)

    merged_model = model.merge_and_unload()

    merged_model.save_pretrained(save_path, safe_serialization=True)
    tokenizer.save_pretrained(save_path)

    print(f"Save Model In: {save_path}")


if __name__ == '__main__':
    merge_lora_weights(
        model_base_name_or_path="unsloth/Meta-Llama-3.1-8B-Instruct",
        lora_path="checkpoint",
        save_path="llama_checkpoint"
    )
