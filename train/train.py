import logging
import os
import sys
import torch
import wandb
from transformers import HfArgumentParser
from trl import SFTTrainer, SFTConfig

from components.arguments import DataArguments, LoraArguments, ModelArguments
from utils.io import load_dataset_manually
from utils.system_prompt import ee_prompt, ner_prompt, re_prompt
from utils.util import MODEL_CLASSES, MODEL_PATHS, print_trainable_parameters

torch.cuda.manual_seed(23)
torch.random.manual_seed(23)

os.environ['UNSLOTH_RETURN_LOGITS'] = '1'

logger = logging.getLogger(__name__)


def main(
        data_args: DataArguments,
        lora_args: LoraArguments,
        model_args: ModelArguments,
        training_args: SFTConfig
):
    # Load and format the dataset
    train_dataset = load_dataset_manually(data_args.train_file_path)

    # Load tokenizer and model
    model_class = MODEL_CLASSES[model_args.model_type]
    model, tokenizer = model_class.from_pretrained(
        MODEL_PATHS[model_args.model_type],
        max_seq_length=training_args.max_seq_length,
        load_in_4bit=model_args.load_in_4bit
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id

    model = model_class.get_peft_model(
        model,
        r=lora_args.lora_r,
        lora_alpha=lora_args.lora_alpha,
        lora_dropout=lora_args.lora_dropout,
        use_gradient_checkpointing=lora_args.use_gradient_checkpointing,
        use_rslora=lora_args.use_rslora,
        target_modules=lora_args.target_modules,
    )

    print_trainable_parameters(model)

    def format_instruction(example):
        labels = example['labels']
        labels = [x.strip() for x in labels if len(x) >= 2]

        if example['task'] == 'EE':
            system_prompt = "You are an expert in Event Extraction (EE) task."
            user_prompt = ee_prompt
        elif example['task'] == 'NER':
            system_prompt = "You are an expert in Named Entity Recognition (NER) task."
            user_prompt = ner_prompt
        else:
            system_prompt = "You are an expert in Relation Extraction (RE) task."
            user_prompt = re_prompt

        user_prompt = user_prompt.format(labels=labels, text=example['query'])

        messages = [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
            {
                "role": "assistant",
                "content": example["answer"],
            }
        ]

        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )

        return {'text': prompt}

    train_dataset = train_dataset.map(format_instruction, num_proc=training_args.dataset_num_proc, batch_size=128)

    # Initialize the SFTTrainer with additional configurations
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        # eval_dataset=valid_dataset,
    )

    # Start training
    trainer.train()

    # Save the fine-tuned model
    trainer.save_model(training_args.output_dir)

    # Finish wandb run
    wandb.finish()


if __name__ == '__main__':
    parser = HfArgumentParser((DataArguments, LoraArguments, ModelArguments, SFTConfig))
    data_args, lora_args, model_args, training_args = parser.parse_args_into_dataclasses()

    log_file = os.path.join(training_args.output_dir, 'print_log.txt')

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    wandb.init(
        entity='phuongnamdpn2k2',
        project='htsc_finetuning',
        name='seaLLM',
        config={
            'data_args': vars(data_args),
            'lora_args': vars(lora_args),
            'model_args': vars(model_args),
            'training_args': vars(training_args),
        }
    )

    main(data_args, lora_args, model_args, training_args)
