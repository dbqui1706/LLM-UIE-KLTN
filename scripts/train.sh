#!/bin/bash
export PYTHONPATH=./
export CUDA_VISIBLE_DEVICES=0

python3 train/train.py \
  --max_seq_length 2048 \
  --train_file_path data/train/data.json \
  --dataset_text_field text \
  --dataset_num_proc 2 \
  --model_type llama3.1_8B \
  --load_in_4bit True \
  --lora_r 256 \
  --lora_alpha 512 \
  --use_rslora True \
  --lora_dropout 0 \
  --gradient_checkpointing True \
  --use_gradient_checkpointing unsloth \
  --output_dir checkpoint \
  --num_train_epochs 1 \
  --per_device_train_batch_size 64 \
  --gradient_accumulation_steps 1 \
  --learning_rate 5e-6 \
  --warmup_ratio 0.05 \
  --weight_decay 0.01 \
  --lr_scheduler_type cosine \
  --logging_steps 10 \
  --save_strategy steps \
  --save_steps 2000 \
  --save_total_limit 1 \
  --save_only_model True \
  --ddp_find_unused_parameters False