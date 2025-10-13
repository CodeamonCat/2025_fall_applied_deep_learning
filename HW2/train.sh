#!bin/bash

python train.py \
  --model_name_or_path "Qwen/Qwen3-4B" \
  --output_dir ./result \
  --dataset "./dataset/train.json" \
  --dataset_format "self-defined" \
  --bits 4 \
  --bf16 \
  --gradient_checkpointing True \
  --lr_scheduler_type "cosine" \
  --warmup_ratio 0.03 \
  --per_device_train_batch_size 2 \
  --per_device_eval_batch_size 2 \
  --gradient_accumulation_steps 8 \
  --learning_rate 2e-5 \
  --lora_r 64 \
  --lora_alpha 64 \
  --dataloader_num_workers 4 \
  --dataloader_pin_memory True \
  --max_steps 1000 \
  --save_steps 100 \
  --save_total_limit 1 \
  --do_train False \
  --do_eval True \
  --evaluation_strategy "steps" \
  --eval_steps 100 \
  --overwrite_output_dir \
  --tf32 True

# python train.py \
#   --model_name_or_path "Qwen/Qwen3-4B" \
#   --output_dir ./result \
#   --dataset "./dataset/train.json" \
#   --dataset_format "self-defined" \
#   --bits 4 \
#   --bf16 \
#   --gradient_checkpointing True \
#   --lr_scheduler_type "cosine" \
#   --warmup_ratio 0.03 \
#   --per_device_train_batch_size 4 \
#   --per_device_eval_batch_size 2 \
#   --gradient_accumulation_steps 8 \
#   --learning_rate 3e-5 \
#   --lora_r 64 \
#   --lora_alpha 64 \
#   --max_steps 500 \
#   --save_strategy "steps" \
#   --save_steps 100 \
#   --save_total_limit 1 \
#   --do_train True \
#   --do_eval True \
#   --eval_strategy "steps" \
#   --eval_steps 100 \
#   --logging_steps 10 \
#   --load_best_model_at_end True \
#   --metric_for_best_model "eval_loss" \
#   --overwrite_output_dir \
#   --tf32 True