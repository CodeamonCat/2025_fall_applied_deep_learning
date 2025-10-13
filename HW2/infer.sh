#!bin/bash

python infer.py \
    --base_model_path ./result/checkpoint-1000 \
    --peft_path ./adapter_checkpoint \
    --input_path ./dataset/public_test.json \
    --output_path ./output.json