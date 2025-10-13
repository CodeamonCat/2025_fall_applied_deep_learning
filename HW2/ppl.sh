#!bin/bash
python ppl.py \
    --base_model_path ./result/checkpoint-1000 \
    --peft_path ./adapter_checkpoint \
    --test_data_path ./dataset/public_test.json

# python ppl.py \
#     --base_model_path ./result/checkpoint-1000 \
#     --peft_path ./adapter_model \
#     --test_data_path ./dataset/private_test.json