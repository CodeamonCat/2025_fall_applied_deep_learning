#!bin/bash
# ${1}: path to the model checkpoint folder
# ${2}: path to the adapter_checkpoint downloaded under your folder
# ${3}: path to the input file (.json)
# ${4}: path to the output file (.json)

python infer.py \
    --base_model_path ${1} \
    --peft_path ${2} \
    --input_path ${3} \
    --output_path ${4} \