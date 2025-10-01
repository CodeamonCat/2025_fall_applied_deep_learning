#!/bin/bash

# Create logs directory if not exists
mkdir -p logs

# List of models to run
models=(
    "bert-base-chinese"
    "hfl/chinese-bert-wwm-ext"
    "hfl/chinese-lert-base"
    "hfl/chinese-macbert-base"
    "hfl/chinese-pert-base-mrc"
    "hfl/chinese-roberta-wwm-ext"
)

for model in "${models[@]}"; do
    # Replace / with _ for safe filenames
    model_name=$(echo "$model" | tr '/' '_')

    echo "=============================="
    echo ">>> Running pipeline for model: $model"
    echo "=============================="

    echo ">>> Training and evaluating MC task..."
    #### 1. Run MC task with hw1_swag.py
    python hw1_mc.py \
        --model_name_or_path "$model" \
        --tokenizer_name "$model" \
        --context_file ./dataset/context.json \
        --train_file ./dataset/train.json \
        --validation_file ./dataset/valid.json \
        --max_seq_length 512 \
        --do_train \
        --do_eval \
        --do_predict false \
        --gradient_accumulation_steps 2 \
        --learning_rate 3e-5 \
        --num_train_epochs 1 \
        --output_dir ./result_mc_"$model_name" \
        --overwrite_output_dir \
        --per_device_eval_batch_size 2 \
        --per_device_train_batch_size 1 \
        --save_total_limit 1 \
        --save_strategy steps \
        --save_steps 10000 \
        > logs/log_mc_"$model_name"_train.txt 2>&1

    echo ">>> Predicting MC task..."
    python hw1_mc.py \
        --model_name_or_path ./result_mc_"$model_name" \
        --tokenizer_name ./result_mc_"$model_name" \
        --context_file ./dataset/context.json \
        --test_file ./dataset/test.json \
        --max_seq_length 512 \
        --do_train false \
        --do_eval false \
        --do_predict true \
        --gradient_accumulation_steps 2 \
        --learning_rate 3e-5 \
        --num_train_epochs 1 \
        --output_dir ./result_mc_"$model_name" \
        --overwrite_output_dir \
        --per_device_eval_batch_size 2 \
        --per_device_train_batch_size 1 \
        --save_total_limit 1 \
        --save_strategy steps \
        --save_steps 10000 \
        > logs/log_mc_"$model_name"_predict.txt 2>&1
    
    echo ">>> Training and evaluating QA task..."
    #### 2. Run QA training with hw1_qa.py
    python hw1_qa.py \
        --model_name_or_path "$model" \
        --tokenizer_name "$model" \
        --context_file ./dataset/context.json \
        --train_file ./dataset/train.json \
        --validation_file ./dataset/valid.json \
        --max_seq_length 512 \
        --do_train \
        --do_eval \
        --do_predict false \
        --doc_stride 128 \
        --gradient_accumulation_steps 2 \
        --learning_rate 3e-5 \
        --num_train_epochs 2 \
        --output_dir ./result_qa_"$model_name" \
        --overwrite_output_dir \
        --per_device_train_batch_size 1 \
        --save_total_limit 1 \
        --save_strategy steps \
        --save_steps 10000 \
        > logs/log_qa_"$model_name"_train.txt 2>&1

    echo ">>> Predicting QA task..."
    #### 3. Run QA prediction with hw1_qa.py
    python hw1_qa.py \
        --model_name_or_path ./result_qa_"$model_name" \
        --tokenizer_name ./result_qa_"$model_name" \
        --context_file ./dataset/context.json \
        --test_file ./result_mc_"$model_name"/test_predictions.json \
        --max_seq_length 512 \
        --do_train false \
        --do_eval false \
        --do_predict true \
        --doc_stride 128 \
        --gradient_accumulation_steps 2 \
        --learning_rate 3e-5 \
        --num_train_epochs 2 \
        --output_dir ./result_qa_"$model_name" \
        --overwrite_output_dir \
        --per_device_train_batch_size 1 \
        --save_total_limit 1 \
        --save_strategy steps \
        --save_steps 10000 \
        > logs/log_qa_"$model_name"_predict.txt 2>&1

    done
