#!/bin/bash

# Create logs directory if not exists
mkdir -p dataset
mkdir -p logs
mkdir -p result_mc
mkdir -p result_qa

# Download dataset
wget -O ./dataset/context.json https://raw.githubusercontent.com/ADL-2025/ADL_HW1/main/dataset/context.json
wget -O ./dataset/train.json https://raw.githubusercontent.com/ADL-2025/ADL_HW1/main/dataset/train.json
wget -O ./dataset/valid.json https://raw.githubusercontent.com/ADL-2025/ADL_HW1/main/dataset/valid.json
wget -O ./dataset/test.json https://raw.githubusercontent.com/ADL-2025/ADL_HW1/main/dataset/test.json

# Download pre-trained models and unzip them
# hfl/chinese-lert-base
# Download and unzip result_mc.zip
gdown --id 1qD4KDv_VmaEngYaFi36_x_G9jHhGt2i6 -O result_mc.zip
unzip result_mc.zip -d result_mc

# Download and unzip result_qa.zip
gdown --id 1WdJzhfv7yk-cL5dDbVxtiQjGjfoP3LiR -O result_qa.zip
unzip result_qa.zip -d result_qa
