#!/bin/bash

# Create logs directory if not exists
mkdir -p adapter_checkpoint
mkdir -p dataset
mkdir -p result

# Download dataset
wget -O ./dataset/train.json https://raw.githubusercontent.com/CodeamonCat/2025_fall_applied_deep_learning/main/HW2/dataset/train.json
wget -O ./dataset/public_test.json https://raw.githubusercontent.com/CodeamonCat/2025_fall_applied_deep_learning/main/HW2/dataset/public_test.json
wget -O ./dataset/private_test.json https://raw.githubusercontent.com/CodeamonCat/2025_fall_applied_deep_learning/main/HW2/dataset/private_test.json


# Download adapter configuration
gdown --id 15pQFAmYqn9yGKS_GzWs2tM5hHvkA7B2a -O ./adapter_checkpoint/adapter_config.json

# Download adapter weights
gdown --id 1C8IDkdgYyjm8p19XVdK03WjgAi41CKNG -O ./adapter_checkpoint/adapter_model.safetensors