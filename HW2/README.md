# <h1 align="center">NTU 2025 Fall Applied Deep Learning HW2</h1>
<p align="center">
<strong>CHIH-HAO LIAO</strong><br>
School of Forestry and Resource Conservation<br>
Graduate Institute of Biomedical Electronics and Bioinformatics<br>
National Taiwan University<br>
<a href="mailto:R11625015@ntu.edu.tw">R11625015@ntu.edu.tw</a><br>
October 13, 2025
</p>

## Installation
```bash
$ conda env create -f environment.yml -y
$ conda conda activate ADL_HW2
$ bash ./download.sh
```
## Quickstart
To run infer directly, use the script below
```bash
$ bash ./run.sh /path/to/model-folder /path/to/adapter_checkpoint \ /path/to/input.json /path/to/output.json
```

## Getting Started
We followed the example provided in the QLoRA source code [[1](#qlora)] to train our base model Qwen/Qwen3-4B [[2](#Qwen3-4B)]

### Model Training and Evaluation
```bash
$ bash train.sh
```

### Model Inference
```bash
$ bash infer.sh
```

### Perplexity Calculation
```bash
$ bash ppl.sh
```

## Reference
1. <a id="qlora"></a> [QLoRA: Efficient Finetuning of Quantized LLMs Source Code](https://github.com/artidoro/qlora/blob/main/qlora.py)
2. <a id="Qwen3-4B"></a> [Hugging Face Qwen/Qwen3-4B Model](https://huggingface.co/Qwen/Qwen3-4B)