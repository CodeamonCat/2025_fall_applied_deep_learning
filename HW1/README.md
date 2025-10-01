# <h1 align="center">NTU 2025 Fall Applied Deep Learning HW1</h1>
<p align="center">
<strong>CHIH-HAO LIAO</strong><br>
School of Forestry and Resource Conservation<br>
Graduate Institute of Biomedical Electronics and Bioinformatics<br>
National Taiwan University<br>
<a href="mailto:R11625015@ntu.edu.tw">R11625015@ntu.edu.tw</a><br>
October 01, 2025
</p>

## Installation
```bash
$ conda env create -f environment.yml
$ conda conda activate ADL_HW1
$ bash ./download.sh
```

## Quickstart
To run infer directly, use the script below
```bash
$ bash ./run.sh /path/to/context.json /path/to/test.json /path/to/pred/prediction.csv
```
To run all possible models directly, use the script below.
```bash
$ bash ./logs.sh
```

## Getting Started
### Paragraph Selection
Given a set of paragraphs and a question, select the paragraph that contains the answer [[1](#swag_mc)].

#### Model Training and Evaluation
```bash
$ python hw1_mc.py config_mc.json
```

#### Model Inference
```bash
$ python hw1_mc.py config_mc_test.json
```

### Span Selection (Extractive QA)
Given a paragraph and a question, extract the exact span of text from the paragraph that answers the question [[2](#swag_qa)].

#### Model Training and Evaluation
```bash
$ python hw1_qa.py config_qa.json
```

#### Model Inference
```bash
$ python hw1_qa.py config_qa_test.json
```

## Reference
1. <a id="swag_mc"></a> [SWAG Multiple Choice Example](https://github.com/huggingface/transformers/blob/main/examples/pytorch/multiple-choice/run_swag_no_trainer.py)
2. <a id="huggingface_mc"></a> [Hugging Face Multiple choice](https://huggingface.co/docs/transformers/en/tasks/multiple_choice)
3. <a id="swag_qa"></a> [Extractive QA Example](https://github.com/huggingface/transformers/blob/main/examples/pytorch/question-answering/run_qa_no_trainer.py)
4. <a id="huggingface_qa"></a> [Hugging Face Question answering](https://huggingface.co/docs/transformers/en/tasks/question_answering)
