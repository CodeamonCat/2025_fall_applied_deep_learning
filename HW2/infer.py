import argparse
import importlib
import json
import os
import pandas as pd
import torch
from packaging import version

from peft import LoraConfig
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from train import is_ipex_available
from utils import get_bnb_config, get_prompt


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_model_path",
        type=str,
        default="./result/checkpoint-1000",
        required=False,
        help="Base model path (default: ./result/checkpoint-1000)",
    )
    parser.add_argument(
        "--peft_path",
        type=str,
        default="./adapter_checkpoint",
        required=False,
        help="PEFT adapter model path (default: ./adapter_checkpoint)",
    )
    parser.add_argument(
        "--input_path",
        type=str,
        default="./dataset/public_test.json",
        required=False,
        help="Test data path (default: dataset/public_test.json)",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="./output.json",
        required=False,
        help="Output path for results (default: ./output.json)",
    )
    parser.add_argument(
        "--max_memory_MB",
        type=int,
        default=80000,
        required=False,
        help="Free memory per GPU (default: 80000)",
    )
    return parser.parse_args()


def main() -> None:
    args = get_args()

    if torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()
    if is_ipex_available() and torch.xpu.is_available():
        n_gpus = torch.xpu.device_count()

    max_memory = {i: f"{args.max_memory_MB}MB" for i in range(n_gpus)}
    device_map = "auto"
    if os.environ.get("LOCAL_RANK") is not None:
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        device_map = {"": local_rank}
        max_memory = {"": max_memory[local_rank]}

    print(f"loading base model {args.base_model_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model_path,
        device_map=device_map,
        max_memory=max_memory,
        quantization_config=get_bnb_config(),
    )
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path)
    model = PeftModel.from_pretrained(model, args.peft_path, is_trainable=False)

    with open(args.input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    f.close()

    # results = list()
    # for item in tqdm(data, desc="Inferencing"):
    #     instruction = item["instruction"]
    #     prompt = get_prompt(instruction)
    #     inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    #     input_ids = inputs["input_ids"].to(model.device)
    #     generation_output = model.generate(
    #         input_ids=input_ids,
    #         max_new_tokens=512,
    #         do_sample=True,
    #         top_p=0.7,
    #         temperature=0.95,
    #         repetition_penalty=1.2,
    #         pad_token_id=tokenizer.eos_token_id,
    #     )
    #     outputs = tokenizer.batch_decode(generation_output, skip_special_tokens=True)
    #     result = outputs[0].split("請輸出翻譯結果：")[-1].strip()
    #     results.append({"id": item["id"], "output": result})

    results = []
    for i in tqdm(range(len(data))):
        result_dict = {}
        prompt = get_prompt(data[i]["instruction"])
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(**inputs, max_new_tokens=128)
        result_dict["id"] = data[i]["id"]
        result_dict["output"] = tokenizer.decode(outputs[0], skip_special_tokens=True)[
            len(prompt) :
        ].strip()
        results.append(result_dict)

    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    f.close()


if __name__ == "__main__":
    main()
