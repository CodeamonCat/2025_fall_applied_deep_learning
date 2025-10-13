from transformers import BitsAndBytesConfig
import torch


def get_prompt(instruction: str) -> str:
    """Format the instruction as a prompt for LLM."""
    # return f"你是人工智慧助理，以下是用戶和人工智能助理之間的對話。你要對用戶的問題提供有用、安全、詳細和禮貌的回答。USER: {instruction} ASSISTANT:"
    return (
        "你是世界頂尖的中文系教授，以下是文言文與白話文轉換的任務。根據指令要求，生成合適的翻譯。\n"
        f"指令：{instruction}\n"
        "請輸出翻譯結果："
    )


def get_bnb_config() -> BitsAndBytesConfig:
    """Get the BitsAndBytesConfig."""
    # pass
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        # load_in_4bit=True,
        load_in_8bit=False,
        llm_int8_threshold=6.0,
        llm_int8_has_fp16_weight=False,
        # bnb_4bit_compute_dtype=torch.bfloat16,
        # bnb_4bit_use_double_quant=True,
        # bnb_4bit_quant_type="nf4",
    )


def get_few_shot_prompt(instruction: str) -> str:
    examples = [
        (
            "翻譯成文言文：\n雅裏惱怒地說：從前在福山田獵時，你誣陷獵官，現在又說這種話。",
            "雅裏怒曰：昔畋於福山，卿誣獵官，今復有此言。",
        ),
        (
            "翻譯成白話文：\n議雖不從，天下咸重其言。",
            "他的建議雖然不被采納，但天下都很敬重他的話。",
        ),
    ]
    few_shot_examples = "\n\n".join(
        [f"指令：{ins}\n回答：{out}" for ins, out in examples]
    )
    return f"{few_shot_examples}\n\n指令：{instruction}\n回答："
