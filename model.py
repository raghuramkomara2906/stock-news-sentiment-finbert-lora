import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import LoraConfig, get_peft_model


def load_tokenizer(model_id: str):
    """Load and configure tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": tokenizer.eos_token or "[PAD]"})
    tokenizer.padding_side = "right"
    return tokenizer


def build_model(model_id: str, num_labels: int, device: str):
    """Load base sequence classification model and wrap with LoRA."""
    model = AutoModelForSequenceClassification.from_pretrained(
        model_id, num_labels=num_labels
    ).to(device)
    lora_cfg = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["query", "value"],
        lora_dropout=0.05,
        bias="none",
        task_type="SEQ_CLS",
    )
    return get_peft_model(model, lora_cfg)