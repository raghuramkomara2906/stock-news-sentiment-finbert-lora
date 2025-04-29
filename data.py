from datasets import load_dataset


def load_raw_data(path: str):
    """Load raw CSV into a Hugging Face Dataset."""
    return load_dataset("csv", data_files=path)["train"]


def format_instruction_stock(example):
    """Map raw example to prompt/label format."""
    label_map = {0: "negative", 1: "positive", 2: "neutral"}
    prompt = (
        "### Instruction:\n"
        "Classify the sentiment of this stock news headline.\n\n"
        "### Input:\n"
        f"{example['headline']}\n\n"
        "### Response:"
    )
    label = label_map.get(example.get("label", 2), str(example.get("label")))
    return {"prompt": prompt, "label": label}


def get_splits(ds, seed: int = 42):
    """Return train, val, test splits (80/10/10)."""
    split1 = ds.train_test_split(test_size=0.10, seed=seed)
    train_val, test = split1["train"], split1["test"]
    split2 = train_val.train_test_split(test_size=0.111111, seed=seed)
    train, val = split2["train"], split2["test"]
    return train, val, test