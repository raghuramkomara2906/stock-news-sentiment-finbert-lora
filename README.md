# stock-news-sentiment-finbert-lora

This repository contains scripts to fine-tune a LoRA adapter on FinBERT for stock-news sentiment classification and evaluate the results.

## Structure

- `data.py`       : Load and preprocess raw data.
- `model.py`      : Build the PEFT-wrapped model.
- `train.py`      : Train the LoRA adapter.
- `evaluate.py`   : Evaluate baseline and LoRA models.
- `utils.py`      : Seeding, device, and helper functions.
- `requirements.txt` : Dependency list.
- `README.md`     : This file.

## Setup
```bash
pip install -r requirements.txt
```

## Training
```bash
python train.py --data_path path/to/stock_news.csv --adapter_dir finbert_lora_adapter
```

## Evaluation
```bash
python evaluate.py --data_path path/to/stock_news.csv
```
