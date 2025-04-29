import argparse
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

from data import load_raw_data, format_instruction_stock, get_splits
from utils import device

LABEL_LIST = ['negative','positive','neutral']

class SeqDataset(torch.utils.data.Dataset):
    def __init__(self, hf_dataset, tokenizer, max_length=256):
        self.ds = hf_dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
    def __len__(self): return len(self.ds)
    def __getitem__(self, idx):
        ex = self.ds[idx]
        tok = self.tokenizer(
            ex['prompt'], padding='max_length', truncation=True, max_length=self.max_length
        )
        return {
            'input_ids': torch.tensor(tok['input_ids'],dtype=torch.long),
            'attention_mask': torch.tensor(tok['attention_mask'],dtype=torch.long),
            'labels': torch.tensor(label2id[ex['label']],dtype=torch.long)
        }

def evaluate(model, loader):
    model.eval()
    preds, labels = [],[]
    with torch.no_grad():
        for b in loader:
            inputs = dict(input_ids=b['input_ids'].to(device), attention_mask=b['attention_mask'].to(device))
            logits = model(**inputs).logits
            preds.extend(logits.argmax(-1).cpu().tolist())
            labels.extend(b['labels'].tolist())
    return labels, preds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--adapter_dir', type=str, default='./finbert_lora_adapter')
    parser.add_argument('--model_id', type=str, default='ProsusAI/finbert')
    parser.add_argument('--batch_size', type=int, default=8)
    args = parser.parse_args()

    raw = load_raw_data(args.data_path)
    ds = raw.map(format_instruction_stock, remove_columns=raw.column_names)
    _,_,test_ds = get_splits(ds)
    tokenizer = AutoTokenizer.from_pretrained(args.adapter_dir)
    test_loader = DataLoader(SeqDataset(test_ds, tokenizer), batch_size=args.batch_size)

    base = AutoModelForSequenceClassification.from_pretrained(args.model_id, num_labels=3).to(device)
    true, pred_base = evaluate(base, test_loader)

    lora = AutoModelForSequenceClassification.from_pretrained(args.adapter_dir).to(device)
    _, pred_lora = evaluate(lora, test_loader)

    def pm(name, t, p):
        print(f"-- {name} --")
        print(f"Acc: {accuracy_score(t,p)*100:.2f}% | F1: {f1_score(t,p,average='weighted')*100:.2f}%")
        print(confusion_matrix(t,p))
        print(classification_report(t,p,target_names=LABEL_LIST))
    pm('Baseline', true, pred_base)
    pm('LoRA',     true, pred_lora)

    accs = [accuracy_score(true,p)*100 for p in [pred_base, pred_lora]]
    plt.bar(['Base','LoRA'], accs)
    plt.ylabel('Accuracy')
    plt.show()
    cm = confusion_matrix(true, pred_lora, normalize='true')
    ConfusionMatrixDisplay(cm, display_labels=LABEL_LIST).plot()
    plt.show()

if __name__ == '__main__': main()