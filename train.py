import argparse
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import load_raw_data, format_instruction_stock, get_splits
from model import load_tokenizer, build_model
from utils import set_seed, device, clip_grad
from sklearn.model_selection import train_test_split

# Label mapping

label2id = {"negative": 0, "positive": 1, "neutral": 2}

class SequenceClassificationDataset(torch.utils.data.Dataset):
    def __init__(self, hf_dataset, tokenizer, max_length=256):
        self.ds = hf_dataset
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        ex = self.ds[idx]
        tok = self.tokenizer(
            ex['prompt'], padding='max_length', truncation=True, max_length=self.max_length
        )
        input_ids = tok['input_ids']
        attn = tok['attention_mask']
        label = label2id[ex['label']]
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attn, dtype=torch.long),
            'labels': torch.tensor(label, dtype=torch.long)
        }


def train_loop(model, loader, optimizer, scheduler, grad_accum, log_steps):
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()
    for step, batch in enumerate(tqdm(loader, desc='Training'), 1):
        outputs = model(
            input_ids=batch['input_ids'].to(device),
            attention_mask=batch['attention_mask'].to(device),
            labels=batch['labels'].to(device)
        )
        loss = outputs.loss / grad_accum
        loss.backward()
        if step % grad_accum == 0:
            clip_grad(model.parameters())
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        total_loss += loss.item() * grad_accum
        if step % log_steps == 0:
            avg = total_loss / step
            tqdm.write(f"Step {step}: avg loss={avg:.4f}")
    return total_loss / len(loader)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--model_id', type=str, default='ProsusAI/finbert')
    parser.add_argument('--adapter_dir', type=str, default='./finbert_lora_adapter')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--warmup_steps', type=int, default=50)
    parser.add_argument('--grad_accum', type=int, default=4)
    parser.add_argument('--log_steps', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    raw = load_raw_data(args.data_path)
    ds = raw.map(format_instruction_stock, remove_columns=raw.column_names)
    train_ds, val_ds, test_ds = get_splits(ds, seed=args.seed)

    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")

    tokenizer = load_tokenizer(args.model_id)
    train_loader = DataLoader(SequenceClassificationDataset(train_ds, tokenizer), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(SequenceClassificationDataset(val_ds, tokenizer), batch_size=args.batch_size)

    model = build_model(args.model_id, num_labels=3, device=device)
    optimizer = AdamW(model.parameters(), lr=args.lr)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=total_steps
    )

    for epoch in range(1, args.epochs + 1):
        loss = train_loop(
            model, train_loader, optimizer, scheduler,
            args.grad_accum, args.log_steps
        )
        print(f"Epoch {epoch} loss: {loss:.4f}")

    model.save_pretrained(args.adapter_dir)
    tokenizer.save_pretrained(args.adapter_dir)
    print(f"Saved adapter to {args.adapter_dir}")

if __name__ == '__main__':
    main()