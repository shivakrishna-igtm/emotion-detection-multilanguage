"""
models/distilbert_model.py
Fine-tuned DistilBERT classifier via HuggingFace Transformers.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from torch.optim import AdamW
from utils.evaluate import compute_metrics

MODEL_NAME = "distilbert-base-uncased"
MAX_LEN    = 64
EPOCHS     = 4
BATCH_SIZE = 32
LR         = 2e-5


class EmotionDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        enc = tokenizer(
            texts, truncation=True, padding="max_length",
            max_length=MAX_LEN, return_tensors="pt"
        )
        self.input_ids      = enc["input_ids"]
        self.attention_mask = enc["attention_mask"]
        self.labels         = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids":      self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels":         self.labels[idx],
        }


def train_distilbert(data: dict) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)
    model = DistilBertForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=data["num_classes"]
    ).to(device)

    train_loader = DataLoader(
        EmotionDataset(data["train_texts"], data["train_labels"], tokenizer),
        batch_size=BATCH_SIZE, shuffle=True
    )
    val_loader = DataLoader(
        EmotionDataset(data["val_texts"], data["val_labels"], tokenizer),
        batch_size=BATCH_SIZE
    )
    test_loader = DataLoader(
        EmotionDataset(data["test_texts"], data["test_labels"], tokenizer),
        batch_size=BATCH_SIZE
    )

    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=total_steps // 10,
        num_training_steps=total_steps
    )

    best_val_acc, best_state = 0.0, None

    for epoch in range(1, EPOCHS + 1):
        # ── Train ──────────────────────────────────────────────
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            running_loss += loss.item()

        # ── Validate ───────────────────────────────────────────
        model.eval()
        val_preds, val_true = [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                logits = model(**batch).logits
                val_preds.extend(logits.argmax(-1).cpu().numpy())
                val_true.extend(batch["labels"].cpu().numpy())

        val_acc = (np.array(val_preds) == np.array(val_true)).mean()
        avg_loss = running_loss / len(train_loader)
        print(f"  Epoch {epoch}/{EPOCHS}  loss={avg_loss:.4f}  val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    # ── Test ───────────────────────────────────────────────────
    model.load_state_dict(best_state)
    model.eval()
    test_preds = []
    with torch.no_grad():
        for batch in test_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(**batch).logits
            test_preds.extend(logits.argmax(-1).cpu().numpy())

    y_pred = np.array(test_preds)
    metrics = compute_metrics(data["test_labels"], y_pred)
    print(f"  [DistilBERT] Accuracy: {metrics['accuracy']:.4f}  F1: {metrics['f1']:.4f}")

    return {
        "model": model,
        "y_pred": y_pred,
        "y_true": data["test_labels"],
        "metrics": metrics,
    }
