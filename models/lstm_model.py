"""
models/lstm_model.py
Bidirectional LSTM using PyTorch (M1/M2/M3 compatible).
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
from utils.evaluate import compute_metrics

EMBED_DIM  = 128
LSTM_UNITS = 128
DROPOUT    = 0.4
EPOCHS     = 10
BATCH_SIZE = 64
LR         = 1e-3


class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, lstm_units=128, num_classes=6, dropout=0.4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm1 = nn.LSTM(embed_dim, lstm_units, batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(lstm_units * 2, lstm_units // 2, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(lstm_units, 64)
        self.fc2 = nn.Linear(64, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x = x[:, -1, :]
        x = self.dropout(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)


def train_lstm(data: dict) -> dict:
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"  Device: {device}")

    # FIXED: use actual vocabulary size from word_index
    # Previously used: data["tokenizer"].num_words or 20_000
    # That returned the cap (20,000) not actual words found (~15,000)
    # causing empty embedding rows and collapsed predictions
    vocab_size  = len(data["tokenizer"].word_index)
    num_classes = data["num_classes"]

    def to_loader(X, y, shuffle=False):
        ds = TensorDataset(
            torch.tensor(X, dtype=torch.long),
            torch.tensor(y, dtype=torch.long)
        )
        return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle)

    train_loader = to_loader(data["X_train_seq"], data["train_labels"], shuffle=True)
    val_loader   = to_loader(data["X_val_seq"],   data["val_labels"])
    test_loader  = to_loader(data["X_test_seq"],  data["test_labels"])

    model     = BiLSTMClassifier(vocab_size, EMBED_DIM, LSTM_UNITS, num_classes, DROPOUT).to(device)
    optimizer = Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    best_val_acc, best_state, patience_count = 0.0, None, 0
    PATIENCE = 3

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X_batch), y_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        model.eval()
        val_preds, val_true = [], []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                logits = model(X_batch.to(device))
                val_preds.extend(logits.argmax(-1).cpu().numpy())
                val_true.extend(y_batch.numpy())

        val_acc = (np.array(val_preds) == np.array(val_true)).mean()
        print(f"  Epoch {epoch}/{EPOCHS}  loss={running_loss/len(train_loader):.4f}  val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state   = {k: v.clone() for k, v in model.state_dict().items()}
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= PATIENCE:
                print(f"  Early stopping at epoch {epoch}")
                break

    model.load_state_dict(best_state)
    model.eval()
    test_preds = []
    with torch.no_grad():
        for X_batch, _ in test_loader:
            logits = model(X_batch.to(device))
            test_preds.extend(logits.argmax(-1).cpu().numpy())

    y_pred  = np.array(test_preds)
    metrics = compute_metrics(data["test_labels"], y_pred)
    print(f"  [LSTM] Accuracy: {metrics['accuracy']:.4f}  F1: {metrics['f1']:.4f}")

    return {
        "model":   model,
        "y_pred":  y_pred,
        "y_true":  data["test_labels"],
        "metrics": metrics,
    }
