"""
demo.py — interactive emotion predictor.
Run AFTER main.py has finished. Loads saved models instantly.
"""

import re, pickle, os
import numpy as np
import torch
import torch.nn as nn
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

LABELS = {
    0: "😢 Sadness",
    1: "😊 Joy",
    2: "❤️  Love",
    3: "😠 Anger",
    4: "😨 Fear",
    5: "😲 Surprise"
}

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    return re.sub(r"\s+", " ", text).strip()

def pad_sequence(seq, maxlen=64):
    seq = seq[:maxlen]
    return np.array(seq + [0] * (maxlen - len(seq)), dtype=np.int32)

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


def load_models():
    if not os.path.exists("saved_models/lr_model.pkl"):
        print("\n❌ No saved models found. Please run main.py first!\n")
        exit()

    print("  Loading models", end="", flush=True)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    with open("saved_models/tokenizer.pkl", "rb") as f:
        tok = pickle.load(f)
    print(".", end="", flush=True)

    with open("saved_models/tfidf.pkl", "rb") as f:
        tfidf = pickle.load(f)
    print(".", end="", flush=True)

    with open("saved_models/lr_model.pkl", "rb") as f:
        lr_model = pickle.load(f)
    print(".", end="", flush=True)

    with open("saved_models/lstm_vocab_size.pkl", "rb") as f:
        vocab_size = pickle.load(f)
    lstm_model = BiLSTMClassifier(vocab_size).to(device)
    lstm_model.load_state_dict(torch.load("saved_models/lstm_weights.pt", map_location=device))
    lstm_model.eval()
    print(".", end="", flush=True)

    bert_tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    bert_model = DistilBertForSequenceClassification.from_pretrained("saved_models/distilbert").to(device)
    bert_model.eval()
    print(". ✅ Ready!\n")

    return lr_model, lstm_model, bert_model, bert_tokenizer, tok, tfidf, device


def predict_all(text, lr_model, lstm_model, bert_model, bert_tokenizer, tok, tfidf, device):
    cleaned = clean_text(text)

    # Logistic Regression
    vec = tfidf.transform([cleaned])
    lr_pred  = lr_model.predict(vec)[0]
    lr_proba = lr_model.predict_proba(vec)[0]

    # LSTM
    oov = tok.word_index.get("<OOV>", 1)
    seq = [tok.word_index.get(w, oov) for w in cleaned.split()]
    padded = torch.tensor([pad_sequence(seq)], dtype=torch.long).to(device)
    with torch.no_grad():
        logits = lstm_model(padded)
        lstm_proba = torch.softmax(logits, dim=-1).cpu().numpy()[0]
    lstm_pred = int(np.argmax(lstm_proba))

    # DistilBERT
    enc = bert_tokenizer(cleaned, return_tensors="pt",
                         truncation=True, padding="max_length", max_length=64)
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        logits = bert_model(**enc).logits
        bert_proba = torch.softmax(logits, dim=-1).cpu().numpy()[0]
    bert_pred = int(np.argmax(bert_proba))

    return (lr_pred, lr_proba), (lstm_pred, lstm_proba), (bert_pred, bert_proba)


def show_result(model_name, pred, proba):
    label      = LABELS[pred]
    confidence = proba[pred] * 100
    bar_len    = int(confidence / 5)
    bar        = "█" * bar_len + "░" * (20 - bar_len)
    print(f"  {model_name:<22}  {label}  ({confidence:.1f}%)")
    print(f"  {'':22}  [{bar}]")


def main():
    print("\n" + "="*56)
    print("    EMOTION DETECTION — INTERACTIVE DEMO")
    print("="*56)
    (lr_model, lstm_model, bert_model,
     bert_tokenizer, tok, tfidf, device) = load_models()

    print("  Type any sentence to detect its emotion.")
    print("  Type 'quit' to exit.")
    print("="*56)

    while True:
        print()
        user_input = input("  📝 Enter text: ").strip()
        if user_input.lower() in ("quit", "exit", "q"):
            print("\n  Goodbye! 👋\n")
            break
        if not user_input:
            continue

        (lr_pred,   lr_proba), \
        (lstm_pred, lstm_proba), \
        (bert_pred, bert_proba) = predict_all(
            user_input, lr_model, lstm_model, bert_model,
            bert_tokenizer, tok, tfidf, device
        )

        print(f"\n  Input : \"{user_input}\"")
        print("  " + "─"*52)
        show_result("Logistic Regression", lr_pred,   lr_proba)
        print()
        show_result("LSTM",                lstm_pred, lstm_proba)
        print()
        show_result("DistilBERT ⭐",        bert_pred, bert_proba)
        print()

        preds = [lr_pred, lstm_pred, bert_pred]
        if len(set(preds)) == 1:
            print(f"  ✅ All 3 models agree → {LABELS[bert_pred]}")
        else:
            print(f"  ⚠️  Models disagree  → Trust DistilBERT: {LABELS[bert_pred]}")
        print("  " + "="*52)


if __name__ == "__main__":
    main()
