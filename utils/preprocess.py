"""
utils/preprocess.py
Load the dair-ai/emotion dataset and prepare splits for all three models.
No TensorFlow dependency — uses PyTorch-friendly tokenizer.
"""

import re
import numpy as np
from collections import Counter
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer

MAX_VOCAB   = 20_000
MAX_LEN     = 64
TFIDF_MAX   = 30_000


class SimpleTokenizer:
    """Lightweight word tokenizer — no TensorFlow dependency."""
    def __init__(self, num_words=MAX_VOCAB):
        self.num_words = num_words
        self.word_index = {}

    def fit_on_texts(self, texts):
        counts = Counter(w for t in texts for w in t.split())
        vocab = ["<PAD>", "<OOV>"] + [w for w, _ in counts.most_common(self.num_words - 2)]
        self.word_index = {w: i for i, w in enumerate(vocab)}

    def texts_to_sequences(self, texts):
        oov = self.word_index.get("<OOV>", 1)
        return [[self.word_index.get(w, oov) for w in t.split()] for t in texts]


def pad_sequences(seqs, maxlen=MAX_LEN):
    out = np.zeros((len(seqs), maxlen), dtype=np.int32)
    for i, s in enumerate(seqs):
        s = s[:maxlen]
        out[i, :len(s)] = s
    return out


def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_and_preprocess():
    dataset = load_dataset("dair-ai/emotion", trust_remote_code=True)

    def extract(split):
        texts  = [clean_text(t) for t in split["text"]]
        labels = list(split["label"])
        return texts, labels

    train_texts, train_labels = extract(dataset["train"])
    val_texts,   val_labels   = extract(dataset["validation"])
    test_texts,  test_labels  = extract(dataset["test"])

    num_classes = len(set(train_labels))
    print(f"  Classes : {num_classes}")
    print(f"  Train   : {len(train_texts)}")
    print(f"  Val     : {len(val_texts)}")
    print(f"  Test    : {len(test_texts)}")

    # TF-IDF for Logistic Regression
    tfidf = TfidfVectorizer(max_features=TFIDF_MAX, ngram_range=(1, 2))
    X_train_tfidf = tfidf.fit_transform(train_texts)
    X_val_tfidf   = tfidf.transform(val_texts)
    X_test_tfidf  = tfidf.transform(test_texts)

    # Integer sequences for LSTM
    tok = SimpleTokenizer(num_words=MAX_VOCAB)
    tok.fit_on_texts(train_texts)

    X_train_seq = pad_sequences(tok.texts_to_sequences(train_texts))
    X_val_seq   = pad_sequences(tok.texts_to_sequences(val_texts))
    X_test_seq  = pad_sequences(tok.texts_to_sequences(test_texts))

    return {
        "train_texts": train_texts, "val_texts": val_texts, "test_texts": test_texts,
        "train_labels": np.array(train_labels),
        "val_labels":   np.array(val_labels),
        "test_labels":  np.array(test_labels),
        "X_train_tfidf": X_train_tfidf,
        "X_val_tfidf":   X_val_tfidf,
        "X_test_tfidf":  X_test_tfidf,
        "X_train_seq": X_train_seq,
        "X_val_seq":   X_val_seq,
        "X_test_seq":  X_test_seq,
        "tokenizer":   tok,
        "tfidf":       tfidf,
        "num_classes": num_classes,
    }
