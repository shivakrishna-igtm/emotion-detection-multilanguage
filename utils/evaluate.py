"""
utils/evaluate.py
Shared evaluation helpers: metrics, confusion matrices, result saving.
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, confusion_matrix
)

os.makedirs("results", exist_ok=True)

LABEL_NAMES = ["sadness", "joy", "love", "anger", "fear", "surprise"]


def compute_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    return {"accuracy": acc, "precision": p, "recall": r, "f1": f1}


def compare_models(all_results: dict):
    print(f"\n{'Model':<25} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print("-" * 65)
    for name, res in all_results.items():
        m = res["metrics"]
        print(
            f"{name:<25} {m['accuracy']:>10.4f} {m['precision']:>10.4f}"
            f" {m['recall']:>10.4f} {m['f1']:>10.4f}"
        )

    # Bar chart comparison
    models = list(all_results.keys())
    metrics = ["accuracy", "precision", "recall", "f1"]
    x = np.arange(len(models))
    width = 0.2

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, metric in enumerate(metrics):
        vals = [all_results[m]["metrics"][metric] for m in models]
        ax.bar(x + i * width, vals, width, label=metric.capitalize())

    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(models)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison — Emotion Detection")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/model_comparison.png", dpi=150)
    plt.close()
    print("\nBar chart saved → results/model_comparison.png")


def plot_confusion_matrix(y_true, y_pred, label_names, model_name: str):
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm_norm, annot=True, fmt=".2f", cmap="Blues",
        xticklabels=label_names, yticklabels=label_names, ax=ax
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix — {model_name}")
    plt.tight_layout()
    safe_name = model_name.lower().replace(" ", "_")
    path = f"results/cm_{safe_name}.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Confusion matrix saved → {path}")


def save_results(all_results: dict):
    out = {}
    for name, res in all_results.items():
        out[name] = {k: float(v) for k, v in res["metrics"].items()}
    with open("results/metrics.json", "w") as f:
        json.dump(out, f, indent=2)
    print("Metrics JSON saved → results/metrics.json")
