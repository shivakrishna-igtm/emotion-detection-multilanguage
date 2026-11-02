"""
models/logistic_regression.py
TF-IDF + Logistic Regression baseline.
"""

from sklearn.linear_model import LogisticRegression
from utils.evaluate import compute_metrics


def train_logistic_regression(data: dict) -> dict:
    model = LogisticRegression(
        max_iter=1000,
        C=5.0,
        solver="lbfgs",
        multi_class="multinomial",
        n_jobs=-1,
    )
    model.fit(data["X_train_tfidf"], data["train_labels"])

    y_pred = model.predict(data["X_test_tfidf"])
    metrics = compute_metrics(data["test_labels"], y_pred)

    print(f"  [LR] Accuracy: {metrics['accuracy']:.4f}  F1: {metrics['f1']:.4f}")
    return {"model": model, "y_pred": y_pred, "y_true": data["test_labels"], "metrics": metrics}
