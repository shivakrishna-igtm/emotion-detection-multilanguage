"""
main.py — trains all 3 models and saves them for the demo.
"""

import warnings, os, pickle
warnings.filterwarnings("ignore")
import torch

from utils.preprocess import load_and_preprocess
from models.logistic_regression import train_logistic_regression
from models.lstm_model import train_lstm
from models.distilbert_model import train_distilbert
from utils.evaluate import compare_models, plot_confusion_matrix, save_results

LABEL_NAMES = ["sadness", "joy", "love", "anger", "fear", "surprise"]
os.makedirs("saved_models", exist_ok=True)

def main():
    print("="*60)
    print("  EMOTION DETECTION - DEEP LEARNING COMPARISON")
    print("="*60)

    print("\n[1/4] Loading and preprocessing dataset...")
    data = load_and_preprocess()

    # Save tokenizer and tfidf for demo
    with open("saved_models/tokenizer.pkl", "wb") as f:
        pickle.dump(data["tokenizer"], f)
    with open("saved_models/tfidf.pkl", "wb") as f:
        pickle.dump(data["tfidf"], f)

    print("\n[2/4] Training Logistic Regression (baseline)...")
    lr_results = train_logistic_regression(data)
    with open("saved_models/lr_model.pkl", "wb") as f:
        pickle.dump(lr_results["model"], f)

    print("\n[3/4] Training LSTM model...")
    lstm_results = train_lstm(data)
    torch.save(lstm_results["model"].state_dict(), "saved_models/lstm_weights.pt")
    with open("saved_models/lstm_vocab_size.pkl", "wb") as f:
        pickle.dump(len(data["tokenizer"].word_index), f)

    print("\n[4/4] Training DistilBERT model...")
    bert_results = train_distilbert(data)
    bert_results["model"].save_pretrained("saved_models/distilbert")

    all_results = {
        "Logistic Regression": lr_results,
        "LSTM":                lstm_results,
        "DistilBERT":          bert_results,
    }

    print("\n" + "="*60)
    print("  FINAL RESULTS")
    print("="*60)
    compare_models(all_results)

    for name, res in all_results.items():
        plot_confusion_matrix(res["y_true"], res["y_pred"], LABEL_NAMES, name)

    save_results(all_results)
    print("\n✅ Models saved to saved_models/")
    print("✅ Run demo.py to test with your own sentences!\n")

if __name__ == "__main__":
    main()
