# src/baseline.py

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sentence_transformers import SentenceTransformer
import joblib

# Resolve paths relative to this file
SCRIPT_DIR   = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
DATA_PATH    = os.path.join(PROJECT_ROOT, "data", "sales_transcripts.csv")

def run_sbert_baseline():
    """
    Train and evaluate a baseline classifier using generic SBERT embeddings.
    """
    # 1) Load the CSV
    df = pd.read_csv(DATA_PATH)
    all_transcripts = df["transcript"].tolist()
    all_labels      = df["label"].tolist()

    # 2) Stratified 80/20 train/test split
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        all_transcripts,
        all_labels,
        test_size=0.2,
        stratify=all_labels,
        random_state=42
    )

    # 3) Compute SBERT embeddings
    sbert_model      = SentenceTransformer("all-MiniLM-L6-v2")
    train_embeddings = sbert_model.encode(train_texts, show_progress_bar=True)
    test_embeddings  = sbert_model.encode(test_texts,  show_progress_bar=True)

    # 4) Fit a Logistic Regression classifier
    baseline_classifier = LogisticRegression(max_iter=500)
    baseline_classifier.fit(train_embeddings, train_labels)

    # 5) Predict & print metrics
    test_predictions = baseline_classifier.predict(test_embeddings)
    print("=== Baseline SBERT Classification Report ===")
    print(classification_report(test_labels, test_predictions))

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(
        baseline_classifier,
        os.path.join(MODEL_DIR, "clf_sbert_base.joblib")
    )
    print(f"✅ Saved SBERT classifier to models/clf_sbert_base.joblib")


if __name__ == "__main__":
    run_sbert_baseline()
