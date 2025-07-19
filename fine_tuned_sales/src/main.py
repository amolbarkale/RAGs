# src/main.py

import os
import argparse
import joblib
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression

# Resolve paths
SCRIPT_DIR   = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
DATA_PATH    = os.path.join(PROJECT_ROOT, "data", "sales_transcripts.csv")
MODEL_DIR    = os.path.join(PROJECT_ROOT, "models")

def load_models(use_contrastive: bool):
    """
    Load both encoders and the logistic classifier.
    """
    # 1) Baseline SBERT
    sbert_encoder = SentenceTransformer("all-MiniLM-L6-v2")

    # 2) Contrastive‑fine‑tuned BERT
    crt_encoder = SentenceTransformer(os.path.join(MODEL_DIR, "bert_contrastive"))

    # 3) Logistic Reg trained on *contrastive* embeddings?
    #    If you kept separate classifiers for each, load both. 
    #    Here, we'll assume you re‑trained a fresh LR on the fine‑tuned embeddings,
    #    and saved it as `clf_bert_contrastive.joblib`.
    baseline_clf = joblib.load(os.path.join(MODEL_DIR, "clf_sbert_base.joblib"))
    contrastive_clf = joblib.load(os.path.join(MODEL_DIR, "clf_bert_contrastive.joblib"))

    return (sbert_encoder, baseline_clf), (crt_encoder, contrastive_clf)

def classify_text(text: str, encoder, classifier):
    emb = encoder.encode([text])
    probs = classifier.predict_proba(emb)[0]
    labels = classifier.classes_
    idx = np.argmax(probs)
    return labels[idx], probs[idx]

def main():
    parser = argparse.ArgumentParser(description="Classify a sales call transcript")
    parser.add_argument("--text", help="Transcript text", required=True)
    parser.add_argument(
        "--model",
        choices=["sbert", "bert-contrastive", "both"],
        default="both",
        help="Which encoder+classifier to use"
    )
    args = parser.parse_args()

    (sbert_enc, sbert_clf), (crt_enc, crt_clf) = load_models(True)

    if args.model in ("sbert", "both"):
        label, conf = classify_text(args.text, sbert_enc, sbert_clf)
        print(f"[SBERT Baseline] Prediction: {label} (confidence {conf:.2f})")

    if args.model in ("bert-contrastive", "both"):
        label, conf = classify_text(args.text, crt_enc, crt_clf)
        print(f"[BERT Contrastive] Prediction: {label} (confidence {conf:.2f})")

if __name__ == "__main__":
    main()
