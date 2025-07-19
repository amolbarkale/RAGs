# src/evaluate.py

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
MODELS_DIR   = os.path.join(PROJECT_ROOT, "models")
BERT_MODEL_PATH = os.path.join(MODELS_DIR, "bert_contrastive")

def load_data():
    """Load the sales transcripts dataset."""
    df = pd.read_csv(DATA_PATH)
    texts = df["transcript"].tolist()
    labels = df["label"].tolist()
    return texts, labels

def eval_model(model, model_name, texts, labels):
    """Evaluate a sentence transformer model and save the classifier."""
    print(f"\n=== Evaluating {model_name} ===")
    
    # Split data
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts, labels, test_size=0.2, stratify=labels, random_state=42
    )
    
    # Generate embeddings
    print("Generating embeddings...")
    train_embeddings = model.encode(train_texts, show_progress_bar=True)
    test_embeddings = model.encode(test_texts, show_progress_bar=True)
    
    # Train classifier
    classifier = LogisticRegression(max_iter=500)
    classifier.fit(train_embeddings, train_labels)
    
    # Evaluate
    test_predictions = classifier.predict(test_embeddings)
    print(f"\n{model_name} Classification Report:")
    print(classification_report(test_labels, test_predictions))
    
    # Save classifier
    if model_name == "BERT-Contrastive":
        os.makedirs(MODELS_DIR, exist_ok=True)
        classifier_path = os.path.join(MODELS_DIR, "clf_bert_contrastive.joblib")
        joblib.dump(classifier, classifier_path)
        print(f"✅ {model_name} classifier saved to: {classifier_path}")

def main():
    """Main evaluation function."""
    texts, labels = load_data()

    # 1) Generic SBERT baseline
    print("Loading SBERT baseline model...")
    sbert = SentenceTransformer("all-MiniLM-L6-v2")
    eval_model(sbert, "SBERT-Base", texts, labels)

    # 2) Contrastive-fine-tuned BERT
    print(f"\nLoading BERT-Contrastive model from {BERT_MODEL_PATH}...")
    if os.path.exists(BERT_MODEL_PATH):
        bert_ft = SentenceTransformer(BERT_MODEL_PATH)
        eval_model(bert_ft, "BERT-Contrastive", texts, labels)
    else:
        print(f"❌ BERT-Contrastive model not found at {BERT_MODEL_PATH}")
        print("Run contrastive_finetune_llama.py first to train the model.")

if __name__ == "__main__":
    main()
