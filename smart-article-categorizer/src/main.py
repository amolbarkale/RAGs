# src/main.py

import argparse
import os
import joblib
import numpy as np
from embedding_models import (
    load_word_vectors,
    embed_avg_word2vec,
    embed_bert_cls,
    embed_sentence_bert,
    embed_openai
)


def classify_text(text: str, models_dir: str):
    """
    Load saved models and classify a single text across all embedding types.
    """
    # Load embeddings & classifiers
    w2v = load_word_vectors(os.path.join(models_dir, "GoogleNews-vectors-negative300.bin"))
    clf_w2v    = joblib.load(os.path.join(models_dir, "clf_w2v.joblib"))
    clf_bert   = joblib.load(os.path.join(models_dir, "clf_bert.joblib"))
    clf_sbert  = joblib.load(os.path.join(models_dir, "clf_sbert.joblib"))
    clf_openai = joblib.load(os.path.join(models_dir, "clf_openai.joblib"))

    # Generate embeddings
    emb_w2v   = embed_avg_word2vec([text], w2v)
    emb_bert  = embed_bert_cls([text])
    emb_sbert = embed_sentence_bert([text])
    emb_oa    = embed_openai([text])

    # Predict probabilities
    probs = {
        'Word2Vec': clf_w2v.predict_proba(emb_w2v)[0],
        'BERT':     clf_bert.predict_proba(emb_bert)[0],
        'SBERT':    clf_sbert.predict_proba(emb_sbert)[0],
        'OpenAI':   clf_openai.predict_proba(emb_oa)[0],
    }
    labels = clf_w2v.classes_

    # Display top prediction per model
    for model_name, prob in probs.items():
        idx = np.argmax(prob)
        print(f"{model_name}: {labels[idx]} (confidence {prob[idx]:.2f})")


def main():
    parser = argparse.ArgumentParser(
        description="Classify an article into predefined categories."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        '--text', help='Article text to classify.'
    )
    group.add_argument(
        '--file', help='Path to a text file containing the article.'
    )
    parser.add_argument(
        '--models-dir',
        default=os.path.join(os.path.dirname(__file__), '../models'),
        help='Directory where trained models are saved.'
    )
    args = parser.parse_args()

    if args.file:
        with open(args.file, 'r', encoding='utf-8') as f:
            content = f.read()
    else:
        content = args.text

    classify_text(content, args.models_dir)


if __name__ == '__main__':
    main()
