import os
import glob
import joblib
import numpy as np
import pandas as pd
from data_loader import load_and_prepare_data
from embedding_models import (
    load_word_vectors,
    embed_avg_word2vec,
    embed_bert_cls,
    embed_sentence_bert,
    # embed_openai
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def evaluate(clf, X_test, y_test):
    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_test, preds, average="weighted"
    )
    return acc, prec, rec, f1


def main():
    # Paths
    base_dir = os.path.dirname(__file__)
    data_pattern = os.path.join(base_dir, "../data/*.csv")
    w2v_path = os.path.join(base_dir, "../models/GoogleNews-vectors-negative300.bin")
    output_dir = os.path.join(base_dir, "../models")
    os.makedirs(output_dir, exist_ok=True)

    # Load and split data
    X_train, X_test, y_train, y_test = load_and_prepare_data(data_pattern)

    results = []

    # Word2Vec
    print("[Train] Word2Vec embeddings")
    w2v = load_word_vectors(w2v_path)
    X_tr_w2v = embed_avg_word2vec(X_train, w2v)
    X_te_w2v = embed_avg_word2vec(X_test,  w2v)
    clf_w2v = LogisticRegression(max_iter=1000)
    clf_w2v.fit(X_tr_w2v, y_train)
    metrics = evaluate(clf_w2v, X_te_w2v, y_test)
    results.append(("Word2Vec", *metrics))
    joblib.dump(clf_w2v, os.path.join(output_dir, "clf_w2v.joblib"))

    # BERT [CLS]
    print("[Train] BERT [CLS] embeddings")
    X_tr_bert = embed_bert_cls(X_train)
    X_te_bert = embed_bert_cls(X_test)
    clf_bert = LogisticRegression(max_iter=1000)
    clf_bert.fit(X_tr_bert, y_train)
    metrics = evaluate(clf_bert, X_te_bert, y_test)
    results.append(("BERT", *metrics))
    joblib.dump(clf_bert, os.path.join(output_dir, "clf_bert.joblib"))

    # Sentence-BERT
    print("[Train] Sentence-BERT embeddings")
    X_tr_sbert = embed_sentence_bert(X_train)
    X_te_sbert = embed_sentence_bert(X_test)
    clf_sbert = LogisticRegression(max_iter=1000)
    clf_sbert.fit(X_tr_sbert, y_train)
    metrics = evaluate(clf_sbert, X_te_sbert, y_test)
    results.append(("SBERT", *metrics))
    joblib.dump(clf_sbert, os.path.join(output_dir, "clf_sbert.joblib"))

    # OpenAI
    # print("[Train] OpenAI embeddings")
    # X_tr_oa = embed_openai(X_train)
    # X_te_oa = embed_openai(X_test)
    # clf_oa = LogisticRegression(max_iter=1000)
    # clf_oa.fit(X_tr_oa, y_train)
    # metrics = evaluate(clf_oa, X_te_oa, y_test)
    # results.append(("OpenAI", *metrics))
    # joblib.dump(clf_oa, os.path.join(output_dir, "clf_openai.joblib"))

    # Save performance report
    df = pd.DataFrame(
        results,
        columns=["Model", "Accuracy", "Precision", "Recall", "F1"]
    )
    report_path = os.path.join(base_dir, "../results/performance_comparison.csv")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    df.to_csv(report_path, index=False)
    print("Training complete. Results saved to:", report_path)


if __name__ == "__main__":
    main()
