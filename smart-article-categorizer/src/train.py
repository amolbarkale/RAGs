# src/train.py

import os
import joblib
import numpy as np
import pandas as pd

from data_loader import load_hf_datasets
from embedding_models import (
    load_word_vectors,
    embed_avg_word2vec,
    embed_bert_cls,
    embed_sentence_bert,
    # embed_openai
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def evaluate_model_performance(classifier, test_embeddings: np.ndarray, test_labels: list):
    """
    Evaluate a trained classifier on test data.

    Returns:
        Tuple of (accuracy, precision, recall, f1_score)
    """
    predictions = classifier.predict(test_embeddings)
    accuracy = accuracy_score(test_labels, predictions)
    precision, recall, f1_score, _ = precision_recall_fscore_support(
        test_labels, predictions, average="weighted"
    )
    return accuracy, precision, recall, f1_score


def main():
    # Project paths
    project_root = os.path.dirname(__file__)
    word2vec_file = os.path.join(project_root, "../models/GoogleNews-vectors-negative300.bin")
    models_output_dir = os.path.join(project_root, "../models")
    os.makedirs(models_output_dir, exist_ok=True)

    # Load and split data from HuggingFace datasets
    training_texts, testing_texts, training_labels, testing_labels = load_hf_datasets()

    performance_records = []

    # 1) Word2Vec Embeddings + Classifier
    print("[Training] Word2Vec embeddings and classifier")
    word2vec_model = load_word_vectors(word2vec_file)
    train_w2v_embeddings = embed_avg_word2vec(training_texts, word2vec_model)
    test_w2v_embeddings  = embed_avg_word2vec(testing_texts,  word2vec_model)

    w2v_classifier = LogisticRegression(max_iter=1000)
    w2v_classifier.fit(train_w2v_embeddings, training_labels)

    w2v_metrics = evaluate_model_performance(
        w2v_classifier, test_w2v_embeddings, testing_labels
    )
    performance_records.append(("Word2Vec", *w2v_metrics))
    joblib.dump(
        w2v_classifier,
        os.path.join(models_output_dir, "classifier_word2vec.joblib")
    )

    # 2) BERT [CLS] Token Embeddings + Classifier
    print("[Training] BERT [CLS] embeddings and classifier")
    train_bert_embeddings = embed_bert_cls(training_texts)
    test_bert_embeddings  = embed_bert_cls(testing_texts)

    bert_classifier = LogisticRegression(max_iter=1000)
    bert_classifier.fit(train_bert_embeddings, training_labels)

    bert_metrics = evaluate_model_performance(
        bert_classifier, test_bert_embeddings, testing_labels
    )
    performance_records.append(("BERT_CLS", *bert_metrics))
    joblib.dump(
        bert_classifier,
        os.path.join(models_output_dir, "classifier_bert_cls.joblib")
    )

    # 3) Sentence-BERT Embeddings + Classifier
    print("[Training] Sentence-BERT embeddings and classifier")
    train_sbert_embeddings = embed_sentence_bert(training_texts)
    test_sbert_embeddings  = embed_sentence_bert(testing_texts)

    sbert_classifier = LogisticRegression(max_iter=1000)
    sbert_classifier.fit(train_sbert_embeddings, training_labels)

    sbert_metrics = evaluate_model_performance(
        sbert_classifier, test_sbert_embeddings, testing_labels
    )
    performance_records.append(("SentenceBERT", *sbert_metrics))
    joblib.dump(
        sbert_classifier,
        os.path.join(models_output_dir, "classifier_sentencebert.joblib")
    )

    # 4) OpenAI Ada Embeddings + Classifier
    # print("[Training] OpenAI text-embedding-ada-002 embeddings and classifier")
    # train_oa_embeddings = embed_openai(training_texts)
    # test_oa_embeddings  = embed_openai(testing_texts)

    # openai_classifier = LogisticRegression(max_iter=1000)
    # openai_classifier.fit(train_oa_embeddings, training_labels)

    # openai_metrics = evaluate_model_performance(
    #     openai_classifier, test_oa_embeddings, testing_labels
    # )
    # performance_records.append(("OpenAI_Ada", *openai_metrics))
    # joblib.dump(
    #     openai_classifier,
    #     os.path.join(models_output_dir, "classifier_openai_ada.joblib")
    # )

    # Save performance report to CSV
    performance_df = pd.DataFrame(
        performance_records,
        columns=["EmbeddingType", "Accuracy", "Precision", "Recall", "F1Score"]
    )
    report_file = os.path.join(project_root, "../results/embedding_performance.csv")
    os.makedirs(os.path.dirname(report_file), exist_ok=True)
    performance_df.to_csv(report_file, index=False)

    print(f"Training complete. Performance report saved to: {report_file}")


if __name__ == "__main__":
    main()