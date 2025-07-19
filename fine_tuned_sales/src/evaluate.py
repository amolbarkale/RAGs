from sentence_transformers import SentenceTransformer
# …
def main():
    texts, labels = load_data()

    # 1) Generic SBERT baseline
    sbert = SentenceTransformer("all-MiniLM-L6-v2")
    eval_model(sbert, "SBERT-Base", texts, labels)

    # 2) Contrastive‑fine‑tuned BERT
    bert_ft = SentenceTransformer("../models/bert_contrastive")
    eval_model(bert_ft, "BERT-Contrastive", texts, labels)
