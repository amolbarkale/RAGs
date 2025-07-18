import streamlit as st
import joblib
import pandas as pd
import numpy as np

from embedding_models import (
    load_word_vectors,
    embed_avg_word2vec,
    embed_bert_cls,
    embed_sentence_bert,
)
from sklearn.preprocessing import LabelEncoder
from umap import UMAP
import plotly.express as px

# --- Load models & vectorizers once ---
@st.cache_resource
def load_resources():
    w2v = load_word_vectors("../models/GoogleNews-vectors-negative300.bin")
    clf_w2v    = joblib.load("../models/classifier_word2vec.joblib")
    clf_bert   = joblib.load("../models/classifier_bert_cls.joblib")
    clf_sbert  = joblib.load("../models/classifier_sentencebert.joblib")
    return w2v, clf_w2v, clf_bert, clf_sbert

w2v_model, clf_w2v, clf_bert, clf_sbert = load_resources()

st.title("🌟 Smart Article Categorizer")

article = st.text_area("Paste your article text here", height=200)
if st.button("Classify") and article.strip():
    # 1) Embed
    emb_w2v   = embed_avg_word2vec([article], w2v_model)
    emb_bert  = embed_bert_cls([article])
    emb_sbert = embed_sentence_bert([article])

    # 2) Predict & get probs
    models = {
        "Word2Vec": (clf_w2v, emb_w2v),
        "BERT [CLS]": (clf_bert, emb_bert),
        "Sentence‑BERT": (clf_sbert, emb_sbert),
    }

    results = {}
    for name, (clf, emb) in models.items():
        probs = clf.predict_proba(emb)[0]
        labels = clf.classes_
        top_idx = np.argmax(probs)
        results[name] = {
            "Prediction": labels[top_idx],
            "Confidence": float(probs[top_idx])
        }

    # 3) Show a summary table
    df = pd.DataFrame(results).T
    st.subheader("Model Predictions")
    st.dataframe(df)

    # 4) Bar chart of confidences
    st.subheader("Confidence Scores")
    conf_df = pd.DataFrame({
        model: clf.predict_proba(emb)[0]
        for model, (clf, emb) in models.items()
    }, index=clf_w2v.classes_)
    st.bar_chart(conf_df)

# --- Optional: Embedding Cluster Visualization ---
if st.checkbox("Show embedding clusters"):
    st.write("Projecting 100 articles into 2D via UMAP…")
    # Load a sample of your test set embeddings & true labels
    import joblib
    sample_texts = joblib.load("../results/sample_texts.joblib")
    sample_labels = joblib.load("../results/sample_labels.joblib")
    # Choose one embedding type (e.g. SBERT) for cluster viz
    embs = embed_sentence_bert(sample_texts)
    reducer = UMAP(n_components=2, random_state=42)
    proj   = reducer.fit_transform(embs)
    fig = px.scatter(
        x=proj[:,0], y=proj[:,1],
        color=sample_labels,
        title="Sentence‑BERT Embedding Clusters",
        width=700, height=500
    )
    st.plotly_chart(fig)
