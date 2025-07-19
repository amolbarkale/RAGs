# src/streamlit_app.py

import os
import streamlit as st
import joblib
import numpy as np
from sentence_transformers import SentenceTransformer

# ----------- Path setup -----------------------------------------------------------
SCRIPT_DIR    = os.path.dirname(__file__)
PROJECT_ROOT  = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
MODELS_DIR    = os.path.join(PROJECT_ROOT, "models")

# These must match exactly what your training scripts wrote:
SBERT_CLF_FILE = "clf_sbert_base.joblib"
BERT_CLF_FILE  = "clf_bert_contrastive.joblib"
BERT_EMB_DIR   = "bert_contrastive"

SBERT_CLF_PATH = os.path.join(MODELS_DIR, SBERT_CLF_FILE)
BERT_CLF_PATH  = os.path.join(MODELS_DIR, BERT_CLF_FILE)
BERT_EMB_PATH  = os.path.join(MODELS_DIR, BERT_EMB_DIR)

# ----------- Streamlit UI ---------------------------------------------------------
st.set_page_config(page_title="Sales Conversion Predictor", layout="centered")
st.title("🔮 Sales Conversion Predictor")
st.write("Paste your sales call transcript below and click **Predict**.")

transcript = st.text_area("Call transcript", height=200)

# ----------- Model availability checks -------------------------------------------
col1, col2 = st.columns(2)

with col1:
    if not os.path.isfile(SBERT_CLF_PATH):
        st.warning(f"⚠️ SBERT classifier not found at `{SBERT_CLF_PATH}`.  \n"
                   "Run `python src/baseline.py` to generate it.")
    else:
        st.success("✅ SBERT classifier loaded")

with col2:
    if not (os.path.isdir(BERT_EMB_PATH) and os.path.isfile(BERT_CLF_PATH)):
        st.warning(f"⚠️ BERT-contrastive model or classifier missing.  \n"
                   "Run `python src/contrastive_finetune_llama.py` and then "
                   "`python src/evaluate.py` to generate them.")
    else:
        st.success("✅ BERT-contrastive model loaded")

# ----------- Load only the available ones ----------------------------------------
@st.cache_resource
def load_sbert_pipeline():
    encoder   = SentenceTransformer("all-MiniLM-L6-v2")
    classifier = joblib.load(SBERT_CLF_PATH)
    return encoder, classifier

@st.cache_resource
def load_bert_contrastive_pipeline():
    encoder   = SentenceTransformer(BERT_EMB_PATH)
    classifier = joblib.load(BERT_CLF_PATH)
    return encoder, classifier

# ----------- Prediction function -------------------------------------------------
def predict(text, encoder, classifier):
    emb   = encoder.encode([text])
    probs = classifier.predict_proba(emb)[0]
    idx   = np.argmax(probs)
    return classifier.classes_[idx], float(probs[idx])

# ----------- Run predictions on button click -------------------------------------
if st.button("Predict") and transcript.strip():
    
    predictions_made = False
    
    # SBERT baseline
    if os.path.isfile(SBERT_CLF_PATH):
        try:
            sbert_enc, sbert_clf = load_sbert_pipeline()
            sb_label, sb_conf    = predict(transcript, sbert_enc, sbert_clf)
            st.write(f"**SBERT Baseline** → {sb_label} _(confidence: {sb_conf:.2f})_")
            predictions_made = True
        except Exception as e:
            st.error(f"Error loading SBERT model: {str(e)}")

    # BERT-contrastive
    if os.path.isdir(BERT_EMB_PATH) and os.path.isfile(BERT_CLF_PATH):
        try:
            bert_enc, bert_clf = load_bert_contrastive_pipeline()
            bt_label, bt_conf  = predict(transcript, bert_enc, bert_clf)
            st.write(f"**BERT-Contrastive** → {bt_label} _(confidence: {bt_conf:.2f})_")
            predictions_made = True
        except Exception as e:
            st.error(f"Error loading BERT-contrastive model: {str(e)}")
    
    # If no models are available, show instructions
    if not predictions_made:
        st.info("🚀 **Get Started:**")
        st.write("1. Train a baseline model: `python src/baseline.py`")
        st.write("2. Train a contrastive model: `python src/contrastive_finetune_llama.py`")
        st.write("3. Evaluate models: `python src/evaluate.py`")
        st.write("4. Refresh this page to see predictions!")
