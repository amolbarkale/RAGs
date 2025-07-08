 # model loading

from sentence_transformers import SentenceTransformer
import streamlit as st

def load_models():
    """Load different embedding models"""
    try:
        models = {
            'MiniLM (Fast)': SentenceTransformer('all-MiniLM-L6-v2'),
            'MPNet (Balanced)': SentenceTransformer('all-mpnet-base-v2'),
        }
        return models
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return {'MiniLM (Fast)': SentenceTransformer('all-MiniLM-L6-v2')}
