import numpy as np
from typing import List

# --- Word2Vec / GloVe (Average Word Vectors) ---
from gensim.models import KeyedVectors

def load_word_vectors(path: str) -> KeyedVectors:
    """
    Load pre-trained Word2Vec/GloVe vectors in word2vec format.
    """
    return KeyedVectors.load_word2vec_format(path, binary=True)


def embed_avg_word2vec(texts: List[str], w2v_model: KeyedVectors) -> np.ndarray:
    """
    Compute document embeddings by averaging word vectors.

    Args:
        texts: List of raw text documents
        w2v_model: Loaded KeyedVectors model
    Returns:
        2D numpy array of shape (n_texts, vector_dim)
    """
    vectors = []
    for doc in texts:
        tokens = [tok for tok in doc.lower().split() if tok in w2v_model]
        if tokens:
            vec = np.mean([w2v_model[tok] for tok in tokens], axis=0)
        else:
            vec = np.zeros(w2v_model.vector_size, dtype=float)
        vectors.append(vec)
    return np.vstack(vectors)


# --- BERT [CLS] Embeddings ---
from transformers import AutoTokenizer, AutoModel
import torch

tokenizer_bert = AutoTokenizer.from_pretrained("bert-base-uncased")
model_bert     = AutoModel.from_pretrained("bert-base-uncased")

def embed_bert_cls(texts: List[str], max_length: int = 512) -> np.ndarray:
    """
    Use BERT to get [CLS] token embedding for each text.

    Args:
        texts: List of raw text documents
        max_length: Maximum token length for truncation
    Returns:
        2D numpy array of shape (n_texts, hidden_size)
    """
    model_bert.eval()
    with torch.no_grad():
        encoded = tokenizer_bert(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
        outputs = model_bert(**encoded)
        cls_embeddings = outputs.last_hidden_state[:, 0, :]
    return cls_embeddings.cpu().numpy()


# --- Sentence-BERT (all-MiniLM-L6-v2) ---
from sentence_transformers import SentenceTransformer

sbert_model = SentenceTransformer("all-MiniLM-L6-v2")

def embed_sentence_bert(texts: List[str]) -> np.ndarray:
    """
    Generate embeddings using a Sentence-BERT model.

    Args:
        texts: List of raw text documents
    Returns:
        2D numpy array of shape (n_texts, embedding_dim)
    """
    return sbert_model.encode(texts, batch_size=32, show_progress_bar=True)


# --- OpenAI Embeddings (text-embedding-ada-002) via LangChain ---
from langchain.embeddings import OpenAIEmbeddings

openai_embedder = OpenAIEmbeddings(model="text-embedding-ada-002")

def embed_openai(texts: List[str]) -> np.ndarray:
    """
    Use OpenAI's text-embedding-ada-002 model for document embeddings.

    Args:
        texts: List of raw text documents
    Returns:
        2D numpy array of shape (n_texts, embedding_dim)
    """
    embeddings = openai_embedder.embed_documents(texts)
    return np.array(embeddings)
