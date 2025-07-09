# preprocessing + cosine similarity

from sklearn.metrics.pairwise import cosine_similarity

def preprocess_text(text):
    """Simple text preprocessing"""
    return text.strip().lower()

def calculate_similarity_matrix(texts, model):
    """Calculate pairwise similarity matrix"""
    processed_texts = [preprocess_text(text) for text in texts]
    embeddings = model.encode(processed_texts)
    similarity_matrix = cosine_similarity(embeddings)
    similarity_percentage = (similarity_matrix * 100).round(2)
    return similarity_percentage
