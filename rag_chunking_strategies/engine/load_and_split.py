import nltk

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import sent_tokenize
from typing import List
import numpy as np
import os

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

def load_pdf(file_path):
    """Load PDF and return full text"""
    loader = PyPDFLoader(str(file_path))
    docs = loader.load()
    full_text = "\n".join([doc.page_content for doc in docs])
    return full_text

def recursive_chunking(text, chunk_size=1000, chunk_overlap=200):
    """Split text with RecursiveCharacterTextSplitter"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    return splitter.split_text(text)

def markdown_aware_chunking(text, max_chunk_size=1000):
    """Chunk markdown text preserving headers and structure"""
    sections = text.split('\n## ')
    chunks = []
    for section in sections:
        if len(section) <= max_chunk_size:
            chunks.append(section)
        else:
            paras = section.split('\n\n')
            current = ""
            for para in paras:
                if len(current) + len(para) <= max_chunk_size:
                    current += para + "\n\n"
                else:
                    chunks.append(current.strip())
                    current = para + "\n\n"
            if current:
                chunks.append(current.strip())
    return chunks

def semantic_chunking(text: str, similarity_threshold: float = 0.6) -> List[str]:
    """Chunk text based on semantic similarity between sentences using Gemini Embeddings"""
    sentences = sent_tokenize(text)

    embedding_model = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=os.getenv("GEMINI_API_KEY")
    )

    embeddings = embedding_model.embed_documents(sentences)

    chunks = []
    current_chunk = [sentences[0]]

    for i in range(1, len(sentences)):
        sim = cosine_similarity(
            [embeddings[i - 1]], [embeddings[i]]
        )[0][0]

        if sim > similarity_threshold:
            current_chunk.append(sentences[i])
        else:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentences[i]]

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

def get_chunks(text, strategy="Recursive", chunk_size=1000, chunk_overlap=200):
    """Select chunking method"""
    if strategy == "Recursive":
        return recursive_chunking(text, chunk_size, chunk_overlap)
    elif strategy == "Markdown-aware":
        return markdown_aware_chunking(text, max_chunk_size=chunk_size)
    elif strategy == "Semantic":
        return semantic_chunking(text)
    else:
        raise ValueError(f"Unsupported strategy: {strategy}")