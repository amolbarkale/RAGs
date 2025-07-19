"""
Configuration settings for Enterprise Document Chunking RAG System
"""
import os
from pathlib import Path
from typing import Dict, Any

# Project Root
PROJECT_ROOT = Path(__file__).parent.parent

# Vector Database Configuration (Qdrant)
QDRANT_CONFIG = {
    "host": os.getenv("QDRANT_HOST", "localhost"),
    "port": int(os.getenv("QDRANT_PORT", 6333)),
    "collection_name": "enterprise_docs",
    "vector_size": 384,  # all-MiniLM-L6-v2 dimension
    "distance": "Cosine"
}

# Embedding Model Configuration (HuggingFace)
EMBEDDING_CONFIG = {
    "model_name": "sentence-transformers/all-MiniLM-L6-v2",
    "device": "cpu",  # Change to "cuda" if GPU available
    "batch_size": 32
}

# Document Classification Config
CLASSIFICATION_CONFIG = {
    "confidence_threshold": 0.7,
    "fallback_strategy": "generic"
}

# Chunking Configuration
CHUNKING_CONFIG = {
    "semantic": {
        "chunk_size": 500,
        "chunk_overlap": 50,
        "similarity_threshold": 0.8
    },
    "code_aware": {
        "chunk_size": 800,
        "preserve_functions": True,
        "keep_imports": True
    },
    "hierarchical": {
        "respect_headers": True,
        "max_depth": 3,
        "min_chunk_size": 200
    }
}

# Supported Document Types
DOCUMENT_TYPES = {
    "technical_doc": ["pdf", "md", "rst", "txt"],
    "code_doc": ["py", "js", "ts", "java", "cpp", "md"],
    "policy_doc": ["pdf", "docx", "txt"],
    "support_doc": ["txt", "md", "html"],
    "tutorial": ["md", "rst", "ipynb"]
}

# Data Directories
DATA_DIRS = {
    "raw_docs": PROJECT_ROOT / "data" / "raw",
    "processed": PROJECT_ROOT / "data" / "processed",
    "samples": PROJECT_ROOT / "data" / "samples"
}

def get_config() -> Dict[str, Any]:
    """Get complete configuration dictionary"""
    return {
        "qdrant": QDRANT_CONFIG,
        "embedding": EMBEDDING_CONFIG,
        "classification": CLASSIFICATION_CONFIG,
        "chunking": CHUNKING_CONFIG,
        "document_types": DOCUMENT_TYPES,
        "data_dirs": DATA_DIRS
    } 