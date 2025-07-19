"""
Embeddings Module for Document Chunks

This module handles text embedding generation using HuggingFace 
sentence-transformers for semantic similarity search.
"""

import os
from typing import List, Union, Optional
import numpy as np

from sentence_transformers import SentenceTransformer
import torch

from .config import EMBEDDING_CONFIG


class EmbeddingGenerator:
    """
    Text embedding generator using HuggingFace sentence-transformers
    """
    
    def __init__(self, 
                 model_name: Optional[str] = None,
                 device: Optional[str] = None,
                 batch_size: Optional[int] = None):
        """
        Initialize the embedding generator
        
        Args:
            model_name: Name of the sentence transformer model
            device: Device to run model on ('cpu', 'cuda', or 'auto')
            batch_size: Batch size for embedding generation
        """
        self.model_name = model_name or EMBEDDING_CONFIG["model_name"]
        self.device = device or EMBEDDING_CONFIG["device"]
        self.batch_size = batch_size or EMBEDDING_CONFIG["batch_size"]
        
        # Auto-detect device if not specified
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"Initializing embedding model: {self.model_name}")
        print(f"Using device: {self.device}")
        
        # Load the model
        self.model = SentenceTransformer(self.model_name, device=self.device)
        
        print(f"Model loaded successfully. Embedding dimension: {self.get_embedding_dimension()}")
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings produced by this model"""
        return self.model.get_sentence_embedding_dimension()
    
    def encode_text(self, text: Union[str, List[str]], normalize: bool = True) -> Union[np.ndarray, List[np.ndarray]]:
        """
        Generate embeddings for text(s)
        
        Args:
            text: Single text string or list of texts
            normalize: Whether to normalize embeddings to unit vectors
            
        Returns:
            Numpy array of embeddings (single text) or list of arrays (multiple texts)
        """
        try:
            # Handle single text vs list of texts
            is_single = isinstance(text, str)
            texts = [text] if is_single else text
            
            # Generate embeddings in batches
            embeddings = self.model.encode(
                texts,
                batch_size=self.batch_size,
                normalize_embeddings=normalize,
                show_progress_bar=len(texts) > 10,  # Show progress for larger batches
                convert_to_numpy=True
            )
            
            # Return single array for single text, list for multiple texts
            if is_single:
                return embeddings[0]
            else:
                return embeddings
                
        except Exception as e:
            print(f"Error generating embeddings: {e}")
            if isinstance(text, str):
                return np.zeros(self.get_embedding_dimension())
            else:
                return [np.zeros(self.get_embedding_dimension()).tolist() for _ in text]
    
    def encode_chunks(self, chunks: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of document chunks
        
        Args:
            chunks: List of text chunks
            
        Returns:
            List of embedding vectors as float lists
        """
        print(f"Generating embeddings for {len(chunks)} chunks...")
        
        embeddings = self.encode_text(chunks, normalize=True)
        
        # Convert to list of lists for JSON serialization
        if isinstance(embeddings, np.ndarray):
            return embeddings.tolist()
        elif isinstance(embeddings, list):
            return embeddings
        else:
            # Handle other cases
            return embeddings.tolist() if hasattr(embeddings, 'tolist') else list(embeddings)
    
    def compute_similarity(self, 
                          embedding1: Union[np.ndarray, List[float]], 
                          embedding2: Union[np.ndarray, List[float]]) -> float:
        """
        Compute cosine similarity between two embeddings
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Cosine similarity score (0 to 1)
        """
        try:
            # Convert to numpy arrays if needed
            emb1 = np.array(embedding1) if not isinstance(embedding1, np.ndarray) else embedding1
            emb2 = np.array(embedding2) if not isinstance(embedding2, np.ndarray) else embedding2
            
            # Compute cosine similarity
            dot_product = np.dot(emb1, emb2)
            norm1 = np.linalg.norm(emb1)
            norm2 = np.linalg.norm(emb2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            similarity = dot_product / (norm1 * norm2)
            return float(similarity)
            
        except Exception as e:
            print(f"Error computing similarity: {e}")
            return 0.0
    
    def find_semantic_boundaries(self, 
                                text_segments: List[str], 
                                similarity_threshold: float = 0.8) -> List[int]:
        """
        Find semantic boundaries in text segments based on embedding similarity
        
        Args:
            text_segments: List of text segments (sentences/paragraphs)
            similarity_threshold: Threshold for detecting semantic breaks
            
        Returns:
            List of indices where semantic boundaries occur
        """
        if len(text_segments) < 2:
            return []
        
        # Generate embeddings for all segments
        embeddings = self.encode_text(text_segments)
        boundaries = []
        
        # Compare consecutive segments
        for i in range(len(embeddings) - 1):
            similarity = self.compute_similarity(embeddings[i], embeddings[i + 1])
            
            # If similarity drops below threshold, mark as boundary
            if similarity < similarity_threshold:
                boundaries.append(i + 1)  # Boundary after segment i
        
        return boundaries
    
    def get_model_info(self) -> dict:
        """Get information about the loaded model"""
        return {
            "model_name": self.model_name,
            "device": self.device,
            "embedding_dimension": self.get_embedding_dimension(),
            "batch_size": self.batch_size,
            "max_seq_length": getattr(self.model, 'max_seq_length', 'Unknown')
        }


def create_embedding_generator(model_name: Optional[str] = None,
                             device: Optional[str] = None,
                             batch_size: Optional[int] = None) -> EmbeddingGenerator:
    """
    Factory function to create an embedding generator
    
    Args:
        model_name: Name of the sentence transformer model
        device: Device to run model on
        batch_size: Batch size for processing
        
    Returns:
        EmbeddingGenerator instance
    """
    return EmbeddingGenerator(
        model_name=model_name,
        device=device,
        batch_size=batch_size
    )


# Pre-configured embedding generators for common use cases
def create_fast_embedder() -> EmbeddingGenerator:
    """Create a fast, lightweight embedding generator"""
    return EmbeddingGenerator(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        device="auto",
        batch_size=64
    )


def create_accurate_embedder() -> EmbeddingGenerator:
    """Create a more accurate but slower embedding generator"""
    return EmbeddingGenerator(
        model_name="sentence-transformers/all-mpnet-base-v2", 
        device="auto",
        batch_size=32
    ) 