"""
Qdrant Vector Store Configuration and Management

This module handles Qdrant vector database operations for storing 
and retrieving document chunks with their embeddings and metadata.
Uses modern LangChain QdrantVectorStore for better integration.
"""

import uuid
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass

from langchain_qdrant import QdrantVectorStore as LangChainQdrantVectorStore
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from .config import QDRANT_CONFIG


@dataclass
class DocumentChunk:
    """Represents a document chunk with metadata"""
    id: str
    content: str
    embedding: Optional[List[float]] = None
    metadata: Optional[Dict[str, Any]] = None
    
    @classmethod
    def create(cls, content: str, embedding: Optional[List[float]] = None, **metadata):
        """Create a new document chunk with auto-generated ID"""
        return cls(
            id=str(uuid.uuid4()),
            content=content,
            embedding=embedding,
            metadata=metadata or {}
        )
    
    def to_langchain_document(self) -> Document:
        """Convert to LangChain Document format"""
        metadata = self.metadata.copy() if self.metadata else {}
        metadata["chunk_id"] = self.id
        return Document(page_content=self.content, metadata=metadata)
    
    @classmethod 
    def from_langchain_document(cls, doc: Document, embedding: Optional[List[float]] = None) -> "DocumentChunk":
        """Create DocumentChunk from LangChain Document"""
        metadata = doc.metadata.copy()
        chunk_id = metadata.pop("chunk_id", str(uuid.uuid4()))
        return cls(
            id=chunk_id,
            content=doc.page_content,
            embedding=embedding,
            metadata=metadata
        )


class ModernQdrantVectorStore:
    """
    Modern Qdrant vector database manager using LangChain QdrantVectorStore
    """
    
    def __init__(self, embedding_model: Embeddings, url: Optional[str] = None, collection_name: Optional[str] = None):
        """
        Initialize Qdrant vector store with embedding model
        
        Args:
            embedding_model: LangChain embedding model instance
            url: Qdrant server URL (defaults to config)
            collection_name: Collection name (defaults to config)
        """
        self.embedding_model = embedding_model
        self.url = url or f"http://{QDRANT_CONFIG['host']}:{QDRANT_CONFIG['port']}"
        self.collection_name = collection_name or QDRANT_CONFIG["collection_name"]
        self.vector_store = None
        
        print(f"Initialized Qdrant store: {self.url}")
        print(f"Collection: {self.collection_name}")
    
    def create_from_documents(self, chunks: List[DocumentChunk]) -> bool:
        """
        Create vector store from document chunks
        
        Args:
            chunks: List of DocumentChunk objects
            
        Returns:
            bool: True if successful
        """
        try:
            # Convert chunks to LangChain documents
            documents = [chunk.to_langchain_document() for chunk in chunks]
            
            # Create vector store from documents
            self.vector_store = LangChainQdrantVectorStore.from_documents(
                documents=documents,
                url=self.url,
                collection_name=self.collection_name,
                embedding=self.embedding_model,
            )
            
            print(f"✅ Created vector store with {len(chunks)} chunks")
            return True
            
        except Exception as e:
            print(f"❌ Error creating vector store: {e}")
            return False
    
    def connect_to_existing(self) -> bool:
        """
        Connect to existing collection
        
        Returns:
            bool: True if successful
        """
        try:
            self.vector_store = LangChainQdrantVectorStore.from_existing_collection(
                url=self.url,
                collection_name=self.collection_name,
                embedding=self.embedding_model,
            )
            
            print(f"✅ Connected to existing collection: {self.collection_name}")
            return True
            
        except Exception as e:
            print(f"❌ Error connecting to existing collection: {e}")
            return False
    
    def add_chunks(self, chunks: List[DocumentChunk]) -> bool:
        """
        Add chunks to existing vector store
        
        Args:
            chunks: List of DocumentChunk objects
            
        Returns:
            bool: True if successful
        """
        try:
            if not self.vector_store:
                print("❌ Vector store not initialized. Call create_from_documents() or connect_to_existing() first.")
                return False
            
            # Convert chunks to LangChain documents
            documents = [chunk.to_langchain_document() for chunk in chunks]
            
            # Add documents to vector store
            self.vector_store.add_documents(documents)
            
            print(f"✅ Added {len(chunks)} chunks to vector store")
            return True
            
        except Exception as e:
            print(f"❌ Error adding chunks: {e}")
            return False
    
    def search_similar(self, 
                      query: str, 
                      top_k: int = 5,
                      score_threshold: Optional[float] = None,
                      filter_conditions: Optional[Dict] = None) -> List[Tuple[DocumentChunk, float]]:
        """
        Search for similar document chunks
        
        Args:
            query: Query text
            top_k: Number of results to return
            score_threshold: Minimum similarity score (optional)
            filter_conditions: Metadata filters (optional)
            
        Returns:
            List of tuples (DocumentChunk, similarity_score)
        """
        try:
            if not self.vector_store:
                print("❌ Vector store not initialized")
                return []
            
            # Perform similarity search with scores
            if score_threshold is not None:
                results = self.vector_store.similarity_search_with_score(
                    query=query,
                    k=top_k,
                    score_threshold=score_threshold,
                    filter=filter_conditions
                )
            else:
                results = self.vector_store.similarity_search_with_score(
                    query=query,
                    k=top_k,
                    filter=filter_conditions
                )
            
            # Convert results to DocumentChunk format
            chunks_with_scores = []
            for doc, score in results:
                chunk = DocumentChunk.from_langchain_document(doc)
                chunks_with_scores.append((chunk, score))
            
            return chunks_with_scores
            
        except Exception as e:
            print(f"❌ Error searching vector store: {e}")
            return []
    
    def search_similar_chunks(self, 
                            query: str, 
                            top_k: int = 5) -> List[DocumentChunk]:
        """
        Simple similarity search returning only chunks
        
        Args:
            query: Query text
            top_k: Number of results to return
            
        Returns:
            List of DocumentChunk objects
        """
        try:
            if not self.vector_store:
                print("❌ Vector store not initialized")
                return []
            
            # Perform similarity search
            docs = self.vector_store.similarity_search(query=query, k=top_k)
            
            # Convert to DocumentChunk format
            chunks = [DocumentChunk.from_langchain_document(doc) for doc in docs]
            return chunks
            
        except Exception as e:
            print(f"❌ Error searching vector store: {e}")
            return []
    
    def get_collection_info(self) -> Dict[str, Any]:
        """
        Get information about the current collection
        
        Returns:
            Dictionary with collection statistics
        """
        try:
            if not self.vector_store:
                return {"error": "Vector store not initialized"}
            
            # Get underlying Qdrant client info
            client = self.vector_store.client
            collection_info = client.get_collection(self.collection_name)
            
            return {
                "collection_name": self.collection_name,
                "points_count": collection_info.points_count,
                "vector_size": collection_info.config.params.vectors.size,
                "distance": collection_info.config.params.vectors.distance.value,
                "status": collection_info.status.value,
                "url": self.url
            }
            
        except Exception as e:
            return {"error": f"Could not get collection info: {e}"}
    
    def delete_collection(self) -> bool:
        """
        Delete the entire collection
        
        Returns:
            bool: True if successful
        """
        try:
            if not self.vector_store:
                print("❌ Vector store not initialized")
                return False
            
            # Get underlying client and delete collection
            client = self.vector_store.client
            client.delete_collection(self.collection_name)
            
            print(f"✅ Deleted collection: {self.collection_name}")
            self.vector_store = None
            return True
            
        except Exception as e:
            print(f"❌ Error deleting collection: {e}")
            return False


def create_vector_store(embedding_model: Embeddings,
                       url: Optional[str] = None,
                       collection_name: Optional[str] = None) -> ModernQdrantVectorStore:
    """
    Factory function to create a modern Qdrant vector store
    
    Args:
        embedding_model: LangChain embedding model instance
        url: Qdrant server URL
        collection_name: Collection name
        
    Returns:
        ModernQdrantVectorStore instance
    """
    return ModernQdrantVectorStore(
        embedding_model=embedding_model,
        url=url,
        collection_name=collection_name
    )


# Backward compatibility - alias for the old interface
QdrantVectorStore = ModernQdrantVectorStore 