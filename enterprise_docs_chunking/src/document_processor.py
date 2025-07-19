"""
Document Processor - Complete Pipeline Integration

This module provides the main pipeline that integrates:
1. Document Classification
2. Adaptive Chunking Strategies  
3. Embedding Generation
4. Vector Store Storage
5. Search and Retrieval
"""

import os
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from .document_classifier import create_classifier, DocumentType, ClassificationResult
from .chunking_strategies import chunk_document, ChunkingStrategyFactory
from .embeddings import create_fast_embedder, EmbeddingGenerator
from .vector_store import create_vector_store, ModernQdrantVectorStore, DocumentChunk
from .config import get_config


@dataclass
class ProcessingResult:
    """Result of document processing"""
    document_id: str
    classification: ClassificationResult
    chunks: List[DocumentChunk]
    chunks_stored: int
    processing_time: float
    error: Optional[str] = None


class DocumentProcessor:
    """
    Main document processing pipeline that orchestrates the entire workflow
    """
    
    def __init__(self, 
                 embedding_model: Optional[Embeddings] = None,
                 vector_store_url: Optional[str] = None,
                 collection_name: Optional[str] = None):
        """
        Initialize the document processor
        
        Args:
            embedding_model: LangChain embedding model (if None, creates default)
            vector_store_url: Qdrant server URL
            collection_name: Vector store collection name
        """
        
        # Initialize components
        self.classifier = create_classifier()
        
        # Create embedding model
        if embedding_model:
            self.embedder = embedding_model
        else:
            # Create HuggingFace embedder and wrap it for LangChain compatibility
            self.hf_embedder = create_fast_embedder()
            self.embedder = self._wrap_embedder(self.hf_embedder)
        
        # Create vector store
        self.vector_store = create_vector_store(
            embedding_model=self.embedder,
            url=vector_store_url,
            collection_name=collection_name
        )
        
        # Initialize chunking factory
        self.chunking_factory = ChunkingStrategyFactory()
        
        print("✅ Document processor initialized successfully")
        print(f"   Embedding model: {getattr(self.hf_embedder, 'model_name', 'LangChain model')}")
        print(f"   Vector store: {self.vector_store.url}")
        print(f"   Collection: {self.vector_store.collection_name}")
    
    def _wrap_embedder(self, hf_embedder: EmbeddingGenerator):
        """Wrap HuggingFace embedder for LangChain compatibility"""
        
        class HuggingFaceEmbeddingWrapper(Embeddings):
            def __init__(self, hf_embedder):
                self.hf_embedder = hf_embedder
            
            def embed_documents(self, texts: List[str]) -> List[List[float]]:
                return self.hf_embedder.encode_chunks(texts)
            
            def embed_query(self, text: str) -> List[float]:
                return self.hf_embedder.encode_text(text).tolist()
        
        return HuggingFaceEmbeddingWrapper(hf_embedder)
    
    def process_document(self, 
                        content: str, 
                        filename: Optional[str] = None,
                        metadata: Optional[Dict] = None,
                        doc_id: Optional[str] = None) -> ProcessingResult:
        """
        Process a single document through the complete pipeline
        
        Args:
            content: Document text content
            filename: Optional filename for classification hints
            metadata: Additional metadata to attach to chunks
            doc_id: Optional document ID (auto-generated if None)
            
        Returns:
            ProcessingResult with details of processing
        """
        import time
        import uuid
        
        start_time = time.time()
        doc_id = doc_id or str(uuid.uuid4())
        
        try:
            # Step 1: Document Classification
            print(f"🔍 Classifying document: {filename or doc_id}")
            classification = self.classifier.classify_document(content, filename, metadata)
            
            print(f"   Type: {classification.document_type.value}")
            print(f"   Confidence: {classification.confidence:.2f}")
            print(f"   Patterns: {', '.join(classification.detected_patterns)}")
            
            # Step 2: Get Chunking Strategy
            strategy = self.classifier.get_chunking_strategy(classification.document_type)
            print(f"   Chunking strategy: {strategy}")
            
            # Step 3: Adaptive Chunking
            print(f"✂️ Chunking document using {strategy} strategy...")
            
            # Prepare metadata for chunks
            chunk_metadata = {
                "document_id": doc_id,
                "doc_type": classification.document_type.value,
                "filename": filename,
                "confidence": classification.confidence,
                "detected_patterns": classification.detected_patterns,
                **(metadata or {})
            }
            
            # Apply chunking strategy
            chunks = chunk_document(
                content=content,
                doc_type=classification.document_type,
                strategy=strategy,
                embedder=self.hf_embedder,  # Pass HF embedder for semantic chunking
                metadata=chunk_metadata
            )
            
            print(f"   Created {len(chunks)} chunks")
            
            # Step 4: Generate Embeddings
            print(f"🧠 Generating embeddings for {len(chunks)} chunks...")
            chunk_texts = [chunk.content for chunk in chunks]
            embeddings = self.hf_embedder.encode_chunks(chunk_texts)
            
            # Attach embeddings to chunks
            for chunk, embedding in zip(chunks, embeddings):
                chunk.embedding = embedding
            
            print(f"   Embeddings generated (dimension: {len(embeddings[0])})")
            
            # Step 5: Store in Vector Database
            print(f"💾 Storing chunks in vector database...")
            
            # Try to connect to existing collection first
            if not hasattr(self.vector_store, 'vector_store') or self.vector_store.vector_store is None:
                # Try to connect to existing collection
                connected = self.vector_store.connect_to_existing()
                if not connected:
                    # Create new collection if it doesn't exist
                    success = self.vector_store.create_from_documents(chunks)
                    if not success:
                        raise Exception("Failed to create vector store collection")
                else:
                    # Add chunks to existing collection
                    success = self.vector_store.add_chunks(chunks)
                    if not success:
                        raise Exception("Failed to add chunks to vector store")
            else:
                # Add to existing vector store
                success = self.vector_store.add_chunks(chunks)
                if not success:
                    raise Exception("Failed to add chunks to vector store")
            
            processing_time = time.time() - start_time
            
            print(f"✅ Document processing completed in {processing_time:.2f}s")
            print(f"   Stored {len(chunks)} chunks in vector database")
            
            return ProcessingResult(
                document_id=doc_id,
                classification=classification,
                chunks=chunks,
                chunks_stored=len(chunks),
                processing_time=processing_time
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"Document processing failed: {str(e)}"
            print(f"❌ {error_msg}")
            
            return ProcessingResult(
                document_id=doc_id,
                classification=classification if 'classification' in locals() else None,
                chunks=[],
                chunks_stored=0,
                processing_time=processing_time,
                error=error_msg
            )
    
    def process_documents(self, 
                         documents: List[Tuple[str, Optional[str], Optional[Dict]]],
                         show_progress: bool = True) -> List[ProcessingResult]:
        """
        Process multiple documents in batch
        
        Args:
            documents: List of (content, filename, metadata) tuples
            show_progress: Whether to show progress
            
        Returns:
            List of ProcessingResult objects
        """
        results = []
        total = len(documents)
        
        for i, (content, filename, metadata) in enumerate(documents):
            if show_progress:
                print(f"\n📄 Processing document {i+1}/{total}: {filename or f'doc_{i+1}'}")
            
            result = self.process_document(content, filename, metadata)
            results.append(result)
        
        # Summary
        successful = sum(1 for r in results if r.error is None)
        total_chunks = sum(r.chunks_stored for r in results)
        avg_time = sum(r.processing_time for r in results) / len(results)
        
        print(f"\n📊 Batch Processing Summary:")
        print(f"   Documents processed: {successful}/{total}")
        print(f"   Total chunks created: {total_chunks}")
        print(f"   Average processing time: {avg_time:.2f}s per document")
        
        return results
    
    def search_documents(self, 
                        query: str,
                        top_k: int = 5,
                        doc_type_filter: Optional[str] = None,
                        score_threshold: Optional[float] = None) -> List[Tuple[DocumentChunk, float]]:
        """
        Search for relevant document chunks
        
        Args:
            query: Search query
            top_k: Number of results to return
            doc_type_filter: Filter by document type (optional)
            score_threshold: Minimum similarity score (optional)
            
        Returns:
            List of (DocumentChunk, similarity_score) tuples
        """
        print(f"🔍 Searching for: '{query}'")
        
        # Build filter conditions
        filter_conditions = {}
        if doc_type_filter:
            filter_conditions["doc_type"] = doc_type_filter
        
        # Perform search
        results = self.vector_store.search_similar(
            query=query,
            top_k=top_k,
            score_threshold=score_threshold,
            filter_conditions=filter_conditions if filter_conditions else None
        )
        
        print(f"   Found {len(results)} relevant chunks")
        for i, (chunk, score) in enumerate(results[:3], 1):
            print(f"   {i}. Score: {score:.3f} | Type: {chunk.metadata.get('doc_type', 'unknown')}")
            print(f"      Preview: {chunk.content[:100]}...")
        
        return results
    
    def get_document_summary(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Get summary information about a processed document"""
        # This would require storing document metadata
        # For now, return collection info
        return self.vector_store.get_collection_info()
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get statistics about the processing system"""
        collection_info = self.vector_store.get_collection_info()
        
        return {
            "collection_info": collection_info,
            "embedding_model": getattr(self.hf_embedder, 'model_name', 'Unknown'),
            "embedding_dimension": getattr(self.hf_embedder, 'get_embedding_dimension', lambda: 'Unknown')(),
            "chunking_strategies": ["semantic", "code_aware", "hierarchical"]
        }
    
    def clear_vector_store(self) -> bool:
        """Clear all data from the vector store"""
        print("🗑️ Clearing vector store...")
        return self.vector_store.delete_collection()


def create_document_processor(embedding_model: Optional[Embeddings] = None,
                            vector_store_url: Optional[str] = None,
                            collection_name: Optional[str] = None) -> DocumentProcessor:
    """
    Factory function to create a document processor
    
    Args:
        embedding_model: LangChain embedding model
        vector_store_url: Qdrant server URL
        collection_name: Collection name
        
    Returns:
        DocumentProcessor instance
    """
    return DocumentProcessor(
        embedding_model=embedding_model,
        vector_store_url=vector_store_url,
        collection_name=collection_name
    )


# Convenience functions for common use cases
def process_single_document(content: str, 
                          filename: Optional[str] = None) -> ProcessingResult:
    """Quick function to process a single document"""
    processor = create_document_processor()
    return processor.process_document(content, filename)


def search_knowledge_base(query: str, top_k: int = 5) -> List[Tuple[DocumentChunk, float]]:
    """Quick function to search the knowledge base"""
    processor = create_document_processor()
    return processor.search_documents(query, top_k) 