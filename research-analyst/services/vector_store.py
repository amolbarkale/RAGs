"""
Vector store service for managing embeddings and vector search.

This service handles document embeddings, similarity search, and vector database operations.
For beginners: This is where we store and search text embeddings for RAG.
"""

import logging
import asyncio
import time
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import uuid
import numpy as np

# LangChain imports
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document as LangChainDocument

# Qdrant imports
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct, 
    Filter, FieldCondition, MatchValue, SearchParams,
    UpdateResult, UpdateStatus
)

# Core imports
from core.config import settings
from core.models import DocumentChunk, SourceType, ChunkLevel
from services.models import DocumentDBModel, DocumentChunkDBModel

# Set up logging
logger = logging.getLogger(__name__)

class VectorStoreService:
    """
    Vector store service for managing embeddings and similarity search.
    
    For beginners: This class handles all vector operations - creating embeddings,
    storing them, and searching for similar content.
    """
    
    def __init__(self, qdrant_client: QdrantClient):
        self.qdrant_client = qdrant_client
        self.embeddings = None
        self.text_splitter = None
        self._initialized = False
        
    async def initialize(self):
        """Initialize the vector store service."""
        if self._initialized:
            return
            
        logger.info("🔧 Initializing vector store service...")
        
        # Initialize embedding model
        await self._init_embedding_model()
        
        # Initialize text splitter
        await self._init_text_splitter()
        
        # Ensure collections exist
        await self._ensure_collections()
        
        self._initialized = True
        logger.info("✅ Vector store service initialized!")
    
    async def _init_embedding_model(self):
        """Initialize the embedding model."""
        logger.info(f"📊 Loading embedding model: {settings.embedding_model}")
        
        try:
            # Run in thread pool to avoid blocking
            self.embeddings = await asyncio.get_event_loop().run_in_executor(
                None, 
                lambda: HuggingFaceEmbeddings(
                    model_name=settings.embedding_model,
                    model_kwargs={'device': 'cpu'},  # Use CPU for compatibility
                    encode_kwargs={'normalize_embeddings': True}
                )
            )
            logger.info("✅ Embedding model loaded successfully!")
        except Exception as e:
            logger.error(f"❌ Failed to load embedding model: {e}")
            raise
    
    async def _init_text_splitter(self):
        """Initialize the text splitter for chunking."""
        logger.info("📝 Initializing text splitter...")
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size_medium,
            chunk_overlap=settings.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        logger.info("✅ Text splitter initialized!")
    
    async def _ensure_collections(self):
        """Ensure required collections exist in Qdrant."""
        logger.info("📚 Ensuring Qdrant collections exist...")
        
        collections = {
            "documents": 768,  # Full document embeddings
            "document_chunks": 768,  # Chunk embeddings
            "query_cache": 768,  # Query embeddings for caching
        }
        
        for collection_name, vector_size in collections.items():
            try:
                # Check if collection exists
                await asyncio.get_event_loop().run_in_executor(
                    None, self.qdrant_client.get_collection, collection_name
                )
                logger.info(f"✅ Collection '{collection_name}' exists")
            except Exception:
                # Collection doesn't exist, create it
                await asyncio.get_event_loop().run_in_executor(
                    None,
                    self.qdrant_client.create_collection,
                    collection_name,
                    VectorParams(size=vector_size, distance=Distance.COSINE)
                )
                logger.info(f"✅ Created collection '{collection_name}'")
    
    async def create_document_embeddings(
        self, 
        document_id: str, 
        text: str, 
        metadata: Dict[str, Any] = None
    ) -> List[DocumentChunk]:
        """
        Create embeddings for a document by splitting it into chunks.
        
        For beginners: This takes a document, splits it into smaller pieces,
        and creates embeddings for each piece.
        """
        if not self._initialized:
            await self.initialize()
            
        logger.info(f"📝 Creating embeddings for document: {document_id}")
        
        try:
            # Split document into chunks
            chunks = await self._split_document(text)
            
            # Create embeddings for each chunk
            chunk_embeddings = []
            batch_size = settings.embedding_batch_size
            
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i + batch_size]
                batch_texts = [chunk.page_content for chunk in batch]
                
                # Generate embeddings for batch
                embeddings = await asyncio.get_event_loop().run_in_executor(
                    None, self.embeddings.embed_documents, batch_texts
                )
                
                # Create chunk objects with embeddings
                for j, (chunk, embedding) in enumerate(zip(batch, embeddings)):
                    chunk_id = f"{document_id}_chunk_{i + j}"
                    
                    # Store in Qdrant
                    await self._store_chunk_embedding(
                        chunk_id=chunk_id,
                        embedding=embedding,
                        text=chunk.page_content,
                        metadata={
                            "document_id": document_id,
                            "chunk_index": i + j,
                            "chunk_size": len(chunk.page_content),
                            "created_at": datetime.now().isoformat(),
                            **(metadata or {})
                        }
                    )
                    
                    # Create DocumentChunk object
                    chunk_embeddings.append(DocumentChunk(
                        id=chunk_id,
                        document_id=document_id,
                        chunk_index=i + j,
                        text=chunk.page_content,
                        chunk_level=ChunkLevel.MEDIUM,
                        token_count=len(chunk.page_content.split()),
                        start_char=0,  # TODO: Calculate actual positions
                        end_char=len(chunk.page_content),
                        embedding=embedding,
                        embedding_model=settings.embedding_model
                    ))
                    
                logger.info(f"✅ Processed batch {i//batch_size + 1}/{(len(chunks)-1)//batch_size + 1}")
                
            logger.info(f"✅ Created {len(chunk_embeddings)} chunk embeddings for document {document_id}")
            return chunk_embeddings
            
        except Exception as e:
            logger.error(f"❌ Failed to create embeddings for document {document_id}: {e}")
            raise
    
    async def _split_document(self, text: str) -> List[LangChainDocument]:
        """Split document into chunks using LangChain text splitter."""
        logger.info(f"📝 Splitting document ({len(text)} characters)")
        
        # Run text splitting in thread pool
        chunks = await asyncio.get_event_loop().run_in_executor(
            None, self.text_splitter.create_documents, [text]
        )
        
        logger.info(f"✅ Split into {len(chunks)} chunks")
        return chunks
    
    async def _store_chunk_embedding(
        self, 
        chunk_id: str, 
        embedding: List[float], 
        text: str, 
        metadata: Dict[str, Any]
    ):
        """Store a chunk embedding in Qdrant."""
        point = PointStruct(
            id=chunk_id,
            vector=embedding,
            payload={
                "text": text,
                "metadata": metadata
            }
        )
        
        # Store in Qdrant
        await asyncio.get_event_loop().run_in_executor(
            None,
            self.qdrant_client.upsert,
            "document_chunks",
            [point]
        )
    
    async def similarity_search(
        self, 
        query: str, 
        limit: int = 10,
        min_score: float = 0.0,
        filter_metadata: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """
        Perform similarity search for a query.
        
        For beginners: This takes a question and finds the most similar
        document chunks that might contain the answer.
        """
        if not self._initialized:
            await self.initialize()
            
        logger.info(f"🔍 Performing similarity search for: '{query[:50]}...'")
        
        try:
            # Generate query embedding
            query_embedding = await asyncio.get_event_loop().run_in_executor(
                None, self.embeddings.embed_query, query
            )
            
            # Create search filter if provided
            search_filter = None
            if filter_metadata:
                conditions = []
                for key, value in filter_metadata.items():
                    conditions.append(
                        FieldCondition(
                            key=f"metadata.{key}",
                            match=MatchValue(value=value)
                        )
                    )
                search_filter = Filter(must=conditions)
            
            # Perform search
            search_results = await asyncio.get_event_loop().run_in_executor(
                None,
                self.qdrant_client.search,
                "document_chunks",
                query_embedding,
                limit,
                search_filter,
                True,  # with_payload
                True   # with_vectors
            )
            
            # Format results
            results = []
            for result in search_results:
                if result.score >= min_score:
                    results.append({
                        "id": result.id,
                        "score": result.score,
                        "text": result.payload.get("text", ""),
                        "metadata": result.payload.get("metadata", {}),
                        "embedding": result.vector
                    })
            
            logger.info(f"✅ Found {len(results)} similar chunks (score >= {min_score})")
            return results
            
        except Exception as e:
            logger.error(f"❌ Similarity search failed: {e}")
            raise
    
    async def get_document_chunks(self, document_id: str) -> List[Dict[str, Any]]:
        """Get all chunks for a specific document."""
        logger.info(f"📖 Getting chunks for document: {document_id}")
        
        try:
            # Search with document filter
            results = await self.similarity_search(
                query="*",  # Match all
                limit=1000,  # Large limit to get all chunks
                filter_metadata={"document_id": document_id}
            )
            
            # Sort by chunk index
            results.sort(key=lambda x: x.get("metadata", {}).get("chunk_index", 0))
            
            logger.info(f"✅ Found {len(results)} chunks for document {document_id}")
            return results
            
        except Exception as e:
            logger.error(f"❌ Failed to get chunks for document {document_id}: {e}")
            raise
    
    async def delete_document_embeddings(self, document_id: str) -> bool:
        """Delete all embeddings for a document."""
        logger.info(f"🗑️  Deleting embeddings for document: {document_id}")
        
        try:
            # Delete from document_chunks collection
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                self.qdrant_client.delete,
                "document_chunks",
                Filter(must=[
                    FieldCondition(
                        key="metadata.document_id",
                        match=MatchValue(value=document_id)
                    )
                ])
            )
            
            logger.info(f"✅ Deleted embeddings for document {document_id}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to delete embeddings for document {document_id}: {e}")
            return False
    
    async def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about vector collections."""
        logger.info("📊 Getting collection statistics...")
        
        try:
            stats = {}
            
            for collection_name in ["documents", "document_chunks", "query_cache"]:
                try:
                    collection_info = await asyncio.get_event_loop().run_in_executor(
                        None, self.qdrant_client.get_collection, collection_name
                    )
                    
                    stats[collection_name] = {
                        "points_count": collection_info.points_count,
                        "vector_size": collection_info.config.params.vectors.size,
                        "distance": collection_info.config.params.vectors.distance.name,
                        "status": collection_info.status.name
                    }
                except Exception as e:
                    stats[collection_name] = {"error": str(e)}
            
            logger.info(f"✅ Retrieved stats for {len(stats)} collections")
            return stats
            
        except Exception as e:
            logger.error(f"❌ Failed to get collection stats: {e}")
            return {"error": str(e)}
    
    async def health_check(self) -> Dict[str, Any]:
        """Check the health of the vector store service."""
        logger.info("🏥 Performing vector store health check...")
        
        health = {
            "status": "healthy",
            "initialized": self._initialized,
            "embedding_model": settings.embedding_model,
            "collections": {},
            "timestamp": datetime.now().isoformat()
        }
        
        try:
            # Check collections
            collections = await asyncio.get_event_loop().run_in_executor(
                None, self.qdrant_client.get_collections
            )
            
            health["collections"] = {
                "count": len(collections.collections),
                "names": [col.name for col in collections.collections]
            }
            
            # Test embedding generation
            if self.embeddings:
                test_embedding = await asyncio.get_event_loop().run_in_executor(
                    None, self.embeddings.embed_query, "test"
                )
                health["embedding_test"] = {
                    "success": True,
                    "embedding_size": len(test_embedding)
                }
            
        except Exception as e:
            health["status"] = "unhealthy"
            health["error"] = str(e)
        
        logger.info(f"✅ Health check complete: {health['status']}")
        return health


# ==============================================
# Utility Functions
# ==============================================

async def create_vector_store(qdrant_client: QdrantClient) -> VectorStoreService:
    """Create and initialize a vector store service."""
    vector_store = VectorStoreService(qdrant_client)
    await vector_store.initialize()
    return vector_store


def calculate_similarity_score(embedding1: List[float], embedding2: List[float]) -> float:
    """Calculate cosine similarity between two embeddings."""
    import numpy as np
    
    # Convert to numpy arrays
    vec1 = np.array(embedding1)
    vec2 = np.array(embedding2)
    
    # Calculate cosine similarity
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return dot_product / (norm1 * norm2)


def chunk_text_by_tokens(text: str, chunk_size: int = 512, overlap: int = 50) -> List[str]:
    """Split text into chunks by token count."""
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
    
    return chunks


if __name__ == "__main__":
    # For testing the vector store service
    async def test_vector_store():
        from qdrant_client import QdrantClient
        
        client = QdrantClient(":memory:")
        vector_store = await create_vector_store(client)
        
        # Test health check
        health = await vector_store.health_check()
        print("Vector Store Health:", health)
        
        # Test embedding creation
        test_doc = "This is a test document for the vector store."
        embeddings = await vector_store.create_document_embeddings(
            "test_doc_1", test_doc
        )
        print(f"Created {len(embeddings)} embeddings")
        
        # Test similarity search
        results = await vector_store.similarity_search("test document")
        print(f"Found {len(results)} similar chunks")
    
    asyncio.run(test_vector_store()) 