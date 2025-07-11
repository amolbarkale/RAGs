"""
Vector store service for managing embeddings and vector search.

This service handles document embeddings, similarity search, and vector database operations.
For beginners: This is where we store and search text embeddings for RAG.
"""

import logging
import asyncio
import time
from typing import List, Dict, Any, Optional, Tuple, Union
from datetime import datetime
import uuid
import numpy as np

# Modern LangChain imports
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document as LangChainDocument
from langchain_core.embeddings import Embeddings
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore

# Optional modern embedding providers
try:
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    GOOGLE_AVAILABLE = True
except ImportError:
    GOOGLE_AVAILABLE = False

try:
    from langchain_openai import OpenAIEmbeddings
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# Qdrant imports
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

# Core imports
from core.config import settings
from core.models import DocumentChunk, SourceType, ChunkLevel
from services.models import DocumentDBModel, DocumentChunkDBModel

# Set up logging
logger = logging.getLogger(__name__)

class VectorStoreService:
    """
    Modern vector store service using LangChain's QdrantVectorStore.
    
    For beginners: This class handles all vector operations using the latest
    LangChain practices for embeddings and vector search.
    """
    
    def __init__(self, qdrant_client: QdrantClient):
        self.qdrant_client = qdrant_client
        self.embeddings: Optional[Embeddings] = None
        self.text_splitter: Optional[RecursiveCharacterTextSplitter] = None
        self.vector_stores: Dict[str, QdrantVectorStore] = {}
        self._initialized = False
        
    async def initialize(self):
        """Initialize the vector store service with modern practices."""
        if self._initialized:
            return
            
        logger.info("🔧 Initializing modern vector store service...")
        
        # Initialize embedding model
        await self._init_embedding_model()
        
        # Initialize text splitter
        await self._init_text_splitter()
        
        # Initialize vector stores
        await self._init_vector_stores()
        
        self._initialized = True
        logger.info("✅ Modern vector store service initialized!")
    
    async def _init_embedding_model(self):
        """Initialize embedding model with multiple provider support."""
        logger.info(f"📊 Loading embedding model: {settings.embedding_model}")
        
        try:
            # Determine embedding provider based on model name
            if "google" in settings.embedding_model.lower() or "gemini" in settings.embedding_model.lower():
                if GOOGLE_AVAILABLE and settings.google_api_key:
                    self.embeddings = GoogleGenerativeAIEmbeddings(
                        model=settings.embedding_model,
                        google_api_key=settings.google_api_key
                    )
                    logger.info("✅ Using Google Generative AI embeddings")
                else:
                    logger.warning("Google embeddings not available, falling back to HuggingFace")
                    self.embeddings = await self._init_huggingface_embeddings()
            
            elif "openai" in settings.embedding_model.lower() or "text-embedding" in settings.embedding_model.lower():
                if OPENAI_AVAILABLE and settings.openai_api_key:
                    self.embeddings = OpenAIEmbeddings(
                        model=settings.embedding_model,
                        api_key=settings.openai_api_key
                    )
                    logger.info("✅ Using OpenAI embeddings")
                else:
                    logger.warning("OpenAI embeddings not available, falling back to HuggingFace")
                    self.embeddings = await self._init_huggingface_embeddings()
            
            else:
                # Default to HuggingFace
                self.embeddings = await self._init_huggingface_embeddings()
                
        except Exception as e:
            logger.error(f"❌ Failed to load embedding model: {e}")
            # Fallback to HuggingFace
            self.embeddings = await self._init_huggingface_embeddings()
    
    async def _init_huggingface_embeddings(self) -> HuggingFaceEmbeddings:
        """Initialize HuggingFace embeddings as fallback."""
        return await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: HuggingFaceEmbeddings(
                model_name=settings.embedding_model,
                model_kwargs={'device': 'cpu', 'trust_remote_code': True},
                encode_kwargs={'normalize_embeddings': True}
            )
        )
    
    async def _init_text_splitter(self):
        """Initialize modern text splitter."""
        logger.info("📝 Initializing modern text splitter...")
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size_medium,
            chunk_overlap=settings.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""],
            add_start_index=True  # Modern feature to track chunk positions
        )
        
        logger.info("✅ Modern text splitter initialized!")
    
    async def _ensure_collections_exist(self):
        """Ensure required collections exist in Qdrant."""
        logger.info("🔍 Ensuring vector collections exist...")
        
        collections_config = {
            "documents": {
                "size": 768,  # Default embedding size
                "distance": Distance.COSINE
            },
            "document_chunks": {
                "size": 768,
                "distance": Distance.COSINE
            },
            "query_cache": {
                "size": 768,
                "distance": Distance.COSINE
            }
        }
        
        for collection_name, config in collections_config.items():
            try:
                # Check if collection exists
                await asyncio.get_event_loop().run_in_executor(
                    None, self.qdrant_client.get_collection, collection_name
                )
                logger.info(f"✅ Collection '{collection_name}' already exists")
            except Exception:
                # Collection doesn't exist, create it
                try:
                    vector_params = VectorParams(
                        size=config["size"],
                        distance=config["distance"]
                    )
                    await asyncio.get_event_loop().run_in_executor(
                        None, 
                        self.qdrant_client.create_collection,
                        collection_name,
                        vector_params
                    )
                    logger.info(f"✅ Created collection '{collection_name}'")
                except Exception as e:
                    logger.warning(f"⚠️ Failed to create collection '{collection_name}': {e}")
    
    async def _create_fallback_vector_store_modern(self, collection_name: str) -> QdrantVectorStore:
        """Create a fallback vector store using modern from_texts method."""
        logger.info(f"🔧 Creating fallback vector store for '{collection_name}' with modern syntax...")
        
        # Create with empty texts to initialize the collection using modern syntax
        def create_vector_store():
            return QdrantVectorStore.from_texts(
                texts=[""],  # Empty text to initialize
                embedding=self.embeddings,  # embedding parameter
                url=settings.qdrant_url,  # URL of the Qdrant server
                collection_name=collection_name  # name of the collection
            )
        
        fallback_store = await asyncio.get_event_loop().run_in_executor(None, create_vector_store)
        
        logger.info(f"✅ Created fallback vector store for '{collection_name}' with modern syntax")
        return fallback_store
    
    async def _init_vector_stores(self):
        """Initialize QdrantVectorStore instances using modern LangChain syntax."""
        logger.info("📚 Initializing QdrantVectorStore instances with modern syntax...")
        
        # Ensure collections exist first
        await self._ensure_collections_exist()
        
        collections = ["documents", "document_chunks", "query_cache"]
        
        for collection_name in collections:
            try:
                # Use modern from_existing_collection syntax
                vector_store = await asyncio.get_event_loop().run_in_executor(
                    None,
                    QdrantVectorStore.from_existing_collection,
                    settings.qdrant_url,  # url parameter first
                    collection_name,      # collection_name second
                    self.embeddings       # embedding parameter third
                )
                
                self.vector_stores[collection_name] = vector_store
                logger.info(f"✅ Initialized vector store for '{collection_name}' with modern syntax")
                
            except Exception as e:
                logger.warning(f"⚠️ Failed to initialize vector store for {collection_name}: {e}")
                # Create a fallback vector store using modern from_texts method
                try:
                    vector_store = await self._create_fallback_vector_store_modern(collection_name)
                    self.vector_stores[collection_name] = vector_store
                    logger.info(f"✅ Created fallback vector store for '{collection_name}' with modern syntax")
                except Exception as fallback_error:
                    logger.error(f"❌ Failed to create fallback for {collection_name}: {fallback_error}")
                    # Continue with other collections
                    continue
    
    async def create_document_embeddings(
        self, 
        document_id: str, 
        text: str, 
        metadata: Dict[str, Any] = None
    ) -> List[DocumentChunk]:
        """
        Create embeddings using modern LangChain practices.
        
        Uses QdrantVectorStore.from_texts() for efficient batch processing.
        """
        if not self._initialized:
            await self.initialize()
            
        logger.info(f"📝 Creating embeddings for document: {document_id}")
        
        try:
            # Create document and split into chunks
            document = LangChainDocument(
                page_content=text,
                metadata=metadata or {}
            )
            
            chunks = await asyncio.get_event_loop().run_in_executor(
                None, self.text_splitter.split_documents, [document]
            )
            
            # Prepare texts and metadata for batch processing
            texts = [chunk.page_content for chunk in chunks]
            metadatas = []
            
            for i, chunk in enumerate(chunks):
                chunk_metadata = {
                    "document_id": document_id,
                    "chunk_index": i,
                    "chunk_size": len(chunk.page_content),
                    "created_at": datetime.now().isoformat(),
                    "start_index": chunk.metadata.get("start_index", 0),
                    **(chunk.metadata or {}),
                    **(metadata or {})
                }
                metadatas.append(chunk_metadata)
            
            # Use modern batch processing
            vector_store = self.vector_stores["document_chunks"]
            
            # Add documents to vector store using modern syntax
            chunk_ids = [f"{document_id}_chunk_{i}" for i in range(len(texts))]
            ids = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: vector_store.add_texts(
                    texts=texts,
                    metadatas=metadatas,
                    ids=chunk_ids
                )
            )
            
            # Create DocumentChunk objects for return
            chunk_objects = []
            for i, (chunk_id, chunk_text) in enumerate(zip(ids, texts)):
                chunk_objects.append(DocumentChunk(
                    id=chunk_id,
                    document_id=document_id,
                    chunk_index=i,
                    text=chunk_text,
                    chunk_level=ChunkLevel.MEDIUM,
                    token_count=len(chunk_text.split()),
                    start_char=metadatas[i].get("start_index", 0),
                    end_char=metadatas[i].get("start_index", 0) + len(chunk_text),
                    embedding=None,  # Embedding is stored in vector store
                    embedding_model=settings.embedding_model
                ))
            
            logger.info(f"✅ Created {len(chunk_objects)} chunk embeddings for document {document_id}")
            return chunk_objects
            
        except Exception as e:
            logger.error(f"❌ Failed to create embeddings for document {document_id}: {e}")
            raise
    
    async def similarity_search(
        self, 
        query: str, 
        k: int = 10,
        score_threshold: float = 0.0,
        filter_metadata: Dict[str, Any] = None,
        collection_name: str = "document_chunks"
    ) -> List[Dict[str, Any]]:
        """
        Perform similarity search using modern LangChain QdrantVectorStore.
        
        Uses built-in similarity_search_with_score for better results.
        """
        if not self._initialized:
            await self.initialize()
            
        logger.info(f"🔍 Performing similarity search for: '{query[:50]}...'")
        
        try:
            vector_store = self.vector_stores[collection_name]
            
            # Use modern similarity search with score
            search_results = await asyncio.get_event_loop().run_in_executor(
                None,
                vector_store.similarity_search_with_score,
                query,
                k,
                filter_metadata  # Modern filter support
            )
            
            # Format results
            results = []
            for doc, score in search_results:
                if score >= score_threshold:
                    results.append({
                        "id": doc.metadata.get("id", "unknown"),
                        "score": score,
                        "text": doc.page_content,
                        "metadata": doc.metadata,
                        "document": doc  # Include full document for advanced use
                    })
            
            logger.info(f"✅ Found {len(results)} similar chunks (score >= {score_threshold})")
            return results
            
        except Exception as e:
            logger.error(f"❌ Similarity search failed: {e}")
            raise
    
    async def create_retriever(
        self,
        collection_name: str = "document_chunks",
        search_type: str = "similarity",
        search_kwargs: Dict[str, Any] = None
    ):
        """
        Create a modern LangChain retriever from the vector store.
        
        Modern practice: Use retrievers for better integration with LangChain chains.
        """
        if not self._initialized:
            await self.initialize()
            
        vector_store = self.vector_stores[collection_name]
        
        # Default search kwargs
        default_kwargs = {"k": 10, "score_threshold": 0.0}
        search_kwargs = {**default_kwargs, **(search_kwargs or {})}
        
        # Create retriever with modern configuration
        retriever = vector_store.as_retriever(
            search_type=search_type,
            search_kwargs=search_kwargs
        )
        
        logger.info(f"✅ Created {search_type} retriever for collection '{collection_name}'")
        return retriever
    
    async def from_existing_collection(
        self,
        collection_name: str,
        embedding_model: Optional[Embeddings] = None
    ) -> QdrantVectorStore:
        """
        Connect to existing collection using modern LangChain pattern.
        
        Follows the pattern from your provided code.
        """
        if not self._initialized:
            await self.initialize()
            
        embeddings = embedding_model or self.embeddings
        
        try:
            # Use modern from_existing_collection pattern
            vector_store = await asyncio.get_event_loop().run_in_executor(
                None,
                QdrantVectorStore.from_existing_collection,
                settings.qdrant_url,
                collection_name,
                embeddings
            )
            
            logger.info(f"✅ Connected to existing collection '{collection_name}'")
            return vector_store
            
        except Exception as e:
            logger.error(f"❌ Failed to connect to collection {collection_name}: {e}")
            raise
    
    async def get_document_chunks(self, document_id: str) -> List[Dict[str, Any]]:
        """Get all chunks for a specific document using modern filtering."""
        logger.info(f"📖 Getting chunks for document: {document_id}")
        
        try:
            results = await self.similarity_search(
                query="",  # Empty query for metadata-only search
                k=1000,  # Large limit to get all chunks
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
        """Delete embeddings using modern vector store methods."""
        logger.info(f"🗑️  Deleting embeddings for document: {document_id}")
        
        try:
            vector_store = self.vector_stores["document_chunks"]
            
            # Get all chunk IDs for the document
            chunks = await self.get_document_chunks(document_id)
            chunk_ids = [chunk.get("id") for chunk in chunks if chunk.get("id")]
            
            if chunk_ids:
                # Delete using modern vector store method
                await asyncio.get_event_loop().run_in_executor(
                    None,
                    vector_store.delete,
                    chunk_ids
                )
                
                logger.info(f"✅ Deleted {len(chunk_ids)} embeddings for document {document_id}")
            else:
                logger.info(f"ℹ️ No embeddings found for document {document_id}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to delete embeddings for document {document_id}: {e}")
            return False
    
    async def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about vector collections."""
        logger.info("📊 Getting collection statistics...")
        
        try:
            stats = {}
            
            for collection_name, vector_store in self.vector_stores.items():
                try:
                    # Use Qdrant client for detailed stats
                    collection_info = await asyncio.get_event_loop().run_in_executor(
                        None, self.qdrant_client.get_collection, collection_name
                    )
                    
                    stats[collection_name] = {
                        "points_count": collection_info.points_count,
                        "vector_size": collection_info.config.params.vectors.size,
                        "distance": collection_info.config.params.vectors.distance.name,
                        "status": collection_info.status.name,
                        "embedding_model": settings.embedding_model
                    }
                except Exception as e:
                    stats[collection_name] = {"error": str(e)}
            
            logger.info(f"✅ Retrieved stats for {len(stats)} collections")
            return stats
            
        except Exception as e:
            logger.error(f"❌ Failed to get collection stats: {e}")
            return {"error": str(e)}
    
    async def health_check(self) -> Dict[str, Any]:
        """Enhanced health check with modern features."""
        logger.info("🏥 Performing comprehensive health check...")
        
        health = {
            "status": "healthy",
            "initialized": self._initialized,
            "embedding_model": settings.embedding_model,
            "embedding_provider": self._get_embedding_provider(),
            "collections": {},
            "features": {
                "google_embeddings": GOOGLE_AVAILABLE,
                "openai_embeddings": OPENAI_AVAILABLE,
                "modern_langchain": True
            },
            "timestamp": datetime.now().isoformat()
        }
        
        try:
            # Test vector stores
            for collection_name, vector_store in self.vector_stores.items():
                try:
                    # Test basic functionality
                    test_result = await asyncio.get_event_loop().run_in_executor(
                        None,
                        vector_store.similarity_search,
                        "health check test",
                        1
                    )
                    health["collections"][collection_name] = {
                        "status": "healthy",
                        "test_search": "passed"
                    }
                except Exception as e:
                    health["collections"][collection_name] = {
                        "status": "unhealthy",
                        "error": str(e)
                    }
            
            # Test embedding generation
            if self.embeddings:
                try:
                    test_embedding = await asyncio.get_event_loop().run_in_executor(
                        None, self.embeddings.embed_query, "test"
                    )
                    health["embedding_test"] = {
                        "success": True,
                        "embedding_size": len(test_embedding)
                    }
                except Exception as e:
                    health["embedding_test"] = {
                        "success": False,
                        "error": str(e)
                    }
            
        except Exception as e:
            health["status"] = "unhealthy"
            health["error"] = str(e)
        
        logger.info(f"✅ Health check complete: {health['status']}")
        return health
    
    def _get_embedding_provider(self) -> str:
        """Get the current embedding provider name."""
        if self.embeddings:
            return type(self.embeddings).__name__
        return "unknown"


# ==============================================
# Modern Utility Functions
# ==============================================

async def create_vector_store(qdrant_client: QdrantClient) -> VectorStoreService:
    """Create and initialize a modern vector store service."""
    vector_store = VectorStoreService(qdrant_client)
    await vector_store.initialize()
    return vector_store


async def create_from_documents(
    documents: List[LangChainDocument],
    embeddings: Embeddings,
    collection_name: str,
    qdrant_url: str = "http://localhost:6333"
) -> QdrantVectorStore:
    """
    Create vector store from documents using modern LangChain pattern.
    
    Follows the pattern from your provided code.
    """
    return await asyncio.get_event_loop().run_in_executor(
        None,
        QdrantVectorStore.from_documents,
        documents,
        embeddings,
        url=qdrant_url,
        collection_name=collection_name
    )


def calculate_similarity_score(embedding1: List[float], embedding2: List[float]) -> float:
    """Calculate cosine similarity between two embeddings."""
    vec1 = np.array(embedding1)
    vec2 = np.array(embedding2)
    
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return dot_product / (norm1 * norm2)


def extract_content_and_metadata(search_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Extract content and metadata from search results for LLM context.
    
    Modern practice: Structured context preparation for better LLM responses.
    """
    context_parts = []
    sources = []
    
    for i, result in enumerate(search_results):
        content = result.get("text", "")
        metadata = result.get("metadata", {})
        
        # Format context with metadata
        context_part = f"[Source {i+1}] {content}"
        if metadata.get("document_id"):
            context_part += f" (Document: {metadata['document_id']})"
        if metadata.get("chunk_index") is not None:
            context_part += f" (Chunk: {metadata['chunk_index']})"
        
        context_parts.append(context_part)
        sources.append({
            "document_id": metadata.get("document_id"),
            "chunk_index": metadata.get("chunk_index"),
            "score": result.get("score", 0.0)
        })
    
    return {
        "context": "\n\n".join(context_parts),
        "sources": sources,
        "total_chunks": len(search_results)
    }


if __name__ == "__main__":
    # Modern testing approach
    async def test_modern_vector_store():
        from qdrant_client import QdrantClient
        
        client = QdrantClient(":memory:")
        vector_store = await create_vector_store(client)
        
        # Test health check
        health = await vector_store.health_check()
        print("Modern Vector Store Health:", health)
        
        # Test document processing
        test_doc = "This is a test document for the modern vector store using LangChain best practices."
        embeddings = await vector_store.create_document_embeddings(
            "test_doc_1", test_doc
        )
        print(f"Created {len(embeddings)} embeddings using modern practices")
        
        # Test retriever creation
        retriever = await vector_store.create_retriever(
            search_kwargs={"k": 5, "score_threshold": 0.1}
        )
        print("Created modern retriever")
        
        # Test similarity search
        results = await vector_store.similarity_search(
            "test document", k=3, score_threshold=0.0
        )
        print(f"Found {len(results)} similar chunks")
        
        # Test context extraction
        context_data = extract_content_and_metadata(results)
        print(f"Extracted context with {context_data['total_chunks']} chunks")
    
    asyncio.run(test_modern_vector_store()) 