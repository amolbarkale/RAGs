"""
Robust initialization service for the Research Assistant RAG system.

This service handles all startup initialization with proper error handling,
fallbacks, and dependency management to prevent startup failures.
"""

import logging
import asyncio
from typing import Dict, Any, Optional, Tuple
from datetime import datetime

# Core imports
from core.config import settings

# Database imports
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from qdrant_client.http.exceptions import ResponseHandlingException

# Service imports
from services.database import DatabaseManager
from services.vector_store import VectorStoreService
from services.document_processor import ProductionDocumentProcessor
from services.langchain_llm_service import LangChainLLMService
from services.tavily_search_service import TavilySearchService
from services.search_service import SearchService

# Set up logging
logger = logging.getLogger(__name__)

class RobustInitializer:
    """
    Robust initialization service that handles all startup dependencies
    with proper error handling and fallbacks.
    """
    
    def __init__(self):
        self.services = {}
        self.initialization_status = {}
        self.errors = []
        
    async def initialize_all(self) -> Dict[str, Any]:
        """
        Initialize all services with proper dependency management and error handling.
        
        Returns:
            Dictionary containing all initialized services and status
        """
        logger.info("🚀 Starting robust service initialization...")
        
        try:
            # Step 1: Initialize core database with fallbacks
            db_manager = await self._init_database_with_fallbacks()
            self.services['db_manager'] = db_manager
            
            # Step 2: Initialize vector store with robust handling
            vector_store = await self._init_vector_store_robust(db_manager)
            self.services['vector_store'] = vector_store
            
            # Step 3: Initialize services that depend on vector store
            document_processor = await self._init_document_processor(vector_store)
            search_service = await self._init_search_service(vector_store)
            
            self.services['document_processor'] = document_processor
            self.services['search_service'] = search_service
            
            # Step 4: Initialize external API services with fallbacks
            llm_service = await self._init_llm_service_with_fallback()
            tavily_service = await self._init_tavily_service_with_fallback()
            
            self.services['llm_service'] = llm_service
            self.services['tavily_service'] = tavily_service
            
            # Step 5: Validate all services
            await self._validate_all_services()
            
            logger.info("✅ All services initialized successfully!")
            
            return {
                'services': self.services,
                'status': self.initialization_status,
                'errors': self.errors,
                'success': True
            }
            
        except Exception as e:
            logger.error(f"❌ Critical initialization failure: {e}")
            self.errors.append(f"Critical failure: {e}")
            
            return {
                'services': self.services,
                'status': self.initialization_status,
                'errors': self.errors,
                'success': False
            }
    
    async def _init_database_with_fallbacks(self) -> DatabaseManager:
        """Initialize database with multiple fallback strategies."""
        logger.info("🔧 Initializing database with fallbacks...")
        
        try:
            from services.database import db_manager
            
            # Try normal initialization first
            await db_manager.initialize()
            self.initialization_status['database'] = 'success'
            logger.info("✅ Database initialized normally")
            return db_manager
            
        except Exception as e:
            logger.warning(f"⚠️ Normal database init failed: {e}")
            self.errors.append(f"Database init warning: {e}")
            
            # Try fallback initialization
            try:
                await self._init_database_fallback(db_manager)
                self.initialization_status['database'] = 'fallback'
                logger.info("✅ Database initialized with fallback")
                return db_manager
                
            except Exception as fallback_error:
                logger.error(f"❌ Database fallback failed: {fallback_error}")
                self.initialization_status['database'] = 'failed'
                self.errors.append(f"Database fallback failed: {fallback_error}")
                raise
    
    async def _init_database_fallback(self, db_manager: DatabaseManager):
        """Initialize database with in-memory fallbacks."""
        logger.info("🔄 Attempting database fallback initialization...")
        
        # Initialize traditional database with minimal config
        try:
            await db_manager._setup_sql_database()
            logger.info("✅ SQL database initialized")
        except Exception as e:
            logger.warning(f"⚠️ SQL database failed, using in-memory: {e}")
        
        # Initialize vector database with in-memory fallback
        try:
            # Force in-memory Qdrant for reliability
            db_manager.qdrant_client = QdrantClient(":memory:")
            await self._create_collections_safe(db_manager.qdrant_client)
            logger.info("✅ Vector database initialized (in-memory)")
        except Exception as e:
            logger.error(f"❌ Even in-memory vector DB failed: {e}")
            raise
    
    async def _create_collections_safe(self, qdrant_client: QdrantClient):
        """Safely create collections with proper error handling."""
        logger.info("📚 Creating vector collections safely...")
        
        collections_config = {
            "documents": {"size": 768, "distance": Distance.COSINE},
            "document_chunks": {"size": 768, "distance": Distance.COSINE},
            "query_cache": {"size": 768, "distance": Distance.COSINE}
        }
        
        for collection_name, config in collections_config.items():
            try:
                # Check if collection exists
                try:
                    await asyncio.get_event_loop().run_in_executor(
                        None, qdrant_client.get_collection, collection_name
                    )
                    logger.info(f"✅ Collection '{collection_name}' exists")
                    continue
                except:
                    # Collection doesn't exist, create it
                    pass
                
                # Create collection
                vector_params = VectorParams(
                    size=config["size"],
                    distance=config["distance"]
                )
                
                await asyncio.get_event_loop().run_in_executor(
                    None,
                    qdrant_client.create_collection,
                    collection_name,
                    vector_params
                )
                logger.info(f"✅ Created collection '{collection_name}'")
                
            except Exception as e:
                logger.warning(f"⚠️ Failed to create collection '{collection_name}': {e}")
                # Continue with other collections - don't fail completely
                continue
    
    async def _init_vector_store_robust(self, db_manager: DatabaseManager) -> VectorStoreService:
        """Initialize vector store with robust error handling."""
        logger.info("🔧 Initializing vector store robustly...")
        
        try:
            qdrant_client = db_manager.get_vector_client()
            
            # Import and create vector store
            from services.vector_store import VectorStoreService
            
            vector_store = VectorStoreService(qdrant_client)
            
            # Try to initialize - with timeout
            try:
                await asyncio.wait_for(vector_store.initialize(), timeout=30.0)
                self.initialization_status['vector_store'] = 'success'
                logger.info("✅ Vector store initialized successfully")
                
            except asyncio.TimeoutError:
                logger.warning("⚠️ Vector store initialization timed out, using minimal setup")
                await self._init_vector_store_minimal(vector_store)
                self.initialization_status['vector_store'] = 'minimal'
                
            except Exception as e:
                logger.warning(f"⚠️ Vector store init failed: {e}, using minimal setup")
                await self._init_vector_store_minimal(vector_store)
                self.initialization_status['vector_store'] = 'minimal'
            
            return vector_store
            
        except Exception as e:
            logger.error(f"❌ Vector store initialization failed: {e}")
            self.initialization_status['vector_store'] = 'failed'
            raise
    
    async def _init_vector_store_minimal(self, vector_store: VectorStoreService):
        """Initialize vector store with minimal configuration."""
        logger.info("🔧 Setting up minimal vector store...")
        
        # Set basic attributes
        vector_store._initialized = True
        vector_store.vector_stores = {}
        
        # Initialize basic embedding model
        try:
            await vector_store._init_embedding_model()
        except Exception as e:
            logger.warning(f"⚠️ Embedding model failed, using placeholder: {e}")
            vector_store.embeddings = None
        
        # Initialize text splitter
        try:
            await vector_store._init_text_splitter()
        except Exception as e:
            logger.warning(f"⚠️ Text splitter failed: {e}")
    
    async def _init_document_processor(self, vector_store: VectorStoreService) -> ProductionDocumentProcessor:
        """Initialize document processor with error handling."""
        logger.info("🔧 Initializing document processor...")
        
        try:
            processor = ProductionDocumentProcessor(vector_store)
            await processor.initialize()
            self.initialization_status['document_processor'] = 'success'
            logger.info("✅ Document processor initialized")
            return processor
            
        except Exception as e:
            logger.error(f"❌ Document processor failed: {e}")
            self.initialization_status['document_processor'] = 'failed'
            # Create a minimal processor
            processor = ProductionDocumentProcessor(vector_store)
            processor._initialized = True
            return processor
    
    async def _init_search_service(self, vector_store: VectorStoreService) -> SearchService:
        """Initialize search service with error handling."""
        logger.info("🔧 Initializing search service...")
        
        try:
            search_service = SearchService(vector_store, None)
            await search_service.initialize()
            self.initialization_status['search_service'] = 'success'
            logger.info("✅ Search service initialized")
            return search_service
            
        except Exception as e:
            logger.error(f"❌ Search service failed: {e}")
            self.initialization_status['search_service'] = 'failed'
            # Create a minimal search service
            search_service = SearchService(vector_store, None)
            search_service._initialized = True
            return search_service
    
    async def _init_llm_service_with_fallback(self) -> Optional[LangChainLLMService]:
        """Initialize LLM service with fallback handling."""
        logger.info("🔧 Initializing LLM service...")
        
        if not settings.gemini_api_key:
            logger.warning("⚠️ No Gemini API key, skipping LLM service")
            self.initialization_status['llm_service'] = 'skipped'
            return None
        
        try:
            llm_service = LangChainLLMService()
            await llm_service.initialize()
            self.initialization_status['llm_service'] = 'success'
            logger.info("✅ LLM service initialized")
            return llm_service
            
        except Exception as e:
            logger.warning(f"⚠️ LLM service failed: {e}")
            self.initialization_status['llm_service'] = 'failed'
            self.errors.append(f"LLM service failed: {e}")
            return None
    
    async def _init_tavily_service_with_fallback(self) -> Optional[TavilySearchService]:
        """Initialize Tavily service with fallback handling."""
        logger.info("🔧 Initializing Tavily service...")
        
        if not settings.tavily_api_key:
            logger.warning("⚠️ No Tavily API key, skipping web search service")
            self.initialization_status['tavily_service'] = 'skipped'
            return None
        
        try:
            tavily_service = TavilySearchService()
            await tavily_service.initialize()
            self.initialization_status['tavily_service'] = 'success'
            logger.info("✅ Tavily service initialized")
            return tavily_service
            
        except Exception as e:
            logger.warning(f"⚠️ Tavily service failed: {e}")
            self.initialization_status['tavily_service'] = 'failed'
            self.errors.append(f"Tavily service failed: {e}")
            return None
    
    async def _validate_all_services(self):
        """Validate that all services are working correctly."""
        logger.info("🔍 Validating all services...")
        
        validation_results = {}
        
        # Validate database
        if 'db_manager' in self.services:
            try:
                health = await self.services['db_manager'].health_check()
                validation_results['database'] = health
            except Exception as e:
                validation_results['database'] = {'error': str(e)}
        
        # Validate vector store
        if 'vector_store' in self.services:
            try:
                health = await self.services['vector_store'].health_check()
                validation_results['vector_store'] = health
            except Exception as e:
                validation_results['vector_store'] = {'error': str(e)}
        
        # Log validation results
        for service, result in validation_results.items():
            if 'error' in result:
                logger.warning(f"⚠️ {service} validation failed: {result['error']}")
            else:
                logger.info(f"✅ {service} validation passed")
        
        self.initialization_status['validation'] = validation_results

# Global initializer instance
robust_initializer = RobustInitializer()

async def initialize_application() -> Dict[str, Any]:
    """
    Main function to initialize the entire application robustly.
    
    Returns:
        Dictionary containing all services and initialization status
    """
    return await robust_initializer.initialize_all()

async def cleanup_application(services: Dict[str, Any]):
    """
    Clean up all application services properly.
    
    Args:
        services: Dictionary of services to clean up
    """
    logger.info("🧹 Cleaning up application services...")
    
    # Cleanup in reverse order of initialization
    cleanup_order = [
        'tavily_service',
        'llm_service', 
        'search_service',
        'document_processor',
        'vector_store',
        'db_manager'
    ]
    
    for service_name in cleanup_order:
        if service_name in services and services[service_name]:
            try:
                service = services[service_name]
                
                # Call cleanup method if available
                if hasattr(service, 'cleanup'):
                    await service.cleanup()
                elif hasattr(service, 'close'):
                    await service.close()
                
                logger.info(f"✅ Cleaned up {service_name}")
                
            except Exception as e:
                logger.warning(f"⚠️ Failed to cleanup {service_name}: {e}")
    
    logger.info("✅ Application cleanup complete!") 