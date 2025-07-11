"""
Database configuration and connection management for the Research Assistant RAG system.

This file sets up both traditional (SQLAlchemy) and vector (Qdrant) databases.
For beginners: This is where we configure database connections and create tables.
"""

import logging
from typing import AsyncGenerator, Optional
from contextlib import asynccontextmanager
import asyncio

# SQLAlchemy imports for traditional database
from sqlalchemy import create_engine, MetaData, event
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import Session, sessionmaker, declarative_base
from sqlalchemy.pool import StaticPool

# Qdrant imports for vector database
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, CollectionInfo
from qdrant_client.http.exceptions import ResponseHandlingException

# Core imports
from core.config import settings

# Set up logging
logger = logging.getLogger(__name__)

# ==============================================
# SQLAlchemy Database Setup (Traditional DB)
# ==============================================

# Import the base class from models to avoid circular imports
from services.models import Base

# Database metadata
metadata = MetaData()

class DatabaseManager:
    """
    Manages both traditional and vector database connections.
    
    For beginners: This class handles all database setup and provides
    easy access to both SQL and vector databases.
    """
    
    def __init__(self):
        self.engine = None
        self.async_engine = None
        self.SessionLocal = None
        self.AsyncSessionLocal = None
        self.qdrant_client = None
        self._initialized = False
    
    async def initialize(self):
        """Initialize both database connections."""
        if self._initialized:
            return
        
        logger.info("🔧 Initializing database connections...")
        
        # Initialize traditional database
        await self._setup_sql_database()
        
        # Initialize vector database
        await self._setup_vector_database()
        
        self._initialized = True
        logger.info("✅ Database connections initialized successfully!")
    
    async def _setup_sql_database(self):
        """Set up SQLAlchemy database connection."""
        logger.info("📊 Setting up traditional database (SQLAlchemy)...")
        
        # Create database URL
        if settings.database_url.startswith("sqlite"):
            # SQLite configuration
            database_url = settings.database_url
            async_database_url = database_url.replace("sqlite:///", "sqlite+aiosqlite:///")
            
            # Sync engine for initial setup
            self.engine = create_engine(
                database_url,
                poolclass=StaticPool,
                connect_args={
                    "check_same_thread": False,
                    "timeout": 30
                },
                echo=settings.development  # Show SQL queries in development
            )
            
            # Async engine for FastAPI
            self.async_engine = create_async_engine(
                async_database_url,
                poolclass=StaticPool,
                connect_args={
                    "check_same_thread": False,
                    "timeout": 30
                },
                echo=settings.development
            )
        else:
            # PostgreSQL or other database
            self.engine = create_engine(
                settings.database_url,
                pool_size=20,
                max_overflow=30,
                pool_pre_ping=True,
                echo=settings.development
            )
            
            # Convert to async URL
            async_database_url = settings.database_url.replace("postgresql://", "postgresql+asyncpg://")
            self.async_engine = create_async_engine(
                async_database_url,
                pool_size=20,
                max_overflow=30,
                pool_pre_ping=True,
                echo=settings.development
            )
        
        # Create session factories
        self.SessionLocal = sessionmaker(
            bind=self.engine,
            autocommit=False,
            autoflush=False
        )
        
        self.AsyncSessionLocal = async_sessionmaker(
            bind=self.async_engine,
            class_=AsyncSession,
            autocommit=False,
            autoflush=False,
            expire_on_commit=False
        )
        
        # Create tables
        await self._create_tables()
        
        logger.info("✅ Traditional database setup complete!")
    
    async def _create_tables(self):
        """Create all database tables."""
        logger.info("🏗️  Creating database tables...")
        
        # Import models inside function to avoid circular import issues
        # This ensures all models are registered with Base.metadata
        from services.models import (
            DocumentDBModel, DocumentChunkDBModel, 
            QueryHistoryDBModel, UserDBModel, DocumentTagDBModel, SystemMetricsDBModel, CacheEntryDBModel
        )
        
        # Create tables asynchronously
        async with self.async_engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        
        logger.info("✅ Database tables created successfully!")
    
    async def _setup_vector_database(self):
        """Set up Qdrant vector database connection."""
        logger.info("🔍 Setting up vector database (Qdrant)...")
        
        try:
            # Create Qdrant client
            self.qdrant_client = QdrantClient(
                url=settings.qdrant_url,
                api_key=settings.qdrant_api_key,
                timeout=30
            )
            
            # Test connection
            collections = await asyncio.get_event_loop().run_in_executor(
                None, self.qdrant_client.get_collections
            )
            
            # Create required collections
            await self._create_vector_collections()
            
            logger.info("✅ Vector database setup complete!")
            
        except Exception as e:
            logger.error(f"❌ Failed to connect to Qdrant: {e}")
            # Fall back to in-memory vector storage
            logger.warning("🔄 Using in-memory vector storage as fallback")
            self.qdrant_client = QdrantClient(":memory:")
            await self._create_vector_collections()
    
    async def _create_vector_collections(self):
        """Create required vector collections in Qdrant."""
        logger.info("📚 Creating vector collections...")
        
        collections_config = {
            "documents": {
                "vectors": VectorParams(
                    size=768,  # Sentence transformer embedding size
                    distance=Distance.COSINE
                )
            },
            "document_chunks": {
                "vectors": VectorParams(
                    size=768,
                    distance=Distance.COSINE
                )
            },
            "query_cache": {
                "vectors": VectorParams(
                    size=768,
                    distance=Distance.COSINE
                )
            }
        }
        
        for collection_name, config in collections_config.items():
            try:
                # Check if collection exists
                await asyncio.get_event_loop().run_in_executor(
                    None, self.qdrant_client.get_collection, collection_name
                )
                logger.info(f"✅ Collection '{collection_name}' already exists")
            except ResponseHandlingException:
                # Collection doesn't exist, create it
                await asyncio.get_event_loop().run_in_executor(
                    None, 
                    self.qdrant_client.create_collection,
                    collection_name,
                    config["vectors"]
                )
                logger.info(f"✅ Created collection '{collection_name}'")
    
    async def get_db_session(self) -> AsyncGenerator[AsyncSession, None]:
        """
        Get an async database session.
        
        For beginners: This is a dependency function that FastAPI will use
        to provide database sessions to our endpoints.
        """
        if not self._initialized:
            await self.initialize()
        
        async with self.AsyncSessionLocal() as session:
            try:
                yield session
            except Exception as e:
                logger.error(f"Database session error: {e}")
                await session.rollback()
                raise
            finally:
                await session.close()
    
    def get_sync_db_session(self) -> Session:
        """
        Get a synchronous database session.
        
        For beginners: This is for synchronous operations or migrations.
        """
        if not self._initialized:
            raise RuntimeError("Database not initialized. Call initialize() first.")
        
        return self.SessionLocal()
    
    def get_vector_client(self) -> QdrantClient:
        """
        Get the Qdrant vector database client.
        
        For beginners: This provides access to our vector database
        for embedding storage and similarity search.
        """
        if not self._initialized:
            raise RuntimeError("Database not initialized. Call initialize() first.")
        
        return self.qdrant_client
    
    async def health_check(self) -> dict:
        """
        Check the health of both databases.
        
        For beginners: This endpoint will tell us if our databases are working.
        """
        health = {
            "traditional_db": False,
            "vector_db": False,
            "details": {}
        }
        
        # Check traditional database
        try:
            async with self.get_db_session() as session:
                result = await session.execute("SELECT 1")
                health["traditional_db"] = True
                health["details"]["traditional_db"] = "Connected"
        except Exception as e:
            health["details"]["traditional_db"] = f"Error: {str(e)}"
        
        # Check vector database
        try:
            collections = await asyncio.get_event_loop().run_in_executor(
                None, self.qdrant_client.get_collections
            )
            health["vector_db"] = True
            health["details"]["vector_db"] = f"Connected ({len(collections.collections)} collections)"
        except Exception as e:
            health["details"]["vector_db"] = f"Error: {str(e)}"
        
        return health
    
    async def close(self):
        """Close all database connections."""
        logger.info("🔄 Closing database connections...")
        
        if self.async_engine:
            await self.async_engine.dispose()
        
        if self.engine:
            self.engine.dispose()
        
        if self.qdrant_client:
            await asyncio.get_event_loop().run_in_executor(
                None, self.qdrant_client.close
            )
        
        self._initialized = False
        logger.info("✅ Database connections closed!")


# ==============================================
# Global Database Manager Instance
# ==============================================

# Create a global database manager instance
db_manager = DatabaseManager()

# Convenience functions for FastAPI dependency injection
async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency for database sessions."""
    async for session in db_manager.get_db_session():
        yield session

def get_vector_db() -> QdrantClient:
    """FastAPI dependency for vector database client."""
    return db_manager.get_vector_client()

# ==============================================
# Database Lifecycle Management
# ==============================================

@asynccontextmanager
async def database_lifespan():
    """
    Context manager for database lifecycle.
    
    For beginners: This ensures databases are properly
    initialized and cleaned up.
    """
    try:
        await db_manager.initialize()
        yield db_manager
    finally:
        await db_manager.close()


# ==============================================
# Utility Functions
# ==============================================

async def init_database():
    """Initialize database for testing or manual setup."""
    await db_manager.initialize()
    return db_manager

async def reset_database():
    """Reset database (development only)."""
    if not settings.development:
        raise RuntimeError("Database reset only allowed in development!")
    
    logger.warning("🔄 Resetting database...")
    
    # Drop all tables
    async with db_manager.async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    
    # Recreate tables
    await db_manager._create_tables()
    
    # Clear vector collections
    for collection_name in ["documents", "document_chunks", "query_cache"]:
        try:
            await asyncio.get_event_loop().run_in_executor(
                None, 
                db_manager.qdrant_client.delete_collection, 
                collection_name
            )
        except:
            pass
    
    await db_manager._create_vector_collections()
    
    logger.info("✅ Database reset complete!")

if __name__ == "__main__":
    # For testing database setup
    async def test_database():
        async with database_lifespan() as db:
            health = await db.health_check()
            print("Database Health Check:")
            print(f"Traditional DB: {'✅' if health['traditional_db'] else '❌'}")
            print(f"Vector DB: {'✅' if health['vector_db'] else '❌'}")
            print("Details:", health['details'])
    
    asyncio.run(test_database()) 