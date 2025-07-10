"""
Configuration management for the Research Assistant RAG system.
This module handles all environment variables and settings.
"""

import os
from typing import Optional, List
from pydantic import Field
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Settings(BaseSettings):
    """
    Application settings and configuration.
    
    For beginners: This class defines all the configuration our RAG system needs.
    It automatically loads values from environment variables or uses defaults.
    """
    
    # ==============================================
    # Application Settings
    # ==============================================
    app_name: str = "Research Assistant RAG"
    version: str = "0.1.0"
    description: str = "Advanced Hybrid RAG Research Assistant with LangChain and Gemini"
    debug: bool = False
    development: bool = True
    
    # ==============================================
    # API Keys (Keep these secret!)
    # ==============================================
    gemini_api_key: Optional[str] = None
    tavily_api_key: Optional[str] = None  # Updated to use Tavily instead of Serper
    
    # ==============================================
    # Database Configuration
    # ==============================================
    # Traditional Database (SQLAlchemy)
    database_url: str = "sqlite:///./data/research_assistant.db"
    database_echo: bool = False  # Set to True to see SQL queries
    
    # Vector Database (Qdrant)
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: Optional[str] = None
    qdrant_collection_name: str = "research_documents"
    
    # Cache Database (Redis)
    redis_url: str = "redis://localhost:6379"
    redis_password: Optional[str] = None
    
    # ==============================================
    # LangChain AI Models Configuration
    # ==============================================
    embedding_model: str = "sentence-transformers/bge-large-en-v1.5"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    llm_model: str = "gemini-2.0-flash"
    backup_llm_model: str = "gemini-1.5-flash"
    
    # ==============================================
    # Performance Settings
    # ==============================================
    max_concurrent_requests: int = 10
    embedding_batch_size: int = 32
    search_timeout: int = 30
    cache_ttl: int = 3600  # 1 hour
    
    # ==============================================
    # Quality Thresholds
    # ==============================================
    min_similarity_threshold: float = 0.5
    max_chunks_per_document: int = 100
    max_web_results: int = 10
    max_document_size_mb: int = 50
    
    # ==============================================
    # LangChain Chunking Strategy Configuration
    # ==============================================
    chunk_size_small: int = 128
    chunk_size_medium: int = 512
    chunk_size_large: int = 2048
    chunk_overlap: int = 50
    
    # ==============================================
    # Hybrid Search Weights
    # ==============================================
    dense_weight: float = 0.6
    sparse_weight: float = 0.3
    freshness_weight: float = 0.1
    
    # ==============================================
    # Logging Configuration
    # ==============================================
    log_level: str = "INFO"
    log_format: str = "json"
    log_file: str = "logs/research_assistant.log"
    
    # ==============================================
    # Security Settings
    # ==============================================
    secret_key: str = "your-secret-key-change-in-production"
    allowed_origins: List[str] = ["*"]
    rate_limit_per_minute: int = 60
    
    # ==============================================
    # Server Configuration
    # ==============================================
    host: str = "127.0.0.1"
    port: int = 8000
    reload: bool = True
    
    # ==============================================
    # Monitoring Configuration
    # ==============================================
    prometheus_port: int = 8001
    metrics_enabled: bool = True
    health_check_interval: int = 30

    class Config:
        env_file = ".env"
        case_sensitive = False
        # Map environment variables to field names
        env_prefix = ""
        
    def validate_api_keys(self) -> None:
        """
        Validate that required API keys are provided.
        
        For beginners: This function checks if we have the API keys we need.
        Without these keys, certain features won't work.
        """
        missing_keys = []
        
        if not self.gemini_api_key:
            missing_keys.append("GEMINI_API_KEY")
        if not self.tavily_api_key:
            missing_keys.append("TAVILY_API_KEY")
            
        if missing_keys:
            print(f"⚠️  Warning: Missing API keys: {', '.join(missing_keys)}")
            print("Some features may not work properly.")
            
    def get_database_url(self) -> str:
        """Get the database connection URL."""
        return self.qdrant_url
        
    def get_cache_url(self) -> str:
        """Get the cache (Redis) connection URL."""
        return self.redis_url
        
    def is_production(self) -> bool:
        """Check if we're running in production mode."""
        return not self.development and not self.debug


# Create global settings instance
settings = Settings()

# Validate API keys on startup
settings.validate_api_keys()


# ==============================================
# Helper Functions
# ==============================================

def get_chunk_sizes() -> dict:
    """
    Get all chunk sizes for multi-level chunking.
    
    For beginners: Different chunk sizes serve different purposes:
    - Small (128): For precise matching
    - Medium (512): For context understanding
    - Large (2048): For broad context
    """
    return {
        "small": settings.chunk_size_small,
        "medium": settings.chunk_size_medium,
        "large": settings.chunk_size_large,
        "overlap": settings.chunk_overlap
    }


def get_search_weights() -> dict:
    """Get weights for hybrid search scoring."""
    return {
        "dense": settings.dense_weight,
        "sparse": settings.sparse_weight,
        "freshness": settings.freshness_weight
    }


def get_model_config() -> dict:
    """Get all LangChain model configurations."""
    return {
        "embedding": settings.embedding_model,
        "reranker": settings.reranker_model,
        "llm": settings.llm_model,
        "backup_llm": settings.backup_llm_model
    }


# ==============================================
# Development Helper
# ==============================================

def print_config_summary():
    """Print a summary of current configuration (for development)."""
    print("\n🔧 Research Assistant Configuration Summary")
    print("=" * 50)
    print(f"Environment: {'Development' if settings.development else 'Production'}")
    print(f"Debug Mode: {settings.debug}")
    print(f"LangChain Integration: Enabled")
    print(f"LLM: Google Gemini Pro")
    print(f"Embedding Model: {settings.embedding_model}")
    print(f"LLM Model: {settings.llm_model}")
    print(f"Web Search: Tavily API")
    print(f"Qdrant URL: {settings.qdrant_url}")
    print(f"Redis URL: {settings.redis_url}")
    print(f"Max Document Size: {settings.max_document_size_mb}MB")
    print(f"Search Timeout: {settings.search_timeout}s")
    print(f"API Keys Configured: {bool(settings.gemini_api_key and settings.tavily_api_key)}")
    print("=" * 50)
    print()


if __name__ == "__main__":
    # Print configuration when run directly
    print_config_summary() 