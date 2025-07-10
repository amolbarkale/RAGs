"""
SQLAlchemy database models for the Research Assistant RAG system.

These models define the actual database table structure.
For beginners: These are different from Pydantic models - they define database tables,
while Pydantic models define API validation and serialization.
"""

from sqlalchemy import Column, Integer, String, Text, Float, Boolean, DateTime, ForeignKey, JSON, LargeBinary
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship, declarative_base
from datetime import datetime
from typing import Optional, List, Dict, Any
import uuid

# Import the Base from database.py
from services.database import Base

# ==============================================
# Database Models (SQLAlchemy)
# ==============================================

class TimestampMixin:
    """
    Mixin for adding timestamp fields to models.
    
    For beginners: This automatically adds created_at and updated_at
    to any model that inherits from it.
    """
    created_at = Column(DateTime, default=func.now(), nullable=False)
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now(), nullable=True)


class UserDBModel(Base, TimestampMixin):
    """
    Database model for user management.
    
    For beginners: This stores user information in the database.
    """
    __tablename__ = "users"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    username = Column(String(100), unique=True, nullable=False, index=True)
    email = Column(String(255), unique=True, nullable=False, index=True)
    full_name = Column(String(255), nullable=True)
    is_active = Column(Boolean, default=True, nullable=False)
    is_admin = Column(Boolean, default=False, nullable=False)
    
    # API key for authentication
    api_key = Column(String(255), unique=True, nullable=True, index=True)
    
    # User preferences
    preferences = Column(JSON, nullable=True)
    
    # Relationships
    documents = relationship("DocumentDBModel", back_populates="owner")
    query_history = relationship("QueryHistoryDBModel", back_populates="user")
    
    def __repr__(self):
        return f"<User(id={self.id}, username={self.username})>"


class DocumentDBModel(Base, TimestampMixin):
    """
    Database model for document storage.
    
    For beginners: This stores document metadata in the database.
    The actual file content is stored separately.
    """
    __tablename__ = "documents"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    filename = Column(String(255), nullable=False, index=True)
    original_filename = Column(String(255), nullable=False)
    file_type = Column(String(50), nullable=False)  # pdf, txt, docx, etc.
    file_size = Column(Integer, nullable=False)
    file_path = Column(String(500), nullable=False)
    content_hash = Column(String(64), unique=True, nullable=False, index=True)
    
    # Content storage
    raw_text = Column(Text, nullable=True)
    processed_text = Column(Text, nullable=True)
    
    # Document metadata (stored as JSON)
    metadata = Column(JSON, nullable=True)
    
    # Processing status
    is_processed = Column(Boolean, default=False, nullable=False)
    processing_error = Column(Text, nullable=True)
    processing_started_at = Column(DateTime, nullable=True)
    processing_completed_at = Column(DateTime, nullable=True)
    
    # Statistics
    chunk_count = Column(Integer, default=0, nullable=False)
    embedding_count = Column(Integer, default=0, nullable=False)
    word_count = Column(Integer, default=0, nullable=False)
    
    # Relationships
    owner_id = Column(String, ForeignKey("users.id"), nullable=True)
    owner = relationship("UserDBModel", back_populates="documents")
    chunks = relationship("DocumentChunkDBModel", back_populates="document", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Document(id={self.id}, filename={self.filename})>"


class DocumentChunkDBModel(Base, TimestampMixin):
    """
    Database model for document chunks.
    
    For beginners: Documents are split into smaller chunks for better search.
    This stores information about each chunk.
    """
    __tablename__ = "document_chunks"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    document_id = Column(String, ForeignKey("documents.id"), nullable=False)
    chunk_index = Column(Integer, nullable=False)  # Position within document
    
    # Chunk content
    text = Column(Text, nullable=False)
    chunk_level = Column(String(20), nullable=False)  # small, medium, large
    token_count = Column(Integer, nullable=False)
    
    # Position information
    start_char = Column(Integer, nullable=False)
    end_char = Column(Integer, nullable=False)
    page_number = Column(Integer, nullable=True)
    
    # Embedding information (stored in vector database)
    embedding_id = Column(String, nullable=True)  # Reference to vector DB
    embedding_model = Column(String(100), nullable=True)
    
    # Chunk metadata
    metadata = Column(JSON, nullable=True)
    
    # Quality scores
    content_quality_score = Column(Float, default=0.0, nullable=False)
    readability_score = Column(Float, default=0.0, nullable=False)
    
    # Relationships
    document = relationship("DocumentDBModel", back_populates="chunks")
    
    def __repr__(self):
        return f"<DocumentChunk(id={self.id}, doc_id={self.document_id}, index={self.chunk_index})>"


class QueryHistoryDBModel(Base, TimestampMixin):
    """
    Database model for query history and analytics.
    
    For beginners: This stores user queries and responses for analytics
    and improving the system.
    """
    __tablename__ = "query_history"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, ForeignKey("users.id"), nullable=True)
    session_id = Column(String(100), nullable=True, index=True)
    
    # Query information
    query_text = Column(Text, nullable=False)
    query_type = Column(String(50), nullable=True)  # factual, analytical, etc.
    query_hash = Column(String(64), nullable=False, index=True)  # For deduplication
    
    # Response information
    response_text = Column(Text, nullable=True)
    response_strategy = Column(String(50), nullable=True)  # rag_only, web_only, mixed
    
    # Performance metrics
    processing_time = Column(Float, nullable=False)
    confidence_score = Column(Float, nullable=False)
    relevance_score = Column(Float, nullable=False)
    
    # Search results (stored as JSON)
    search_results = Column(JSON, nullable=True)
    sources_used = Column(JSON, nullable=True)
    
    # User feedback
    user_rating = Column(Integer, nullable=True)  # 1-5 stars
    user_feedback = Column(Text, nullable=True)
    
    # System information
    model_version = Column(String(50), nullable=True)
    api_version = Column(String(20), nullable=True)
    
    # Relationships
    user = relationship("UserDBModel", back_populates="query_history")
    
    def __repr__(self):
        return f"<QueryHistory(id={self.id}, user_id={self.user_id})>"


class DocumentTagDBModel(Base, TimestampMixin):
    """
    Database model for document tags and categories.
    
    For beginners: This allows us to categorize and tag documents
    for better organization and search.
    """
    __tablename__ = "document_tags"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    document_id = Column(String, ForeignKey("documents.id"), nullable=False)
    tag_name = Column(String(100), nullable=False, index=True)
    tag_type = Column(String(50), nullable=False)  # manual, auto, system
    confidence_score = Column(Float, default=1.0, nullable=False)
    
    # Tag metadata
    metadata = Column(JSON, nullable=True)
    
    def __repr__(self):
        return f"<DocumentTag(doc_id={self.document_id}, tag={self.tag_name})>"


class SystemMetricsDBModel(Base, TimestampMixin):
    """
    Database model for system metrics and monitoring.
    
    For beginners: This stores system performance metrics
    for monitoring and optimization.
    """
    __tablename__ = "system_metrics"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    metric_name = Column(String(100), nullable=False, index=True)
    metric_value = Column(Float, nullable=False)
    metric_unit = Column(String(50), nullable=True)
    
    # Metric metadata
    metadata = Column(JSON, nullable=True)
    
    # Aggregation period
    period_start = Column(DateTime, nullable=True)
    period_end = Column(DateTime, nullable=True)
    
    def __repr__(self):
        return f"<SystemMetrics(name={self.metric_name}, value={self.metric_value})>"


class CacheEntryDBModel(Base, TimestampMixin):
    """
    Database model for caching query results.
    
    For beginners: This stores cached results to speed up repeated queries.
    """
    __tablename__ = "cache_entries"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    cache_key = Column(String(255), unique=True, nullable=False, index=True)
    cache_type = Column(String(50), nullable=False)  # query, embedding, web_search
    
    # Cache content
    cache_value = Column(JSON, nullable=False)
    
    # Cache metadata
    hit_count = Column(Integer, default=0, nullable=False)
    last_accessed = Column(DateTime, default=func.now(), nullable=False)
    expires_at = Column(DateTime, nullable=True)
    
    # Size estimation
    size_bytes = Column(Integer, nullable=True)
    
    def __repr__(self):
        return f"<CacheEntry(key={self.cache_key}, type={self.cache_type})>"


# ==============================================
# Database Utility Functions
# ==============================================

def get_model_by_name(model_name: str):
    """
    Get a model class by its name.
    
    For beginners: This is a utility function to dynamically
    get model classes by their string names.
    """
    models = {
        'user': UserDBModel,
        'document': DocumentDBModel,
        'document_chunk': DocumentChunkDBModel,
        'query_history': QueryHistoryDBModel,
        'document_tag': DocumentTagDBModel,
        'system_metrics': SystemMetricsDBModel,
        'cache_entry': CacheEntryDBModel
    }
    return models.get(model_name.lower())


def get_all_models():
    """
    Get all model classes.
    
    For beginners: This returns all our database models
    for bulk operations like migrations.
    """
    return [
        UserDBModel,
        DocumentDBModel,
        DocumentChunkDBModel,
        QueryHistoryDBModel,
        DocumentTagDBModel,
        SystemMetricsDBModel,
        CacheEntryDBModel
    ]


# ==============================================
# Model Validation Functions
# ==============================================

def validate_document_size(file_size: int) -> bool:
    """Validate document file size."""
    max_size = 50 * 1024 * 1024  # 50MB
    return file_size <= max_size


def validate_chunk_size(text: str, chunk_level: str) -> bool:
    """Validate chunk size based on level."""
    token_count = len(text.split())  # Simple token count
    
    limits = {
        'small': 128,
        'medium': 512,
        'large': 2048
    }
    
    return token_count <= limits.get(chunk_level, 512)


def generate_content_hash(content: str) -> str:
    """Generate content hash for deduplication."""
    import hashlib
    return hashlib.sha256(content.encode()).hexdigest()


# ==============================================
# Database Schema Information
# ==============================================

def get_database_schema():
    """
    Get database schema information.
    
    For beginners: This provides information about our database structure
    for documentation and debugging.
    """
    schema = {}
    
    for model in get_all_models():
        table_name = model.__tablename__
        columns = []
        
        for column in model.__table__.columns:
            columns.append({
                'name': column.name,
                'type': str(column.type),
                'nullable': column.nullable,
                'primary_key': column.primary_key,
                'unique': column.unique
            })
        
        schema[table_name] = {
            'model': model.__name__,
            'columns': columns
        }
    
    return schema


if __name__ == "__main__":
    # For testing model definitions
    print("Database Models:")
    for model in get_all_models():
        print(f"- {model.__name__} -> {model.__tablename__}")
    
    print("\nDatabase Schema:")
    schema = get_database_schema()
    for table_name, info in schema.items():
        print(f"\n{table_name}:")
        for col in info['columns']:
            print(f"  - {col['name']}: {col['type']}") 