"""
Data models for the Research Assistant RAG system.

For beginners: These models define the structure of data in our application.
They help ensure data consistency and provide automatic validation.
"""

from typing import List, Optional, Dict, Any, Union
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field, validator
import uuid


# ==============================================
# Similarity Threshold Constants
# ==============================================

# Response strategy thresholds based on similarity scores
HIGH_SIMILARITY_THRESHOLD = 0.80  # Above 80% - use RAG only
LOW_SIMILARITY_THRESHOLD = 0.35   # Below 35% - use web search only
# Between 35-80% - use mixed strategy


# ==============================================
# Enums for Type Safety
# ==============================================

class DocumentType(str, Enum):
    """Types of documents we can process."""
    PDF = "pdf"
    TXT = "txt"
    DOCX = "docx"
    MARKDOWN = "markdown"


class QueryType(str, Enum):
    """Types of queries for different processing strategies."""
    FACTUAL = "factual"          # Simple fact-finding
    ANALYTICAL = "analytical"    # Complex analysis
    RECENT = "recent"           # Recent events/news
    TECHNICAL = "technical"     # Technical documentation


class SourceType(str, Enum):
    """Types of sources for results."""
    DOCUMENT = "document"
    WEB = "web"
    CACHED = "cached"


class ChunkLevel(str, Enum):
    """Chunk levels for multi-level processing."""
    SMALL = "small"      # 128 tokens
    MEDIUM = "medium"    # 512 tokens
    LARGE = "large"      # 2048 tokens


class ResponseStrategy(str, Enum):
    """Response strategy based on similarity scores."""
    RAG_ONLY = "rag_only"      # High similarity (>80%) - use only document RAG
    WEB_ONLY = "web_only"      # Low similarity (<35%) - use only web search
    MIXED = "mixed"            # Medium similarity (35-80%) - combine both sources


# ==============================================
# Base Models
# ==============================================

class TimestampedModel(BaseModel):
    """Base model with timestamp fields."""
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: Optional[datetime] = None
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class BaseResponse(BaseModel):
    """Base response model."""
    success: bool = True
    message: str = "Operation completed successfully"
    timestamp: datetime = Field(default_factory=datetime.now)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


# ==============================================
# Document Models
# ==============================================

class DocumentMetadata(BaseModel):
    """
    Document metadata information.
    
    For beginners: This stores information about uploaded documents.
    """
    title: Optional[str] = None
    author: Optional[str] = None
    subject: Optional[str] = None
    creator: Optional[str] = None
    producer: Optional[str] = None
    creation_date: Optional[datetime] = None
    modification_date: Optional[datetime] = None
    page_count: Optional[int] = None
    word_count: Optional[int] = None
    language: Optional[str] = None


class Document(TimestampedModel):
    """
    Document model representing an uploaded file.
    
    For beginners: This represents a document in our system.
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    filename: str
    original_filename: str
    file_type: DocumentType
    file_size: int  # in bytes
    file_path: str
    content_hash: str  # For deduplication
    
    # Document content
    raw_text: Optional[str] = None
    processed_text: Optional[str] = None
    
    # Metadata
    metadata: DocumentMetadata = Field(default_factory=DocumentMetadata)
    
    # Processing status
    is_processed: bool = False
    processing_error: Optional[str] = None
    
    # Statistics
    chunk_count: int = 0
    embedding_count: int = 0
    
    @validator('file_size')
    def validate_file_size(cls, v):
        """Validate file size is reasonable."""
        max_size = 50 * 1024 * 1024  # 50MB
        if v > max_size:
            raise ValueError(f"File size {v} exceeds maximum of {max_size} bytes")
        return v


class DocumentChunk(TimestampedModel):
    """
    A chunk of text from a document.
    
    For beginners: Documents are split into smaller chunks for processing.
    This makes it easier to find relevant information.
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    document_id: str
    chunk_index: int  # Position within the document
    
    # Chunk content
    text: str
    chunk_level: ChunkLevel
    token_count: int
    
    # Position information
    start_char: int
    end_char: int
    page_number: Optional[int] = None
    
    # Embedding information
    embedding: Optional[List[float]] = None
    embedding_model: Optional[str] = None
    
    # Metadata
    metadata: Dict[str, Any] = Field(default_factory=dict)


# ==============================================
# Query Models
# ==============================================

class QueryRequest(BaseModel):
    """
    User query request.
    
    For beginners: This represents a user's search query.
    """
    query: str = Field(..., min_length=1, max_length=1000)
    query_type: Optional[QueryType] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    
    # Search parameters
    max_results: int = Field(default=10, ge=1, le=50)
    include_web: bool = True
    include_documents: bool = True
    similarity_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    
    # Filters
    document_ids: Optional[List[str]] = None
    date_range: Optional[Dict[str, datetime]] = None
    
    @validator('query')
    def validate_query(cls, v):
        """Validate query is not empty after stripping."""
        if not v.strip():
            raise ValueError("Query cannot be empty")
        return v.strip()


class QueryResponse(BaseResponse):
    """
    Response to a user query.
    
    For beginners: This contains the answer to the user's question.
    """
    query: str
    answer: str
    sources: List['SearchResult'] = Field(default_factory=list)
    
    # Quality metrics
    confidence_score: float = Field(default=0.0, ge=0.0, le=1.0)
    processing_time: float = Field(default=0.0, ge=0.0)
    
    # Response strategy based on similarity scores
    response_strategy: ResponseStrategy = ResponseStrategy.MIXED
    max_similarity_score: float = Field(default=0.0, ge=0.0, le=1.0)
    
    # Metadata
    query_type: Optional[QueryType] = None
    result_count: int = 0
    
    # Citations
    citations: List[str] = Field(default_factory=list)
    
    @classmethod
    def determine_response_strategy(cls, max_similarity_score: float) -> ResponseStrategy:
        """
        Determine response strategy based on similarity score.
        
        For beginners: This decides whether to use RAG, web search, or both.
        - High similarity (>80%): Use RAG only (documents have good answers)
        - Low similarity (<35%): Use web search only (documents don't help)
        - Medium similarity (35-80%): Use both sources for comprehensive answer
        """
        if max_similarity_score >= HIGH_SIMILARITY_THRESHOLD:
            return ResponseStrategy.RAG_ONLY
        elif max_similarity_score <= LOW_SIMILARITY_THRESHOLD:
            return ResponseStrategy.WEB_ONLY
        else:
            return ResponseStrategy.MIXED


# ==============================================
# Search Models
# ==============================================

class SearchResult(BaseModel):
    """
    A search result from document or web search.
    
    For beginners: This represents a piece of information found during search.
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str
    content: str
    url: Optional[str] = None
    source_type: SourceType
    
    # Scoring
    relevance_score: float = Field(default=0.0, ge=0.0, le=1.0)
    dense_score: float = Field(default=0.0, ge=0.0, le=1.0)
    sparse_score: float = Field(default=0.0, ge=0.0, le=1.0)
    freshness_score: float = Field(default=0.0, ge=0.0, le=1.0)
    
    # Source information
    source_id: Optional[str] = None  # Document ID or web source
    source_name: Optional[str] = None
    author: Optional[str] = None
    published_date: Optional[datetime] = None
    
    # Content metadata
    snippet: Optional[str] = None  # Short excerpt
    page_number: Optional[int] = None
    chunk_id: Optional[str] = None
    
    # Quality indicators
    credibility_score: float = Field(default=0.5, ge=0.0, le=1.0)
    is_verified: bool = False


class HybridSearchResult(BaseModel):
    """
    Combined result from hybrid search.
    
    For beginners: This combines results from both document and web search.
    """
    document_results: List[SearchResult] = Field(default_factory=list)
    web_results: List[SearchResult] = Field(default_factory=list)
    combined_results: List[SearchResult] = Field(default_factory=list)
    
    # Search metadata
    total_results: int = 0
    search_time: float = 0.0
    query_classification: Optional[QueryType] = None
    
    # Response strategy based on similarity scores
    response_strategy: ResponseStrategy = ResponseStrategy.MIXED
    max_document_similarity: float = Field(default=0.0, ge=0.0, le=1.0)
    
    def determine_strategy_from_results(self) -> ResponseStrategy:
        """
        Determine response strategy based on document search results.
        
        For beginners: Analyzes the best document match to decide strategy.
        """
        if not self.document_results:
            # No document results, use web search only
            self.max_document_similarity = 0.0
            self.response_strategy = ResponseStrategy.WEB_ONLY
            return self.response_strategy
        
        # Find the highest similarity score from document results
        max_similarity = max(result.relevance_score for result in self.document_results)
        self.max_document_similarity = max_similarity
        
        # Use the same logic as QueryResponse
        self.response_strategy = QueryResponse.determine_response_strategy(max_similarity)
        return self.response_strategy


# ==============================================
# Web Search Models
# ==============================================

class WebSearchResult(BaseModel):
    """Result from web search API."""
    title: str
    url: str
    snippet: str
    content: Optional[str] = None
    
    # Source information
    domain: str
    published_date: Optional[datetime] = None
    
    # Quality metrics
    domain_authority: float = Field(default=0.5, ge=0.0, le=1.0)
    relevance_score: float = Field(default=0.0, ge=0.0, le=1.0)
    freshness_score: float = Field(default=0.0, ge=0.0, le=1.0)


# ==============================================
# File Upload Models
# ==============================================

class FileUploadRequest(BaseModel):
    """File upload request metadata."""
    filename: str
    file_type: DocumentType
    file_size: int
    content_hash: Optional[str] = None
    
    # Processing options
    process_immediately: bool = True
    extract_metadata: bool = True
    create_embeddings: bool = True


class FileUploadResponse(BaseResponse):
    """Response to file upload."""
    document_id: str
    filename: str
    file_size: int
    processing_status: str = "pending"
    
    # Processing estimates
    estimated_processing_time: Optional[float] = None
    chunk_count_estimate: Optional[int] = None


# ==============================================
# Performance Models
# ==============================================

class PerformanceMetrics(BaseModel):
    """Performance metrics for monitoring."""
    query_latency_p95: float = 0.0
    query_latency_p99: float = 0.0
    throughput_qps: float = 0.0
    
    # Accuracy metrics
    citation_accuracy: float = 0.0
    hallucination_rate: float = 0.0
    user_satisfaction: float = 0.0
    
    # System metrics
    memory_usage: float = 0.0
    cpu_usage: float = 0.0
    disk_usage: float = 0.0
    
    # Timestamps
    measurement_time: datetime = Field(default_factory=datetime.now)


# ==============================================
# Configuration Models
# ==============================================

class SystemStatus(BaseModel):
    """System status information."""
    status: str = "healthy"
    version: str = "0.1.0"
    uptime: float = 0.0
    
    # Component status
    database_connected: bool = False
    cache_connected: bool = False
    embedding_model_loaded: bool = False
    llm_available: bool = False
    
    # Statistics
    total_documents: int = 0
    total_queries: int = 0
    total_chunks: int = 0
    
    # Performance
    avg_query_time: float = 0.0
    success_rate: float = 0.0


# ==============================================
# Error Models
# ==============================================

class ErrorResponse(BaseModel):
    """Error response model."""
    success: bool = False
    error_code: str
    error_message: str
    details: Optional[Dict[str, Any]] = None
    timestamp: datetime = Field(default_factory=datetime.now)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


# ==============================================
# Update Forward References
# ==============================================

# Update forward references for proper type hints
QueryResponse.model_rebuild()
HybridSearchResult.model_rebuild()


# ==============================================
# Example Usage (for testing)
# ==============================================

if __name__ == "__main__":
    # Test model creation
    print("Testing data models...")
    
    # Create a document
    doc = Document(
        filename="test.pdf",
        original_filename="test.pdf",
        file_type=DocumentType.PDF,
        file_size=1024,
        file_path="/tmp/test.pdf",
        content_hash="abc123"
    )
    
    print(f"Created document: {doc.id}")
    
    # Create a query
    query = QueryRequest(
        query="What is machine learning?",
        query_type=QueryType.FACTUAL
    )
    
    print(f"Created query: {query.query}")
    
    # Create search results
    result = SearchResult(
        title="Machine Learning Basics",
        content="Machine learning is a subset of AI...",
        source_type=SourceType.DOCUMENT,
        relevance_score=0.85
    )
    
    print(f"Created search result: {result.title}")
    
    print("Model testing completed!") 