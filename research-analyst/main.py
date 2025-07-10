"""
Main FastAPI application for the Research Assistant RAG system.

For beginners: This is the entry point of our web application.
FastAPI is a modern web framework that automatically creates API documentation.
"""

from fastapi import FastAPI, HTTPException, File, UploadFile, Depends, Form, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import uvicorn
from contextlib import asynccontextmanager
import logging
from typing import Dict, Any, List, Optional
import json
import time
from datetime import datetime
import io
import asyncio

# Import our configuration and models
from core.config import settings, print_config_summary
from core.models import (
    Document, DocumentChunk, DocumentMetadata, DocumentType,
    QueryRequest, QueryResponse, QueryType, ResponseStrategy,
    SearchResult, HybridSearchResult, SourceType
)
from utils.logger import setup_logging


# ==============================================
# Application Lifecycle Management
# ==============================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manage application startup and shutdown.
    
    For beginners: This function runs when the app starts and stops.
    It's where we initialize databases, load models, etc.
    """
    # Startup
    print("🚀 Starting Research Assistant RAG system...")
    print_config_summary()
    
    # TODO: Initialize database connections
    # TODO: Load AI models
    # TODO: Setup monitoring
    
    print("✅ Application started successfully!")
    
    yield  # This is where the app runs
    
    # Shutdown
    print("🔄 Shutting down Research Assistant...")
    # TODO: Cleanup database connections
    # TODO: Save any pending data
    print("✅ Application shutdown complete!")


# ==============================================
# Create FastAPI Application
# ==============================================

app = FastAPI(
    title=settings.app_name,
    description=settings.description,
    version=settings.version,
    lifespan=lifespan,
    docs_url="/docs",  # Swagger UI
    redoc_url="/redoc"  # ReDoc UI
)

# ==============================================
# Security (Optional Bearer Token)
# ==============================================

security = HTTPBearer(auto_error=False)

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """
    Optional authentication - returns None if no token provided.
    
    For beginners: This checks if the user provided a valid token.
    """
    if not credentials:
        return None
    
    # TODO: Implement real authentication
    # For now, just return a mock user
    return {"user_id": "demo_user", "username": "demo"}


# ==============================================
# Middleware Configuration
# ==============================================

# CORS middleware for cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==============================================
# Basic Routes
# ==============================================

@app.get("/")
async def root() -> Dict[str, str]:
    """
    Root endpoint - returns basic info about the API.
    
    For beginners: This is the main endpoint that shows our API is running.
    """
    return {
        "message": "Welcome to the Research Assistant RAG API!",
        "version": settings.version,
        "description": settings.description,
        "docs": "/docs",
        "status": "operational"
    }


@app.get("/health")
async def health_check() -> Dict[str, Any]:
    """
    Health check endpoint for monitoring.
    
    For beginners: This endpoint tells us if the system is healthy.
    Useful for monitoring and deployment systems.
    """
    return {
        "status": "healthy",
        "version": settings.version,
        "environment": "development" if settings.development else "production",
        "timestamp": datetime.now().isoformat()
    }


@app.get("/config")
async def get_config() -> Dict[str, Any]:
    """
    Get current configuration (non-sensitive info only).
    
    For beginners: This shows the current settings without revealing secrets.
    """
    if not settings.development:
        raise HTTPException(
            status_code=403, 
            detail="Configuration endpoint only available in development mode"
        )
    
    return {
        "embedding_model": settings.embedding_model,
        "llm_model": settings.llm_model,
        "max_document_size_mb": settings.max_document_size_mb,
        "chunk_sizes": {
            "small": settings.chunk_size_small,
            "medium": settings.chunk_size_medium,
            "large": settings.chunk_size_large
        },
        "search_weights": {
            "dense": settings.dense_weight,
            "sparse": settings.sparse_weight,
            "freshness": settings.freshness_weight
        }
    }


# ==============================================
# Document Management Endpoints
# ==============================================

@app.post("/documents/upload", response_model=Document)
async def upload_document(
    file: UploadFile = File(...),
    title: Optional[str] = Form(None),
    description: Optional[str] = Form(None),
    tags: Optional[str] = Form(None),
    user: Optional[dict] = Depends(get_current_user)
) -> Document:
    """
    Upload a document (PDF, TXT, DOCX).
    
    For beginners: This endpoint receives a file and creates a Document object.
    It shows how FastAPI handles file uploads and integrates with our models.
    """
    # Validate file size
    content = await file.read()
    file_size_mb = len(content) / (1024 * 1024)
    
    if file_size_mb > settings.max_document_size_mb:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File size {file_size_mb:.1f}MB exceeds maximum allowed size of {settings.max_document_size_mb}MB"
        )
    
    # Determine document type
    filename = file.filename or "unknown"
    file_extension = filename.split('.')[-1].lower() if '.' in filename else ''
    
    if file_extension == 'pdf':
        doc_type = DocumentType.PDF
    elif file_extension == 'txt':
        doc_type = DocumentType.TXT
    elif file_extension in ['doc', 'docx']:
        doc_type = DocumentType.DOCX
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type: {file_extension}"
        )
    
    # Create document metadata
    metadata = DocumentMetadata(
        title=title or filename,
        author=user.get('username', 'unknown') if user else 'anonymous',
        language='en'  # TODO: Detect language automatically
    )
    
    # Create document object
    document = Document(
        filename=filename,
        original_filename=filename,
        file_type=doc_type,
        file_size=len(content),
        file_path=f"uploads/{filename}",
        content_hash=f"hash_{len(content)}",  # TODO: Generate real hash
        raw_text=content.decode('utf-8') if doc_type == DocumentType.TXT else "",
        metadata=metadata
    )
    
    # TODO: Save to database
    # TODO: Process document (extract text, create chunks, generate embeddings)
    
    return document


@app.get("/documents", response_model=List[Document])
async def list_documents(
    limit: int = 10,
    offset: int = 0,
    document_type: Optional[DocumentType] = None,
    user: Optional[dict] = Depends(get_current_user)
) -> List[Document]:
    """
    List uploaded documents with filtering and pagination.
    
    For beginners: This shows how to use query parameters and optional filtering.
    """
    # TODO: Implement database query
    # For now, return mock data
    
    mock_documents = [
        Document(
            filename="sample_research_paper.pdf",
            original_filename="sample_research_paper.pdf",
            file_type=DocumentType.PDF,
            file_size=1024000,
            file_path="uploads/sample_research_paper.pdf",
            content_hash="hash_sample_1",
            raw_text="This is a sample research paper about AI...",
            metadata=DocumentMetadata(
                title="Sample Research Paper",
                author="Dr. Smith",
                language="en"
            )
        ),
        Document(
            filename="technical_documentation.txt",
            original_filename="technical_documentation.txt",
            file_type=DocumentType.TXT,
            file_size=512000,
            file_path="uploads/technical_documentation.txt",
            content_hash="hash_sample_2",
            raw_text="API documentation for the research system...",
            metadata=DocumentMetadata(
                title="Technical Documentation",
                author="Dev Team",
                language="en"
            )
        )
    ]
    
    # Apply filtering
    if document_type:
        mock_documents = [doc for doc in mock_documents if doc.file_type == document_type]
    
    # Apply pagination
    return mock_documents[offset:offset + limit]


@app.delete("/documents/{document_id}")
async def delete_document(
    document_id: str,
    user: Optional[dict] = Depends(get_current_user)
) -> Dict[str, str]:
    """
    Delete a document by ID.
    
    For beginners: This shows how to use path parameters and return simple responses.
    """
    # TODO: Implement database deletion
    # TODO: Check user permissions
    
    return {
        "message": f"Document {document_id} deleted successfully",
        "deleted_at": datetime.now().isoformat()
    }


# ==============================================
# Query & Search Endpoints
# ==============================================

@app.post("/query", response_model=QueryResponse)
async def process_query(
    query_request: QueryRequest,
    user: Optional[dict] = Depends(get_current_user)
) -> QueryResponse:
    """
    Process a research query using hybrid RAG.
    
    For beginners: This is the main endpoint that showcases our smart models.
    It demonstrates how FastAPI automatically validates request bodies.
    """
    start_time = time.time()
    
    # Smart response strategy based on similarity (mock 70% for demo)
    strategy = QueryResponse.determine_response_strategy(max_similarity_score=0.7)
    
    # TODO: Implement actual search logic
    # For now, create a mock response
    
    # Simulate search results
    search_results = [
        SearchResult(
            title="AI Research Paper 2024",
            content="Machine learning is a subset of artificial intelligence...",
            source_type=SourceType.DOCUMENT,
            relevance_score=0.85,
            source_name="AI Research Paper 2024"
        ),
        SearchResult(
            title="AI Developments",
            content="Recent developments in LLMs show promising results...",
            source_type=SourceType.WEB,
            relevance_score=0.72,
            source_name="TechNews.com",
            url="https://technews.com/article"
        )
    ]
    
    # Create response
    response = QueryResponse(
        query=query_request.query,
        answer="Based on the latest research, machine learning continues to evolve rapidly...",
        response_strategy=strategy,
        sources=search_results,
        processing_time=time.time() - start_time,
        confidence_score=0.78
    )
    
    return response


@app.post("/search", response_model=List[HybridSearchResult])
async def hybrid_search(
    query: str,
    limit: int = 10,
    include_web: bool = True,
    include_documents: bool = True,
    user: Optional[dict] = Depends(get_current_user)
) -> List[HybridSearchResult]:
    """
    Perform hybrid search across documents and web.
    
    For beginners: This shows how to use query parameters with defaults.
    """
    # TODO: Implement actual hybrid search
    # For now, return mock results
    
    doc_result = SearchResult(
        title="AI Industry Report 2024",
        content="Artificial intelligence is transforming industries...",
        source_type=SourceType.DOCUMENT,
        relevance_score=0.92,
        dense_score=0.88,
        sparse_score=0.85,
        freshness_score=0.95,
        source_name="AI Industry Report 2024"
    )
    
    web_result = SearchResult(
        title="AI Breakthrough News",
        content="Latest AI breakthrough announced by researchers...",
        source_type=SourceType.WEB,
        relevance_score=0.87,
        dense_score=0.83,
        sparse_score=0.78,
        freshness_score=0.99,
        source_name="Science Daily",
        url="https://sciencedaily.com/ai-breakthrough"
    )
    
    mock_results = [
        HybridSearchResult(
            document_results=[doc_result] if include_documents else [],
            web_results=[web_result] if include_web else [],
            combined_results=[doc_result, web_result],
            total_results=2,
            search_time=0.5,
            response_strategy=ResponseStrategy.MIXED,
            max_document_similarity=0.85
        )
    ]
    
    # Apply filters
    if not include_web:
        mock_results = [r for r in mock_results if r.source_type != SourceType.WEB]
    if not include_documents:
        mock_results = [r for r in mock_results if r.source_type != SourceType.DOCUMENT]
    
    return mock_results[:limit]


# ==============================================
# Streaming Response Endpoint
# ==============================================

@app.post("/query/stream")
async def stream_query_response(
    query_request: QueryRequest,
    user: Optional[dict] = Depends(get_current_user)
):
    """
    Stream query response for real-time updates.
    
    For beginners: This shows how to create streaming responses for long-running tasks.
    """
    async def generate_response():
        # Simulate streaming response
        steps = [
            "🔍 Searching documents...",
            "🌐 Fetching web results...",
            "🤖 Generating response...",
            "✅ Complete!"
        ]
        
        for step in steps:
            yield f"data: {json.dumps({'step': step, 'timestamp': datetime.now().isoformat()})}\n\n"
            # Simulate processing time
            await asyncio.sleep(1)
        
        # Final response
        final_response = {
            "response": "This is the final generated response based on your query.",
            "completed": True
        }
        yield f"data: {json.dumps(final_response)}\n\n"
    
    return StreamingResponse(
        generate_response(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )


# ==============================================
# Analytics & Monitoring Endpoints
# ==============================================

@app.get("/analytics/stats")
async def get_analytics_stats(
    user: Optional[dict] = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Get system analytics and usage statistics.
    
    For beginners: This shows how to return complex nested data structures.
    """
    # TODO: Implement real analytics
    return {
        "total_documents": 1250,
        "total_queries": 5643,
        "average_response_time": 1.8,
        "popular_topics": ["AI", "machine learning", "data science"],
        "user_activity": {
            "daily_active_users": 120,
            "weekly_active_users": 450,
            "monthly_active_users": 1200
        },
        "performance_metrics": {
            "embedding_generation_time": 0.5,
            "search_time": 0.3,
            "llm_response_time": 1.0
        }
    }


# ==============================================
# Error Handlers
# ==============================================

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """
    Global exception handler for unexpected errors.
    
    For beginners: This catches any unexpected errors and returns a nice response.
    """
    logging.error(f"Unexpected error: {exc}")
    
    if settings.development:
        # In development, show the actual error
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal Server Error",
                "message": str(exc),
                "type": type(exc).__name__
            }
        )
    else:
        # In production, hide the error details
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal Server Error",
                "message": "Something went wrong. Please try again later."
            }
        )


# ==============================================
# Development Server
# ==============================================

def main():
    """
    Main function to run the development server.
    
    For beginners: This function starts the web server when you run this file.
    """
    print("🎯 Starting Research Assistant RAG API Server...")
    
    # Setup logging
    setup_logging()
    
    # Run the server
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.reload,
        log_level=settings.log_level.lower()
    )


if __name__ == "__main__":
    main()
