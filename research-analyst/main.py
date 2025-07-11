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
    SearchResult, HybridSearchResult, SourceType,
    # Add our new request/response models
    DocumentUploadRequest, DocumentUploadResponse,
    DocumentFilterRequest, DocumentListResponse,
    SearchRequest
)
from utils.logger import setup_logging

# Set up logging
logger = logging.getLogger(__name__)

# Database imports
from sqlalchemy.ext.asyncio import AsyncSession
from services.database import db_manager, get_db, get_vector_db
from services.vector_store import VectorStoreService, create_vector_store
from services.models import DocumentDBModel, DocumentChunkDBModel
from services.document_processor import ProductionDocumentProcessor, create_document_processor, detect_file_type
from services.search_service import SearchService, create_search_service
from services.langchain_llm_service import create_llm_service
from services.tavily_search_service import create_tavily_service


# ==============================================
# Service Safety Functions
# ==============================================

def check_service_availability(service_name: str) -> bool:
    """Check if a service is available and initialized."""
    if not hasattr(app.state, 'services'):
        return False
    return app.state.services.get(service_name) is not None

def get_service_or_error(service_name: str):
    """Get a service or raise an appropriate error."""
    if not hasattr(app.state, 'services'):
        raise HTTPException(
            status_code=503, 
            detail=f"Services not initialized. Service '{service_name}' unavailable."
        )
    
    service = app.state.services.get(service_name)
    if service is None:
        raise HTTPException(
            status_code=503,
            detail=f"Service '{service_name}' is not available. Check system status at /health"
        )
    
    return service


# ==============================================
# Application Lifecycle Management
# ==============================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manage application startup and shutdown using robust initialization.
    
    For beginners: This function runs when the app starts and stops.
    It uses a robust initialization system that handles errors gracefully.
    """
    # Startup
    print("🚀 Starting Research Assistant RAG system...")
    print_config_summary()
    
    try:
        # Use robust initialization system
        from services.robust_init import initialize_application, cleanup_application
        
        print("🔧 Initializing all services robustly...")
        init_result = await initialize_application()
        
        if not init_result['success']:
            print("⚠️ Some services failed to initialize, but continuing with available services...")
            for error in init_result['errors']:
                print(f"   ❌ {error}")
        
        # Store services in app state
        services = init_result['services']
        app.state.services = services
        app.state.initialization_status = init_result['status']
        
        # Store individual services for backward compatibility
        app.state.vector_store = services.get('vector_store')
        app.state.document_processor = services.get('document_processor')
        app.state.search_service = services.get('search_service')
        app.state.llm_service = services.get('llm_service')
        app.state.tavily_service = services.get('tavily_service')
        app.state.db_manager = services.get('db_manager')
        
        # Print initialization status
        print("📊 Service Status:")
        for service, status in init_result['status'].items():
            status_icon = "✅" if status == 'success' else "⚠️" if status in ['fallback', 'minimal'] else "❌"
            print(f"   {status_icon} {service}: {status}")
        
        # Check overall health
        if app.state.db_manager:
            try:
                health = await app.state.db_manager.health_check()
                print(f"📊 Database Health: Traditional DB: {'✅' if health.get('traditional_db') else '❌'}, Vector DB: {'✅' if health.get('vector_db') else '❌'}")
            except Exception as e:
                print(f"⚠️ Health check failed: {e}")
        
        print("✅ Application startup complete!")
        
    except Exception as e:
        print(f"❌ Critical startup failure: {e}")
        # Continue anyway - let the app start in degraded mode
        app.state.services = {}
        app.state.initialization_status = {'critical_error': str(e)}
    
    yield  # This is where the app runs
    
    # Shutdown
    print("🔄 Shutting down Research Assistant...")
    
    try:
        # Use robust cleanup
        if hasattr(app.state, 'services'):
            await cleanup_application(app.state.services)
        else:
            print("⚠️ No services to cleanup")
    except Exception as e:
        print(f"⚠️ Cleanup error: {e}")
    
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
    Health check endpoint for monitoring with robust error handling.
    
    For beginners: This endpoint tells us if the system is healthy.
    Useful for monitoring and deployment systems.
    """
    health_data = {
        "status": "unknown",
        "version": settings.version,
        "environment": "development" if settings.development else "production",
        "timestamp": datetime.now().isoformat(),
        "services": {},
        "initialization_status": getattr(app.state, 'initialization_status', {})
    }
    
    # Check if services were initialized
    if not hasattr(app.state, 'services'):
        health_data["status"] = "degraded"
        health_data["services"]["error"] = "Services not initialized"
        return health_data
    
    services_healthy = 0
    total_services = 0
    
    # Check database health
    try:
        if app.state.services.get('db_manager'):
            db_health = await app.state.services['db_manager'].health_check()
            health_data["services"]["database"] = db_health
            if db_health.get("traditional_db") and db_health.get("vector_db"):
                services_healthy += 1
        else:
            health_data["services"]["database"] = "unavailable"
        total_services += 1
    except Exception as e:
        health_data["services"]["database"] = {"error": str(e)}
    
    # Check vector store health
    try:
        if app.state.services.get('vector_store'):
            vector_health = await app.state.services['vector_store'].health_check()
            health_data["services"]["vector_store"] = vector_health
            if vector_health.get("status") == "healthy":
                services_healthy += 1
        else:
            health_data["services"]["vector_store"] = "unavailable"
        total_services += 1
    except Exception as e:
        health_data["services"]["vector_store"] = {"error": str(e)}
    
    # Check other services
    service_checks = [
        ("llm_service", "llm"),
        ("tavily_service", "web_search"),
        ("search_service", "search"),
        ("document_processor", "document_processing")
    ]
    
    for service_key, display_name in service_checks:
        if app.state.services.get(service_key):
            health_data["services"][display_name] = "available"
            services_healthy += 1
        else:
            health_data["services"][display_name] = "unavailable"
        total_services += 1
    
    # Determine overall health
    if services_healthy == total_services:
        health_data["status"] = "healthy"
    elif services_healthy > total_services // 2:
        health_data["status"] = "degraded"
    else:
        health_data["status"] = "unhealthy"
    
    health_data["services_healthy"] = f"{services_healthy}/{total_services}"
    
    return health_data


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

@app.post("/documents/upload", response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    title: Optional[str] = Form(None),
    description: Optional[str] = Form(None),
    tags: Optional[str] = Form(None),
    user: Optional[dict] = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
) -> DocumentUploadResponse:
    """
    Upload a document (PDF, TXT, DOCX) using Pydantic validation.
    
    For beginners: This shows how FastAPI + Pydantic automatically validates
    file uploads and parameters. No manual validation needed!
    """
    # Read file content
    content = await file.read()
    
    # Parse tags if provided
    parsed_tags = []
    if tags:
        parsed_tags = [tag.strip() for tag in tags.split(",") if tag.strip()]
    
    # Detect file type from extension
    try:
        detected_file_type = detect_file_type(file.filename or "unknown")
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Unsupported file type: {str(e)}"
        )

    # Create a request model to validate all parameters automatically
    try:
        # Let Pydantic handle ALL validation automatically!
        upload_request = DocumentUploadRequest(
            filename=file.filename or "unknown",
            file_type=detected_file_type,  # ✅ Now detects from extension
            file_size=len(content),
            title=title,
            description=description,
            tags=parsed_tags
        )
    except ValueError as e:
        # Pydantic validation failed - return clear error
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Validation error: {str(e)}"
        )
    
    # If we get here, ALL validation passed automatically!
    
    # Save file to disk
    file_path = f"data/{upload_request.filename}"
    with open(file_path, "wb") as f:
        f.write(content)
    
    # Generate content hash for deduplication
    import hashlib
    content_hash = hashlib.sha256(content).hexdigest()
    
    # Create database document record FIRST (without processing)
    db_document = DocumentDBModel(
        filename=upload_request.filename,
        original_filename=file.filename or "unknown",
        file_type=upload_request.file_type.value,
        file_size=upload_request.file_size,
        file_path=file_path,
        content_hash=content_hash,
        raw_text=None,  # Will be filled during processing
        metadata={
            "title": upload_request.title,
            "description": upload_request.description,
            "tags": upload_request.tags,
            "uploaded_by": user.get("username") if user else "anonymous"
        },
        word_count=0,  # Will be calculated during processing
        owner_id=user.get("user_id") if user else None,
        is_processed=False,
        processing_started_at=datetime.now(),  # Mark processing as started
        chunk_count=0,
        embedding_count=0
    )
    
    # Save document to database
    db.add(db_document)
    await db.commit()
    await db.refresh(db_document)
    
    # ✅ NOW PROCESS THE DOCUMENT THROUGH THE PIPELINE
    logger.info(f"🚀 Starting document processing for {db_document.id}")
    
    try:
        # Get document processor from app state using safety function
        document_processor = get_service_or_error('document_processor')
        
        # Process document through complete pipeline
        processing_result = await document_processor.process_document(
            document_id=str(db_document.id),
            file_path=file_path,
            file_type=upload_request.file_type,
            db=db
        )
        
        logger.info(f"✅ Document processing completed: {processing_result}")
        
        # Return SUCCESS response with real processing data
        return DocumentUploadResponse(
            document_id=str(db_document.id),
            filename=upload_request.filename,
            file_size=upload_request.file_size,
            file_type=upload_request.file_type,
            processing_status="completed",  # ✅ Real status
            estimated_processing_time=processing_result.get("processing_time", 0.0),
            estimated_chunk_count=processing_result.get("chunk_count", 0),
            message=f"Document '{upload_request.filename}' processed successfully! "
                   f"Generated {processing_result.get('chunk_count', 0)} chunks and "
                   f"{processing_result.get('embedding_count', 0)} embeddings."
        )
        
    except Exception as e:
        logger.error(f"❌ Document processing failed: {e}")
        
        # Return FAILED response with error details
        return DocumentUploadResponse(
            document_id=str(db_document.id),
            filename=upload_request.filename,
            file_size=upload_request.file_size,
            file_type=upload_request.file_type,
            processing_status="failed",
            estimated_processing_time=0.0,
            estimated_chunk_count=0,
            message=f"Document upload succeeded but processing failed: {str(e)}"
        )


@app.get("/documents", response_model=DocumentListResponse)
async def list_documents(
    filter_request: DocumentFilterRequest = Depends(),
    user: Optional[dict] = Depends(get_current_user)
) -> DocumentListResponse:
    """
    List documents with automatic Pydantic validation.
    
    For beginners: This shows how to use dependency injection with Pydantic
    for automatic query parameter validation.
    """
    # All validation is handled automatically by Pydantic!
    # No manual checks needed for limit, offset, etc.
    
    # TODO: Implement actual database query
    # For now, return mock data
    
    mock_documents = [
        Document(
            filename="ai_report_2024.pdf",
            original_filename="AI Industry Report 2024.pdf",
            file_type=DocumentType.PDF,
            file_size=2048576,  # 2MB
            file_path="uploads/ai_report_2024.pdf",
            content_hash="hash_1",
            raw_text="AI industry analysis...",
            metadata=DocumentMetadata(
                title="AI Industry Report 2024",
                page_count=50,
                word_count=12500
            ),
            is_processed=True,
            chunk_count=25,
            embedding_count=25
        ),
        Document(
            filename="ml_tutorial.txt",
            original_filename="Machine Learning Tutorial.txt",
            file_type=DocumentType.TXT,
            file_size=512000,  # 500KB
            file_path="uploads/ml_tutorial.txt",
            content_hash="hash_2",
            raw_text="Machine learning tutorial content...",
            metadata=DocumentMetadata(
                title="Machine Learning Tutorial",
                word_count=3200
            ),
            is_processed=True,
            chunk_count=8,
            embedding_count=8
        )
    ]
    
    # Apply filters (with automatic validation)
    filtered_docs = mock_documents
    if filter_request.document_type:
        filtered_docs = [d for d in filtered_docs if d.file_type == filter_request.document_type]
    
    if filter_request.title_contains:
        filtered_docs = [d for d in filtered_docs 
                        if filter_request.title_contains.lower() in (d.metadata.title or "").lower()]
    
    # Apply pagination (already validated by Pydantic)
    total_count = len(filtered_docs)
    paginated_docs = filtered_docs[filter_request.offset:filter_request.offset + filter_request.limit]
    
    return DocumentListResponse(
        documents=paginated_docs,
        total_count=total_count,
        offset=filter_request.offset,
        limit=filter_request.limit,
        message=f"Found {total_count} documents"
    )


@app.delete("/documents/{document_id}")
async def delete_document(
    document_id: str,
    user: Optional[dict] = Depends(get_current_user)
) -> Dict[str, str]:
    """
    Delete a document by ID.
    
    For beginners: This shows path parameter validation.
    """
    # TODO: Implement actual document deletion
    # For now, return success message
    
    return {
        "message": f"Document {document_id} deleted successfully",
        "document_id": document_id
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
    Process a user query using the production search service.
    
    This endpoint performs hybrid search across documents and web sources,
    then generates a comprehensive response with citations.
    """
    try:
        # Get search service from app state using safety function
        search_service: SearchService = get_service_or_error('search_service')
        
        # Process the query using the search service
        response = await search_service.process_query(query_request)
        
        logger.info(f"✅ Query processed successfully: {response.result_count} results, {response.processing_time:.2f}s")
        return response
        
    except Exception as e:
        logger.error(f"❌ Query processing failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Query processing failed: {str(e)}"
        )


@app.post("/search", response_model=List[HybridSearchResult])
async def hybrid_search(
    search_request: SearchRequest,
    user: Optional[dict] = Depends(get_current_user)
) -> List[HybridSearchResult]:
    """
    Perform hybrid search using the production search service.
    
    This endpoint searches across documents and web sources based on the
    search request parameters, returning structured results with scoring.
    """
    try:
        # Get search service from app state using safety function
        search_service: SearchService = get_service_or_error('search_service')
        
        # Perform hybrid search using the search service
        results = await search_service.hybrid_search(search_request)
        
        logger.info(f"✅ Hybrid search completed: {len(results)} result sets")
        return results
        
    except Exception as e:
        logger.error(f"❌ Hybrid search failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Hybrid search failed: {str(e)}"
        )


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
