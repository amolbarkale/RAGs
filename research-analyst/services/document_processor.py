"""
Document processing service for extracting text, chunking, and creating embeddings.

This service handles the complete pipeline from uploaded files to searchable vectors.
For beginners: This is where we convert documents into searchable embeddings.
"""

import logging
import asyncio
import os
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import hashlib
from pathlib import Path

# LangChain imports
from langchain_core.documents import Document as LangChainDocument
from langchain_text_splitters import RecursiveCharacterTextSplitter

# LangChain Document Loaders
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import UnstructuredWordDocumentLoader
from langchain_community.document_loaders.csv_loader import CSVLoader

# SQLAlchemy imports
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update

# Core imports
from core.config import settings
from core.models import DocumentType, DocumentChunk, ChunkLevel
from services.models import DocumentDBModel, DocumentChunkDBModel
from services.vector_store import VectorStoreService

# Set up logging
logger = logging.getLogger(__name__)


class DocumentProcessor:
    """
    Service for processing uploaded documents.
    
    For beginners: This class handles the complete pipeline:
    1. Extract text from files (PDF, TXT, DOCX)
    2. Split text into chunks
    3. Generate embeddings
    4. Store in vector database
    """
    
    def __init__(self, vector_store: VectorStoreService):
        self.vector_store = vector_store
        self.text_splitter = None
        self._initialized = False
    
    async def initialize(self):
        """Initialize the document processor."""
        if self._initialized:
            return
        
        logger.info("🔧 Initializing document processor...")
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size_medium,
            chunk_overlap=settings.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""],
            add_start_index=True
        )
        
        # Ensure vector store is initialized
        await self.vector_store.initialize()
        
        self._initialized = True
        logger.info("✅ Document processor initialized!")
    
    async def process_document(
        self,
        document_id: str,
        file_path: str,
        file_type: DocumentType,
        db: AsyncSession
    ) -> Dict[str, Any]:
        """
        Process a document through the complete pipeline.
        
        For beginners: This is the main function that handles everything:
        1. Extract text from the file
        2. Split into chunks
        3. Generate embeddings
        4. Store in vector database
        5. Update database records
        """
        if not self._initialized:
            await self.initialize()
        
        logger.info(f"📄 Processing document {document_id} ({file_type.value})")
        
        try:
            # Step 1: Extract text from file
            logger.info("📖 Step 1: Extracting text from file...")
            raw_text = await self._extract_text(file_path, file_type)
            
            if not raw_text or len(raw_text.strip()) == 0:
                raise ValueError("No text content found in document")
            
            logger.info(f"✅ Extracted {len(raw_text)} characters")
            
            # Step 2: Update document with extracted text
            logger.info("💾 Step 2: Updating document record...")
            await self._update_document_text(document_id, raw_text, db)
            
            # Step 3: Create chunks
            logger.info("✂️ Step 3: Creating document chunks...")
            chunks = await self._create_chunks(document_id, raw_text, db)
            
            logger.info(f"✅ Created {len(chunks)} chunks")
            
            # Step 4: Generate embeddings and store in vector database
            logger.info("🧠 Step 4: Generating embeddings...")
            embeddings = await self.vector_store.create_document_embeddings(
                document_id=document_id,
                text=raw_text,
                metadata={
                    "file_type": file_type.value,
                    "processing_date": datetime.now().isoformat()
                }
            )
            
            logger.info(f"✅ Generated {len(embeddings)} embeddings")
            
            # Step 5: Mark document as processed
            logger.info("✅ Step 5: Marking document as processed...")
            await self._mark_document_processed(
                document_id, 
                len(chunks), 
                len(embeddings), 
                db
            )
            
            # Return processing results
            result = {
                "document_id": document_id,
                "status": "completed",
                "text_length": len(raw_text),
                "chunk_count": len(chunks),
                "embedding_count": len(embeddings),
                "processing_time": 0.0,  # TODO: Calculate actual time
                "message": "Document processed successfully"
            }
            
            logger.info(f"✅ Document {document_id} processed successfully!")
            return result
            
        except Exception as e:
            logger.error(f"❌ Failed to process document {document_id}: {e}")
            
            # Mark document as failed
            await self._mark_document_failed(document_id, str(e), db)
            
            raise
    
    async def _extract_text(self, file_path: str, file_type: DocumentType) -> str:
        """
        Extract text using LangChain document loaders.
        
        For beginners: LangChain loaders handle all the complexity for us!
        They provide better error handling, encoding detection, and metadata extraction.
        """
        logger.info(f"📖 Extracting text from {file_type.value} file using LangChain loaders: {file_path}")
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        try:
            # Use appropriate LangChain loader based on file type
            if file_type == DocumentType.TXT:
                loader = TextLoader(file_path, autodetect_encoding=True)
            elif file_type == DocumentType.PDF:
                loader = PyPDFLoader(file_path)
            elif file_type == DocumentType.DOCX:
                loader = UnstructuredWordDocumentLoader(file_path)
            elif file_type == DocumentType.CSV:
                loader = CSVLoader(file_path=file_path)
            else:
                raise ValueError(f"Unsupported file type: {file_type}")
            
            # Load documents using LangChain loader
            logger.info(f"🔍 Loading document with {type(loader).__name__}...")
            documents = await asyncio.get_event_loop().run_in_executor(
                None, loader.load
            )
            
            if not documents:
                raise ValueError("No content extracted from document")
            
            # Combine all document pages/sections into single text
            full_text = "\n\n".join([doc.page_content for doc in documents if doc.page_content.strip()])
            
            if not full_text.strip():
                raise ValueError("No text content found in document")
            
            logger.info(f"✅ Extracted {len(full_text)} characters using LangChain {type(loader).__name__}")
            logger.info(f"📄 Processed {len(documents)} document sections/pages")
            
            return full_text
                
        except Exception as e:
            logger.error(f"❌ LangChain text extraction failed: {e}")
            raise
    

    
    async def _create_chunks(
        self, 
        document_id: str, 
        text: str, 
        db: AsyncSession
    ) -> List[DocumentChunkDBModel]:
        """
        Split text into chunks and save to database.
        
        For beginners: This breaks large documents into smaller pieces.
        """
        logger.info(f"✂️ Creating chunks for document {document_id}")
        
        try:
            # Create LangChain document
            doc = LangChainDocument(
                page_content=text,
                metadata={"document_id": document_id}
            )
            
            # Split into chunks
            if not self.text_splitter:
                raise RuntimeError("Text splitter not initialized")
            
            chunks = await asyncio.get_event_loop().run_in_executor(
                None, self.text_splitter.split_documents, [doc]
            )
            
            # Save chunks to database
            chunk_models = []
            for i, chunk in enumerate(chunks):
                chunk_model = DocumentChunkDBModel(
                    document_id=document_id,
                    chunk_index=i,
                    text=chunk.page_content,
                    chunk_level=ChunkLevel.MEDIUM.value,
                    token_count=len(chunk.page_content.split()),
                    start_char=chunk.metadata.get("start_index", 0),
                    end_char=chunk.metadata.get("start_index", 0) + len(chunk.page_content),
                    metadata={
                        "chunk_size": len(chunk.page_content),
                        "created_at": datetime.now().isoformat()
                    }
                )
                
                db.add(chunk_model)
                chunk_models.append(chunk_model)
            
            await db.commit()
            
            logger.info(f"✅ Created {len(chunk_models)} chunks in database")
            return chunk_models
            
        except Exception as e:
            logger.error(f"❌ Failed to create chunks: {e}")
            raise
    
    async def _update_document_text(
        self, 
        document_id: str, 
        raw_text: str, 
        db: AsyncSession
    ):
        """Update document with extracted text."""
        try:
            stmt = update(DocumentDBModel).where(
                DocumentDBModel.id == document_id
            ).values(
                raw_text=raw_text,
                word_count=len(raw_text.split()),
                updated_at=datetime.now()
            )
            
            await db.execute(stmt)
            await db.commit()
            
            logger.info(f"✅ Updated document {document_id} with extracted text")
            
        except Exception as e:
            logger.error(f"❌ Failed to update document text: {e}")
            raise
    
    async def _mark_document_processed(
        self, 
        document_id: str, 
        chunk_count: int, 
        embedding_count: int, 
        db: AsyncSession
    ):
        """Mark document as successfully processed."""
        try:
            stmt = update(DocumentDBModel).where(
                DocumentDBModel.id == document_id
            ).values(
                is_processed=True,
                chunk_count=chunk_count,
                embedding_count=embedding_count,
                processing_completed_at=datetime.now(),
                updated_at=datetime.now()
            )
            
            await db.execute(stmt)
            await db.commit()
            
            logger.info(f"✅ Marked document {document_id} as processed")
            
        except Exception as e:
            logger.error(f"❌ Failed to mark document as processed: {e}")
            raise
    
    async def _mark_document_failed(
        self, 
        document_id: str, 
        error_message: str, 
        db: AsyncSession
    ):
        """Mark document as failed processing."""
        try:
            stmt = update(DocumentDBModel).where(
                DocumentDBModel.id == document_id
            ).values(
                is_processed=False,
                processing_error=error_message,
                updated_at=datetime.now()
            )
            
            await db.execute(stmt)
            await db.commit()
            
            logger.info(f"❌ Marked document {document_id} as failed")
            
        except Exception as e:
            logger.error(f"❌ Failed to mark document as failed: {e}")
    
    async def get_processing_stats(self) -> Dict[str, Any]:
        """Get document processing statistics."""
        return {
            "processor_initialized": self._initialized,
            "supported_formats": [t.value for t in DocumentType],
            "chunk_size": settings.chunk_size_medium,
            "chunk_overlap": settings.chunk_overlap,
            "embedding_model": settings.embedding_model
        }


# ==============================================
# Utility Functions
# ==============================================

async def create_document_processor(vector_store: VectorStoreService) -> DocumentProcessor:
    """Create and initialize a document processor."""
    processor = DocumentProcessor(vector_store)
    await processor.initialize()
    return processor


def detect_file_type(filename: str) -> DocumentType:
    """Detect file type from filename extension."""
    extension = Path(filename).suffix.lower()
    
    if extension == '.pdf':
        return DocumentType.PDF
    elif extension == '.txt':
        return DocumentType.TXT
    elif extension in ['.docx', '.doc']:
        return DocumentType.DOCX
    elif extension == '.csv':
        return DocumentType.CSV
    elif extension in ['.md', '.markdown']:
        return DocumentType.MARKDOWN
    else:
        raise ValueError(f"Unsupported file type: {extension}")


if __name__ == "__main__":
    # Test the document processor
    async def test_processor():
        from services.vector_store import create_vector_store
        from qdrant_client import QdrantClient
        
        # Create vector store
        qdrant_client = QdrantClient(":memory:")
        vector_store = await create_vector_store(qdrant_client)
        
        # Create processor
        processor = await create_document_processor(vector_store)
        
        # Test stats
        stats = await processor.get_processing_stats()
        print("Document Processor Stats:", stats)
    
    asyncio.run(test_processor()) 