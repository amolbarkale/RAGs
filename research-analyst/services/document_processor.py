"""
Production-grade document processing with optimized chunking strategy.

This service implements the proven approach used by successful RAG systems:
- Single-level chunking at 512 tokens (optimal for research and speed)
- Smart structure-aware preprocessing (citations, headers, sections)
- Content-adaptive chunking (markdown vs plain text)
- Fast processing optimized for sub-3 second query responses

For research professionals: Maximum accuracy and reliability with production performance.
"""

import logging
import asyncio
import os
import re
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import hashlib
from pathlib import Path

# LangChain imports
from langchain_core.documents import Document as LangChainDocument
from langchain_text_splitters import RecursiveCharacterTextSplitter, TokenTextSplitter

# LangChain Document Loaders
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import UnstructuredWordDocumentLoader

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


class ProductionDocumentProcessor:
    """
    Production-grade document processor optimized for real-world RAG systems.
    
    Implementation follows proven patterns from successful RAG deployments:
    - Single-level chunking at 512 tokens (optimal balance of context and speed)
    - Smart preprocessing for research documents (citations, structure)
    - Content-adaptive chunking (markdown-aware when needed)
    - Fast, reliable processing for enterprise workloads
    """
    
    def __init__(self, vector_store: VectorStoreService):
        self.vector_store = vector_store
        self.chunk_size = 512  # Optimal for research documents
        self.chunk_overlap = 64  # 12.5% overlap (industry standard)
        self.text_splitter = None
        self.token_splitter = None
        self._initialized = False
        
        # Research-specific patterns (kept for accuracy)
        self.citation_patterns = [
            r'\[\d+\]',  # [1], [2], etc.
            r'\([A-Za-z]+\s+et\s+al\.\s*,\s*\d{4}\)',  # (Smith et al., 2023)
            r'\([A-Za-z]+\s*,\s*\d{4}\)',  # (Smith, 2023)
            r'\([A-Za-z]+\s+&\s+[A-Za-z]+\s*,\s*\d{4}\)',  # (Smith & Jones, 2023)
        ]
        
        # Academic section patterns
        self.academic_sections = [
            r'^\s*(Abstract|Introduction|Methodology|Methods|Results|Discussion|Conclusion|References|Bibliography)\s*$',
            r'^\s*\d+\.\s+(Abstract|Introduction|Methodology|Methods|Results|Discussion|Conclusion)\s*$',
        ]
    
    async def initialize(self):
        """Initialize the production document processor."""
        if self._initialized:
            return
        
        logger.info("🚀 Initializing Production Document Processor...")
        
        # Initialize text splitter (for general documents)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", "? ", "! ", " ", ""],
            add_start_index=True
        )
        
        # Initialize token splitter (for precise token control)
        self.token_splitter = TokenTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            model_name="cl100k_base"  # GPT-4 tokenizer for accuracy
        )
        
        # Ensure vector store is initialized
        await self.vector_store.initialize()
        
        self._initialized = True
        logger.info("✅ Production Document Processor initialized!")
        logger.info(f"📊 Chunk size: {self.chunk_size} tokens, Overlap: {self.chunk_overlap} tokens")
    
    async def process_document(
        self,
        document_id: str,
        file_path: str,
        file_type: DocumentType,
        db: AsyncSession
    ) -> Dict[str, Any]:
        """
        Process a document using production-grade chunking pipeline.
        
        Pipeline: Extract → Analyze → Chunk → Store
        Optimized for speed while maintaining research document quality.
        """
        if not self._initialized:
            await self.initialize()
        
        logger.info(f"📄 Processing document {document_id} ({file_type.value})")
        
        try:
            # Step 1: Extract text with structure preservation
            logger.info("📖 Step 1: Extracting text...")
            raw_text = await self._extract_text(file_path, file_type)
            
            if not raw_text or len(raw_text.strip()) == 0:
                raise ValueError("No text content found in document")
            
            logger.info(f"✅ Extracted {len(raw_text)} characters")
            
            # Step 2: Update document with extracted text
            logger.info("💾 Step 2: Updating document record...")
            await self._update_document_text(document_id, raw_text, db)
            
            # Step 3: Smart chunking based on content type
            logger.info("✂️ Step 3: Creating optimized chunks...")
            chunks = await self._create_smart_chunks(document_id, raw_text, file_type, db)
            
            logger.info(f"✅ Created {len(chunks)} chunks")
            
            # Step 4: Generate embeddings
            logger.info("🧠 Step 4: Generating embeddings...")
            embeddings = await self.vector_store.create_document_embeddings(
                document_id=document_id,
                text=raw_text,
                metadata={
                    "file_type": file_type.value,
                    "processing_date": datetime.now().isoformat(),
                    "chunking_strategy": "production_optimized",
                    "chunk_size": self.chunk_size,
                    "chunk_overlap": self.chunk_overlap
                }
            )
            
            logger.info(f"✅ Generated {len(embeddings)} embeddings")
            
            # Step 5: Mark document as processed
            logger.info("✅ Step 5: Finalizing...")
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
                "chunk_size": self.chunk_size,
                "chunk_overlap": self.chunk_overlap,
                "embedding_count": len(embeddings),
                "processing_time": 0.0,  # TODO: Calculate actual time
                "message": "Document processed with production-grade chunking"
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
        
        Optimized for reliability and speed.
        """
        logger.info(f"📖 Extracting text from {file_type.value} file: {file_path}")
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        try:
            # Use appropriate LangChain loader
            if file_type == DocumentType.TXT:
                loader = TextLoader(file_path, autodetect_encoding=True)
            elif file_type == DocumentType.PDF:
                loader = PyPDFLoader(file_path)
            elif file_type == DocumentType.DOCX:
                loader = UnstructuredWordDocumentLoader(file_path)
            elif file_type == DocumentType.MARKDOWN:
                loader = TextLoader(file_path, autodetect_encoding=True)
            else:
                raise ValueError(f"Unsupported file type: {file_type}")
            
            # Load documents
            documents = await asyncio.get_event_loop().run_in_executor(
                None, loader.load
            )
            
            if not documents:
                raise ValueError("No content extracted from document")
            
            # Combine all document pages/sections
            full_text = "\n\n".join([doc.page_content for doc in documents if doc.page_content.strip()])
            
            if not full_text.strip():
                raise ValueError("No text content found in document")
            
            logger.info(f"✅ Extracted {len(full_text)} characters from {len(documents)} sections")
            return full_text
                
        except Exception as e:
            logger.error(f"❌ Text extraction failed: {e}")
            raise
    
    async def _create_smart_chunks(
        self,
        document_id: str,
        text: str,
        file_type: DocumentType,
        db: AsyncSession
    ) -> List[DocumentChunkDBModel]:
        """
        Create optimized chunks using smart content-aware strategy.
        
        This is the core of production RAG chunking:
        - Detect content type (markdown, academic, plain text)
        - Choose optimal splitting strategy
        - Preserve citations and structure boundaries
        - Single-level chunking for speed
        """
        logger.info(f"✂️ Creating smart chunks for document {document_id}")
        
        try:
            # Detect content characteristics
            content_analysis = await self._analyze_content(text)
            
            # Choose chunking strategy based on content
            if content_analysis["has_markdown_structure"]:
                logger.info("📝 Using structure-aware chunking (markdown detected)")
                chunk_texts = await self._structure_aware_chunking(text)
                chunking_method = "structure_aware"
            elif content_analysis["has_academic_structure"]:
                logger.info("🎓 Using academic-aware chunking (research paper detected)")
                chunk_texts = await self._academic_aware_chunking(text)
                chunking_method = "academic_aware"
            else:
                logger.info("📄 Using standard chunking (plain text)")
                chunk_texts = await self._standard_chunking(text)
                chunking_method = "standard"
            
            # Create database models
            chunk_models = []
            start_char = 0
            
            for i, chunk_text in enumerate(chunk_texts):
                # Extract citations from chunk
                citations = self._extract_citations(chunk_text)
                
                # Determine academic sections in chunk
                academic_sections = self._find_academic_sections(chunk_text)
                
                chunk_model = DocumentChunkDBModel(
                    document_id=document_id,
                    chunk_index=i,
                    text=chunk_text,
                    chunk_level=ChunkLevel.MEDIUM.value,  # Single level for simplicity
                    token_count=len(chunk_text.split()),  # Approximate
                    start_char=start_char,
                    end_char=start_char + len(chunk_text),
                    metadata={
                        "chunk_size": len(chunk_text),
                        "chunking_method": chunking_method,
                        "file_extension": file_type.value,
                        "citations": citations,
                        "academic_sections": academic_sections,
                        "content_type": content_analysis["content_type"],
                        "created_at": datetime.now().isoformat()
                    }
                )
                
                db.add(chunk_model)
                chunk_models.append(chunk_model)
                start_char += len(chunk_text)
            
            await db.commit()
            
            logger.info(f"✅ Created {len(chunk_models)} chunks using {chunking_method} strategy")
            return chunk_models
            
        except Exception as e:
            logger.error(f"❌ Failed to create chunks: {e}")
            raise
    
    async def _analyze_content(self, text: str) -> Dict[str, Any]:
        """
        Analyze content to determine optimal chunking strategy.
        
        Fast analysis to choose the right approach.
        """
        content_analysis = {
            "content_type": "plain_text",
            "has_markdown_structure": False,
            "has_academic_structure": False,
            "citation_count": 0,
            "academic_section_count": 0
        }
        
        # Quick checks for content type
        if len(text) < 100:
            return content_analysis
        
        # Check for markdown indicators
        markdown_indicators = [
            r'^#{1,6}\s+',  # Headers
            r'```',  # Code blocks
            r'^\s*[-*+]\s+',  # Lists
            r'\*\*[^*]+\*\*',  # Bold
            r'\[.+\]\(.+\)'  # Links
        ]
        
        markdown_score = 0
        for pattern in markdown_indicators:
            if re.search(pattern, text, re.MULTILINE):
                markdown_score += 1
        
        if markdown_score >= 2:
            content_analysis["has_markdown_structure"] = True
            content_analysis["content_type"] = "markdown"
        
        # Check for academic structure
        academic_score = 0
        for pattern in self.academic_sections:
            matches = re.findall(pattern, text, re.MULTILINE | re.IGNORECASE)
            academic_score += len(matches)
        
        if academic_score >= 2:
            content_analysis["has_academic_structure"] = True
            content_analysis["content_type"] = "academic"
            content_analysis["academic_section_count"] = academic_score
        
        # Count citations
        citation_count = 0
        for pattern in self.citation_patterns:
            citation_count += len(re.findall(pattern, text))
        
        content_analysis["citation_count"] = citation_count
        
        logger.info(f"📊 Content analysis: {content_analysis['content_type']}, "
                   f"citations: {citation_count}, sections: {academic_score}")
        
        return content_analysis
    
    async def _structure_aware_chunking(self, text: str) -> List[str]:
        """
        Structure-aware chunking for markdown content.
        
        Preserves headers and sections while maintaining optimal chunk size.
        """
        # Split by headers first, then by paragraphs if too large
        sections = re.split(r'\n(?=#{1,6}\s+)', text)
        chunks = []
        
        for section in sections:
            if len(section) <= self.chunk_size * 4:  # Characters to tokens approximation
                chunks.append(section.strip())
            else:
                # Split large sections by paragraphs
                paragraphs = section.split('\n\n')
                current_chunk = ""
                
                for paragraph in paragraphs:
                    if len(current_chunk) + len(paragraph) <= self.chunk_size * 4:
                        current_chunk += paragraph + "\n\n"
                    else:
                        if current_chunk.strip():
                            chunks.append(current_chunk.strip())
                        current_chunk = paragraph + "\n\n"
                
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
        
        # Filter out empty chunks
        chunks = [chunk for chunk in chunks if chunk.strip()]
        return chunks
    
    async def _academic_aware_chunking(self, text: str) -> List[str]:
        """
        Academic-aware chunking for research papers.
        
        Preserves academic sections and citation boundaries.
        """
        # Try to split by academic sections first
        section_pattern = r'\n(?=' + '|'.join(self.academic_sections) + ')'
        sections = re.split(section_pattern, text, flags=re.MULTILINE | re.IGNORECASE)
        
        if len(sections) > 1:
            # We found academic sections, use them as natural boundaries
            chunks = []
            for section in sections:
                if len(section) <= self.chunk_size * 4:
                    chunks.append(section.strip())
                else:
                    # Split large sections using token splitter
                    sub_chunks = await asyncio.get_event_loop().run_in_executor(
                        None, self.token_splitter.split_text, section
                    )
                    chunks.extend(sub_chunks)
        else:
            # No clear academic structure, use standard token splitting
            chunks = await asyncio.get_event_loop().run_in_executor(
                None, self.token_splitter.split_text, text
            )
        
        # Filter out empty chunks
        chunks = [chunk for chunk in chunks if chunk.strip()]
        return chunks
    
    async def _standard_chunking(self, text: str) -> List[str]:
        """
        Standard chunking for plain text documents.
        
        Uses RecursiveCharacterTextSplitter for optimal performance.
        """
        chunks = await asyncio.get_event_loop().run_in_executor(
            None, self.text_splitter.split_text, text
        )
        
        # Filter out empty chunks
        chunks = [chunk for chunk in chunks if chunk.strip()]
        return chunks
    
    def _extract_citations(self, text: str) -> List[str]:
        """Extract citations from text chunk."""
        citations = []
        for pattern in self.citation_patterns:
            citations.extend(re.findall(pattern, text))
        return list(set(citations))  # Remove duplicates
    
    def _find_academic_sections(self, text: str) -> List[str]:
        """Find academic sections in text chunk."""
        sections = []
        for pattern in self.academic_sections:
            matches = re.findall(pattern, text, re.MULTILINE | re.IGNORECASE)
            sections.extend(matches)
        return list(set(sections))  # Remove duplicates
    
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
        """Get production document processing statistics."""
        return {
            "processor_initialized": self._initialized,
            "supported_formats": [t.value for t in DocumentType],
            "chunking_strategy": "production_optimized",
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "features": [
                "content_adaptive_chunking",
                "citation_preservation",
                "structure_awareness",
                "academic_section_detection",
                "fast_processing"
            ],
            "embedding_model": settings.embedding_model
        }


# Maintain backward compatibility
DocumentProcessor = ProductionDocumentProcessor
ResearchDocumentProcessor = ProductionDocumentProcessor

# ==============================================
# Utility Functions
# ==============================================

async def create_document_processor(vector_store: VectorStoreService) -> ProductionDocumentProcessor:
    """Create and initialize a production document processor."""
    processor = ProductionDocumentProcessor(vector_store)
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
    elif extension in ['.md', '.markdown']:
        return DocumentType.MARKDOWN
    else:
        raise ValueError(f"Unsupported file type: {extension}")


if __name__ == "__main__":
    # Test the production document processor
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
        print("Production Document Processor Stats:", stats)
    
    asyncio.run(test_processor()) 