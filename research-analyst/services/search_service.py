"""
Production-ready search service for the Research Assistant RAG system.

This service orchestrates all search operations including:
- Vector similarity search
- Query processing and classification
- Hybrid search (document + web)
- Result ranking and scoring
- Real-time web search integration
"""

import asyncio
import logging
import time
from typing import List, Dict, Any, Optional, Tuple, Union
from datetime import datetime
from enum import Enum
import re
import statistics

# Core imports
from core.config import settings
from core.models import (
    QueryRequest, QueryResponse, QueryType, ResponseStrategy,
    SearchResult, HybridSearchResult, SourceType,
    SearchRequest
)

# Services
from services.vector_store import VectorStoreService
from services.models import DocumentDBModel, DocumentChunkDBModel
from services.langchain_llm_service import get_llm_service
from services.tavily_search_service import get_tavily_service

# Database
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func

# Set up logging
logger = logging.getLogger(__name__)

class QueryClassifier:
    """Classifies queries by type and intent for optimal processing."""
    
    @staticmethod
    def classify_query(query: str) -> QueryType:
        """
        Classify query type based on content and patterns.
        
        For beginners: This helps us route queries to the best search strategy.
        """
        query_lower = query.lower()
        
        # Recent/temporal queries
        temporal_patterns = [
            r'\b(recent|latest|current|today|yesterday|this week|this month|2024|2023)\b',
            r'\b(now|currently|at the moment|up to date)\b',
            r'\b(breaking|news|just|happened|announced)\b'
        ]
        
        if any(re.search(pattern, query_lower) for pattern in temporal_patterns):
            return QueryType.RECENT
        
        # Analytical/complex queries
        analytical_patterns = [
            r'\b(analyze|compare|evaluate|assess|explain|how does|why does)\b',
            r'\b(relationship|impact|effect|consequence|implication)\b',
            r'\b(strategy|approach|methodology|framework)\b'
        ]
        
        if any(re.search(pattern, query_lower) for pattern in analytical_patterns):
            return QueryType.ANALYTICAL
        
        # Factual queries (default)
        return QueryType.FACTUAL

class SearchService:
    """
    Production-ready search service with hybrid capabilities.
    
    Orchestrates vector search, query processing, and result ranking.
    """
    
    def __init__(self, vector_store: VectorStoreService, db_session: Optional[AsyncSession] = None):
        self.vector_store = vector_store
        self.db_session = db_session
        self.query_classifier = QueryClassifier()
        self._initialized = False
        
    async def initialize(self):
        """Initialize the search service."""
        if self._initialized:
            return
            
        logger.info("🔍 Initializing search service...")
        
        # Ensure vector store is initialized
        await self.vector_store.initialize()
        
        self._initialized = True
        logger.info("✅ Search service initialized!")
    
    async def process_query(self, query_request: QueryRequest) -> QueryResponse:
        """
        Process a complete query with hybrid search and response generation.
        
        This is the main entry point for query processing.
        """
        if not self._initialized:
            await self.initialize()
            
        start_time = time.time()
        logger.info(f"🔍 Processing query: '{query_request.query[:100]}...'")
        
        try:
            # Step 1: Classify query
            query_type = self.query_classifier.classify_query(query_request.query)
            logger.info(f"📊 Query classified as: {query_type}")
            
            # Step 2: Perform hybrid search
            search_results = await self._perform_hybrid_search(
                query_request.query,
                query_type=query_type,
                max_results=query_request.max_results or 20
            )
            
            # Step 3: Calculate confidence and determine response strategy
            confidence_score = self._calculate_confidence(search_results)
            max_similarity = max(
                (result.relevance_score for result in search_results.combined_results),
                default=0.0
            )
            response_strategy = QueryResponse.determine_response_strategy(max_similarity)
            
            # Step 4: Generate response
            answer = await self._generate_response(
                query_request.query,
                search_results.combined_results,
                response_strategy
            )
            
            # Step 5: Extract citations
            citations = self._extract_citations(search_results.combined_results)
            
            processing_time = time.time() - start_time
            
            # Build response
            response = QueryResponse(
                query=query_request.query,
                answer=answer,
                sources=search_results.combined_results,
                confidence_score=confidence_score,
                processing_time=processing_time,
                response_strategy=response_strategy,
                max_similarity_score=max_similarity,
                query_type=query_type,
                result_count=len(search_results.combined_results),
                citations=citations
            )
            
            logger.info(f"✅ Query processed in {processing_time:.2f}s with {len(search_results.combined_results)} results")
            return response
            
        except Exception as e:
            logger.error(f"❌ Query processing failed: {e}")
            raise
    
    async def hybrid_search(self, search_request: SearchRequest) -> List[HybridSearchResult]:
        """
        Perform hybrid search with document and web sources.
        
        Returns structured search results with separate document and web results.
        """
        if not self._initialized:
            await self.initialize()
            
        start_time = time.time()
        logger.info(f"🔍 Performing hybrid search: '{search_request.query[:100]}...'")
        
        try:
            # Classify query for optimal search strategy
            query_type = self.query_classifier.classify_query(search_request.query)
            
            # Perform search based on request parameters
            search_results = await self._perform_hybrid_search(
                search_request.query,
                query_type=query_type,
                max_results=search_request.limit,
                include_documents=search_request.include_documents,
                include_web=search_request.include_web,
                min_relevance_score=search_request.min_relevance_score
            )
            
            search_time = time.time() - start_time
            
            # Determine response strategy
            max_document_similarity = max(
                (result.relevance_score for result in search_results.document_results),
                default=0.0
            )
            response_strategy = QueryResponse.determine_response_strategy(max_document_similarity)
            
            # Build hybrid search result
            hybrid_result = HybridSearchResult(
                document_results=search_results.document_results,
                web_results=search_results.web_results,
                combined_results=search_results.combined_results,
                total_results=search_results.total_results,
                search_time=search_time,
                response_strategy=response_strategy,
                max_document_similarity=max_document_similarity
            )
            
            logger.info(f"✅ Hybrid search completed in {search_time:.2f}s")
            return [hybrid_result]
            
        except Exception as e:
            logger.error(f"❌ Hybrid search failed: {e}")
            raise
    
    async def _perform_hybrid_search(
        self,
        query: str,
        query_type: QueryType = QueryType.FACTUAL,
        max_results: int = 20,
        include_documents: bool = True,
        include_web: bool = True,
        min_relevance_score: float = 0.0
    ) -> HybridSearchResult:
        """
        Internal method to perform hybrid search across document and web sources.
        """
        document_results = []
        web_results = []
        
        # Search documents if requested
        if include_documents:
            document_results = await self._search_documents(
                query, 
                max_results=max_results // 2 if include_web else max_results,
                min_score=min_relevance_score
            )
        
        # Search web if requested (placeholder for now)
        if include_web:
            web_results = await self._search_web(
                query,
                max_results=max_results // 2 if include_documents else max_results,
                query_type=query_type
            )
        
        # Combine and rank results
        combined_results = await self._combine_and_rank_results(
            document_results, 
            web_results, 
            query,
            max_results=max_results,
            min_relevance_score=min_relevance_score
        )
        
        return HybridSearchResult(
            document_results=document_results,
            web_results=web_results,
            combined_results=combined_results,
            total_results=len(combined_results),
            search_time=0.0,  # Will be set by caller
            response_strategy=ResponseStrategy.MIXED,
            max_document_similarity=max(
                (result.relevance_score for result in document_results),
                default=0.0
            )
        )
    
    async def _search_documents(
        self,
        query: str,
        max_results: int = 10,
        min_score: float = 0.0
    ) -> List[SearchResult]:
        """
        Search through stored documents using vector similarity.
        """
        try:
            # Perform vector similarity search
            search_results = await self.vector_store.similarity_search(
                query=query,
                k=max_results,
                score_threshold=min_score,
                collection_name="document_chunks"
            )
            
            # Convert to SearchResult objects
            document_results = []
            for result in search_results:
                metadata = result.get("metadata", {})
                
                # Get document title from metadata or generate from content
                title = metadata.get("title", "")
                if not title:
                    # Generate title from first 100 characters
                    title = result["text"][:100].strip() + "..."
                
                search_result = SearchResult(
                    title=title,
                    content=result["text"],
                    source_type=SourceType.DOCUMENT,
                    relevance_score=result["score"],
                    dense_score=result["score"],
                    sparse_score=0.0,  # TODO: Implement sparse scoring
                    freshness_score=self._calculate_freshness_score(metadata),
                    source_name=metadata.get("source_name", "Unknown Document"),
                    url=metadata.get("url"),
                    source_id=metadata.get("document_id"),
                    chunk_id=metadata.get("chunk_id"),
                    page_number=metadata.get("page_number")
                )
                document_results.append(search_result)
            
            logger.info(f"📄 Found {len(document_results)} document results")
            return document_results
            
        except Exception as e:
            logger.error(f"❌ Document search failed: {e}")
            return []
    
    async def _search_web(
        self,
        query: str,
        max_results: int = 10,
        query_type: QueryType = QueryType.FACTUAL
    ) -> List[SearchResult]:
        """
        Search web sources using Tavily API integration.
        """
        try:
            # Get Tavily service
            tavily_service = await get_tavily_service()
            
            # Determine if we should include recent results
            include_recent = query_type == QueryType.RECENT
            
            # Perform web search
            web_results = await tavily_service.search_web(
                query=query,
                max_results=max_results,
                query_type=query_type,
                include_recent=include_recent
            )
            
            logger.info(f"🌐 Found {len(web_results)} web results for: '{query[:50]}...'")
            return web_results
            
        except Exception as e:
            logger.error(f"❌ Web search failed: {e}")
            return []
    
    async def _combine_and_rank_results(
        self,
        document_results: List[SearchResult],
        web_results: List[SearchResult],
        query: str,
        max_results: int = 20,
        min_relevance_score: float = 0.0
    ) -> List[SearchResult]:
        """
        Combine document and web results with intelligent ranking.
        """
        # Combine all results
        all_results = document_results + web_results
        
        # Filter by minimum relevance score
        filtered_results = [
            result for result in all_results
            if result.relevance_score >= min_relevance_score
        ]
        
        # Sort by relevance score (descending)
        filtered_results.sort(key=lambda x: x.relevance_score, reverse=True)
        
        # Apply diversity injection to avoid echo chambers
        ranked_results = self._apply_diversity_injection(filtered_results)
        
        # Limit results
        final_results = ranked_results[:max_results]
        
        logger.info(f"🔗 Combined {len(document_results)} document + {len(web_results)} web results → {len(final_results)} final results")
        return final_results
    
    def _apply_diversity_injection(self, results: List[SearchResult]) -> List[SearchResult]:
        """
        Apply diversity injection to avoid echo chambers.
        
        Ensures we don't return too many similar results from the same source.
        """
        if not results:
            return results
        
        # Group results by source
        source_groups = {}
        for result in results:
            source = result.source_name
            if source not in source_groups:
                source_groups[source] = []
            source_groups[source].append(result)
        
        # Interleave results from different sources
        diversified_results = []
        max_per_source = max(2, len(results) // max(1, len(source_groups)))
        
        for source, source_results in source_groups.items():
            # Take top results from each source
            diversified_results.extend(source_results[:max_per_source])
        
        # Sort by relevance score again
        diversified_results.sort(key=lambda x: x.relevance_score, reverse=True)
        
        return diversified_results
    
    def _calculate_freshness_score(self, metadata: Dict[str, Any]) -> float:
        """
        Calculate freshness score based on document metadata.
        """
        try:
            # Check if we have a creation date
            created_at = metadata.get("created_at")
            if not created_at:
                return 0.5  # Default neutral freshness
            
            # Parse date and calculate age
            if isinstance(created_at, str):
                created_date = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
            else:
                created_date = created_at
            
            # Calculate age in days
            age_days = (datetime.now() - created_date.replace(tzinfo=None)).days
            
            # Fresher documents get higher scores
            if age_days < 30:
                return 1.0
            elif age_days < 180:
                return 0.8
            elif age_days < 365:
                return 0.6
            else:
                return 0.4
                
        except Exception as e:
            logger.debug(f"Could not calculate freshness score: {e}")
            return 0.5
    
    def _calculate_confidence(self, search_results: HybridSearchResult) -> float:
        """
        Calculate overall confidence score based on search results.
        """
        if not search_results.combined_results:
            return 0.0
        
        # Get relevance scores
        scores = [result.relevance_score for result in search_results.combined_results]
        
        # Calculate confidence based on top scores and distribution
        top_score = max(scores)
        avg_score = statistics.mean(scores)
        
        # Higher confidence if we have high-scoring results
        confidence = min(1.0, (top_score * 0.7) + (avg_score * 0.3))
        
        return confidence
    
    def _extract_citations(self, results: List[SearchResult]) -> List[str]:
        """
        Extract citations from search results.
        """
        citations = []
        for result in results:
            if result.source_name and result.source_name not in citations:
                citations.append(result.source_name)
        
        return citations[:10]  # Limit to top 10 citations
    
    async def _generate_response(
        self,
        query: str,
        search_results: List[SearchResult],
        response_strategy: ResponseStrategy
    ) -> str:
        """
        Generate response using LangChain with Google Gemini.
        """
        if not search_results:
            return "I couldn't find any relevant information to answer your question. Please try rephrasing your query or provide more specific details."
        
        try:
            # Get the LLM service
            llm_service = await get_llm_service()
            
            # Generate response using Gemini
            response = await llm_service.generate_response(
                query=query,
                search_results=search_results,
                response_strategy=response_strategy,
                query_type=QueryType.FACTUAL,  # TODO: Use actual query type from classification
                max_context_length=4000
            )
            
            return response
            
        except Exception as e:
            logger.error(f"❌ LLM response generation failed: {e}")
            
            # Fallback to simple response
            top_result = search_results[0]
            if response_strategy == ResponseStrategy.RAG_ONLY:
                return f"Based on the available documents: {top_result.content[:500]}..."
            elif response_strategy == ResponseStrategy.WEB_ONLY:
                return f"According to recent web sources: {top_result.content[:500]}..."
            else:  # MIXED
                return f"Based on available information: {top_result.content[:500]}..."


# Factory function
async def create_search_service(
    vector_store: VectorStoreService,
    db_session: Optional[AsyncSession] = None
) -> SearchService:
    """
    Factory function to create and initialize a search service.
    """
    service = SearchService(vector_store, db_session)
    await service.initialize()
    return service 