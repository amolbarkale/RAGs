"""
Tavily Web Search Service with LangChain Integration.

This service provides intelligent web search capabilities using Tavily API
with proper content extraction, quality scoring, and credibility assessment.
"""

import logging
import asyncio
import aiohttp
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import hashlib
import re
from urllib.parse import urlparse

# LangChain imports
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.documents import Document as LangChainDocument

# Core imports
from core.config import settings
from core.models import SearchResult, SourceType, QueryType

# Set up logging
logger = logging.getLogger(__name__)

class TavilySearchService:
    """
    Production-ready Tavily web search service with LangChain integration.
    
    Features:
    - Async web search with rate limiting
    - Content quality scoring
    - Source credibility assessment
    - Intelligent caching
    - Error handling and fallbacks
    """
    
    def __init__(self):
        self.tavily_tool = None
        self.session = None
        self._initialized = False
        self._cache = {}  # Simple in-memory cache
        self._cache_ttl = 3600  # 1 hour TTL
        
        # Quality scoring patterns
        self.high_quality_domains = {
            '.edu', '.gov', '.org', '.mil',
            'wikipedia.org', 'britannica.com', 'nature.com',
            'sciencedirect.com', 'springer.com', 'ieee.org',
            'acm.org', 'arxiv.org', 'pubmed.ncbi.nlm.nih.gov'
        }
        
        self.low_quality_patterns = [
            r'click\s*here', r'buy\s*now', r'limited\s*time',
            r'advertisement', r'sponsored', r'affiliate',
            r'get\s*rich', r'miracle', r'guaranteed'
        ]
        
    async def initialize(self):
        """Initialize the Tavily search service."""
        if self._initialized:
            return
            
        logger.info("🌐 Initializing Tavily search service...")
        
        if not settings.tavily_api_key:
            raise ValueError("TAVILY_API_KEY is required for web search")
        
        try:
            # Initialize LangChain Tavily tool
            self.tavily_tool = TavilySearchResults(
                api_key=settings.tavily_api_key,
                max_results=settings.max_web_results,
                search_depth="advanced",  # Get more detailed results
                include_answer=True,      # Get AI-generated answers
                include_raw_content=True, # Get full content
                include_images=False      # Skip images for now
            )
            
            # Initialize async HTTP session
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=30),
                headers={
                    'User-Agent': 'Research-Assistant/1.0 (Educational)'
                }
            )
            
            self._initialized = True
            logger.info("✅ Tavily search service initialized!")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Tavily service: {e}")
            raise
    
    async def search_web(
        self,
        query: str,
        max_results: int = 10,
        query_type: QueryType = QueryType.FACTUAL,
        include_recent: bool = True
    ) -> List[SearchResult]:
        """
        Perform intelligent web search using Tavily API.
        
        Args:
            query: Search query
            max_results: Maximum number of results to return
            query_type: Type of query for optimization
            include_recent: Whether to prioritize recent results
            
        Returns:
            List of SearchResult objects with web content
        """
        if not self._initialized:
            await self.initialize()
            
        logger.info(f"🔍 Searching web for: '{query[:50]}...'")
        
        # Check cache first
        cache_key = self._get_cache_key(query, max_results, query_type)
        cached_results = self._get_cached_results(cache_key)
        if cached_results:
            logger.info(f"📄 Retrieved {len(cached_results)} results from cache")
            return cached_results
        
        try:
            # Enhance query based on type
            enhanced_query = self._enhance_query(query, query_type, include_recent)
            
            # Perform search using LangChain Tavily tool
            search_results = await self._perform_tavily_search(enhanced_query, max_results)
            
            # Process and score results
            processed_results = await self._process_search_results(
                search_results, query, query_type
            )
            
            # Cache results
            self._cache_results(cache_key, processed_results)
            
            logger.info(f"✅ Found {len(processed_results)} web results")
            return processed_results
            
        except Exception as e:
            logger.error(f"❌ Web search failed: {e}")
            return []
    
    async def _perform_tavily_search(
        self,
        query: str,
        max_results: int
    ) -> List[Dict[str, Any]]:
        """Perform the actual Tavily search."""
        try:
            # Ensure tavily_tool is initialized
            if not self.tavily_tool:
                raise ValueError("Tavily tool not initialized")
            
            # Update max results for this search
            self.tavily_tool.max_results = max_results
            
            # Run search in executor to avoid blocking
            results = await asyncio.get_event_loop().run_in_executor(
                None, self.tavily_tool.run, query
            )
            
            # Handle different result formats
            if isinstance(results, list):
                return results
            elif isinstance(results, dict):
                return results.get('results', [])
            else:
                logger.warning(f"Unexpected Tavily result format: {type(results)}")
                return []
                
        except Exception as e:
            logger.error(f"Tavily search execution failed: {e}")
            return []
    
    async def _process_search_results(
        self,
        raw_results: List[Dict[str, Any]],
        original_query: str,
        query_type: QueryType
    ) -> List[SearchResult]:
        """Process raw Tavily results into SearchResult objects."""
        processed_results = []
        
        for i, result in enumerate(raw_results):
            try:
                # Extract basic information
                title = result.get('title', 'Unknown Title')
                url = result.get('url', '')
                content = result.get('content', '') or result.get('snippet', '')
                
                # Skip if no content
                if not content.strip():
                    continue
                
                # Calculate quality and credibility scores
                quality_score = self._calculate_quality_score(content, url)
                credibility_score = self._calculate_credibility_score(url)
                freshness_score = self._calculate_web_freshness_score(result)
                
                # Calculate overall relevance score
                relevance_score = self._calculate_relevance_score(
                    content, original_query, quality_score, credibility_score, freshness_score
                )
                
                # Create SearchResult object
                search_result = SearchResult(
                    title=title,
                    content=content,
                    source_type=SourceType.WEB,
                    relevance_score=relevance_score,
                    dense_score=0.0,  # Web results don't have dense embeddings
                    sparse_score=relevance_score,  # Use as sparse score
                    freshness_score=freshness_score,
                    source_name=self._extract_domain_name(url),
                    url=url,
                    source_id=f"web_{i}",
                    chunk_id=f"web_{i}",
                    page_number=i + 1,
                    credibility_score=credibility_score  # This field exists in SearchResult
                )
                
                processed_results.append(search_result)
                
            except Exception as e:
                logger.warning(f"Failed to process search result {i}: {e}")
                continue
        
        # Sort by relevance score
        processed_results.sort(key=lambda x: x.relevance_score, reverse=True)
        
        return processed_results
    
    def _enhance_query(
        self,
        query: str,
        query_type: QueryType,
        include_recent: bool
    ) -> str:
        """Enhance query based on type and preferences."""
        enhanced_query = query
        
        # Add temporal constraints for recent queries
        if query_type == QueryType.RECENT or include_recent:
            current_year = datetime.now().year
            enhanced_query += f" {current_year} {current_year-1}"
        
        # Add quality filters for factual queries
        if query_type == QueryType.FACTUAL:
            enhanced_query += " site:edu OR site:gov OR site:org OR wikipedia"
        
        # Add analysis keywords for analytical queries
        if query_type == QueryType.ANALYTICAL:
            enhanced_query += " analysis research study report"
        
        return enhanced_query
    
    def _calculate_quality_score(self, content: str, url: str) -> float:
        """Calculate content quality score."""
        score = 0.5  # Base score
        
        # Length score (longer content generally better)
        if len(content) > 500:
            score += 0.2
        elif len(content) > 200:
            score += 0.1
        
        # Domain quality bonus
        for domain in self.high_quality_domains:
            if domain in url.lower():
                score += 0.3
                break
        
        # Penalty for low-quality patterns
        content_lower = content.lower()
        for pattern in self.low_quality_patterns:
            if re.search(pattern, content_lower):
                score -= 0.1
                break
        
        # Bonus for structured content
        if any(marker in content for marker in ['1.', '2.', '•', '-', 'Abstract:', 'Summary:']):
            score += 0.1
        
        # Penalty for excessive capitalization
        if len(re.findall(r'[A-Z]{3,}', content)) > 3:
            score -= 0.1
        
        return max(0.0, min(1.0, score))
    
    def _calculate_credibility_score(self, url: str) -> float:
        """Calculate source credibility score."""
        score = 0.5  # Base score
        
        domain = urlparse(url).netloc.lower()
        
        # High credibility domains
        if any(hq_domain in domain for hq_domain in self.high_quality_domains):
            score += 0.4
        
        # Medium credibility domains
        elif any(med_domain in domain for med_domain in [
            'reuters.com', 'bbc.com', 'nytimes.com', 'wsj.com',
            'economist.com', 'forbes.com', 'harvard.edu', 'mit.edu'
        ]):
            score += 0.3
        
        # HTTPS bonus
        if url.startswith('https://'):
            score += 0.1
        
        # Penalty for suspicious patterns
        if any(suspicious in domain for suspicious in [
            'ads', 'affiliate', 'sponsored', 'click', 'buy'
        ]):
            score -= 0.2
        
        return max(0.0, min(1.0, score))
    
    def _calculate_web_freshness_score(self, result: Dict[str, Any]) -> float:
        """Calculate freshness score for web results."""
        # Tavily results are generally recent, so give a good freshness score
        published_date = result.get('published_date')
        
        if published_date:
            try:
                # Parse date and calculate age
                pub_date = datetime.fromisoformat(published_date.replace('Z', '+00:00'))
                age_days = (datetime.now() - pub_date.replace(tzinfo=None)).days
                
                # Fresher content gets higher scores
                if age_days < 7:
                    return 1.0
                elif age_days < 30:
                    return 0.8
                elif age_days < 90:
                    return 0.6
                else:
                    return 0.4
            except:
                pass
        
        # Default to good freshness for web results
        return 0.7
    
    def _calculate_relevance_score(
        self,
        content: str,
        query: str,
        quality_score: float,
        credibility_score: float,
        freshness_score: float
    ) -> float:
        """Calculate overall relevance score."""
        # Simple keyword matching for relevance
        query_words = query.lower().split()
        content_lower = content.lower()
        
        # Count keyword matches
        matches = sum(1 for word in query_words if word in content_lower)
        keyword_score = min(1.0, matches / len(query_words)) if query_words else 0.0
        
        # Weighted combination
        relevance_score = (
            keyword_score * 0.4 +
            quality_score * 0.3 +
            credibility_score * 0.2 +
            freshness_score * 0.1
        )
        
        return max(0.0, min(1.0, relevance_score))
    
    def _extract_domain_name(self, url: str) -> str:
        """Extract clean domain name from URL."""
        try:
            domain = urlparse(url).netloc
            # Remove www. prefix
            if domain.startswith('www.'):
                domain = domain[4:]
            return domain
        except:
            return "Unknown Domain"
    
    def _get_cache_key(self, query: str, max_results: int, query_type: QueryType) -> str:
        """Generate cache key for search results."""
        key_data = f"{query}_{max_results}_{query_type.value}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _get_cached_results(self, cache_key: str) -> Optional[List[SearchResult]]:
        """Get cached results if still valid."""
        if cache_key in self._cache:
            cached_data = self._cache[cache_key]
            if datetime.now() - cached_data['timestamp'] < timedelta(seconds=self._cache_ttl):
                return cached_data['results']
            else:
                # Remove expired cache
                del self._cache[cache_key]
        return None
    
    def _cache_results(self, cache_key: str, results: List[SearchResult]):
        """Cache search results."""
        self._cache[cache_key] = {
            'results': results,
            'timestamp': datetime.now()
        }
        
        # Simple cache size management
        if len(self._cache) > 100:
            # Remove oldest entries
            oldest_key = min(self._cache.keys(), 
                           key=lambda k: self._cache[k]['timestamp'])
            del self._cache[oldest_key]
    
    async def cleanup(self):
        """Cleanup resources."""
        if self.session and not self.session.closed:
            await self.session.close()
            self.session = None
        
        self._cache.clear()
        logger.info("🧹 Tavily search service cleaned up")

# Factory function
async def create_tavily_service() -> TavilySearchService:
    """Create and initialize Tavily search service."""
    service = TavilySearchService()
    await service.initialize()
    return service

# Global instance
_tavily_service = None

async def get_tavily_service() -> TavilySearchService:
    """Get global Tavily service instance."""
    global _tavily_service
    if _tavily_service is None:
        _tavily_service = await create_tavily_service()
    return _tavily_service 