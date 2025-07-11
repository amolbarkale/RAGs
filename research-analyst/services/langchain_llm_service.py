"""
LangChain LLM service for response generation using Google Gemini.

This service handles all LLM operations including:
- Response generation with context
- Prompt template management
- Streaming responses
- Error handling and fallbacks
"""

import logging
import asyncio
from typing import List, Dict, Any, Optional, AsyncGenerator
from datetime import datetime

# LangChain imports
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser

# Google Gemini imports
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

# Core imports
from core.config import settings
from core.models import SearchResult, ResponseStrategy, QueryType

# Set up logging
logger = logging.getLogger(__name__)

class LangChainLLMService:
    """
    LangChain-based LLM service using Google Gemini.
    
    Handles intelligent response generation with context from search results.
    """
    
    def __init__(self):
        self.llm = None
        self.backup_llm = None
        self.prompt_templates = {}
        self._initialized = False
        
    async def initialize(self):
        """Initialize the LLM service with Gemini models."""
        if self._initialized:
            return
            
        logger.info("🤖 Initializing LangChain LLM service with Gemini...")
        
        if not GEMINI_AVAILABLE:
            raise ImportError("Google Gemini integration not available. Please install langchain-google-genai")
        
        if not settings.gemini_api_key:
            raise ValueError("GEMINI_API_KEY is required for LLM service")
        
        try:
            # Initialize primary Gemini model
            self.llm = ChatGoogleGenerativeAI(
                model=settings.llm_model,
                google_api_key=settings.gemini_api_key,
                temperature=0.1,
                max_tokens=2048,
                timeout=30
            )
            
            # Initialize backup model
            self.backup_llm = ChatGoogleGenerativeAI(
                model=settings.backup_llm_model,
                google_api_key=settings.gemini_api_key,
                temperature=0.1,
                max_tokens=2048,
                timeout=30
            )
            
            # Initialize prompt templates
            self._init_prompt_templates()
            
            self._initialized = True
            logger.info(f"✅ LLM service initialized with {settings.llm_model}")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize LLM service: {e}")
            raise
    
    def _init_prompt_templates(self):
        """Initialize prompt templates for different response strategies."""
        
        # RAG-only response template
        self.prompt_templates["rag_only"] = ChatPromptTemplate.from_messages([
            SystemMessage(content="""You are an expert research assistant. Your task is to provide accurate, comprehensive answers based solely on the provided document sources.

Guidelines:
- Use only the information from the provided documents
- Cite sources using [Source N] notation
- If information is insufficient, clearly state what you cannot answer
- Maintain academic rigor and precision
- Provide specific details and examples when available"""),
            HumanMessage(content="""Based on the following document sources, please answer the question:

Question: {query}

Sources:
{context}

Please provide a comprehensive answer based solely on these sources.""")
        ])
        
        # Web-only response template
        self.prompt_templates["web_only"] = ChatPromptTemplate.from_messages([
            SystemMessage(content="""You are an expert research assistant with access to current web information. Provide accurate, up-to-date answers based on the provided web sources.

Guidelines:
- Use only the information from the provided web sources
- Cite sources using [Source N] notation
- Focus on current, relevant information
- Distinguish between facts and opinions
- Highlight any conflicting information from different sources"""),
            HumanMessage(content="""Based on the following web sources, please answer the question:

Question: {query}

Sources:
{context}

Please provide a comprehensive answer based on these current sources.""")
        ])
        
        # Mixed response template
        self.prompt_templates["mixed"] = ChatPromptTemplate.from_messages([
            SystemMessage(content="""You are an expert research assistant with access to both document archives and current web information. Provide comprehensive answers that synthesize information from both sources.

Guidelines:
- Clearly distinguish between document sources and web sources
- Use [Doc N] for document sources and [Web N] for web sources
- Highlight any differences between document and web information
- Provide both historical context and current information when relevant
- Synthesize information to provide a complete picture"""),
            HumanMessage(content="""Based on the following sources, please answer the question:

Question: {query}

Sources:
{context}

Please provide a comprehensive answer that synthesizes information from all sources.""")
        ])
        
        # Default response template
        self.prompt_templates["default"] = ChatPromptTemplate.from_messages([
            SystemMessage(content="""You are a helpful research assistant. Provide accurate, well-structured answers based on the provided sources.

Guidelines:
- Use only the information from the provided sources
- Cite sources appropriately
- Be clear and concise
- If information is insufficient, state what you cannot answer"""),
            HumanMessage(content="""Question: {query}

Sources:
{context}

Please provide an answer based on these sources.""")
        ])
    
    async def generate_response(
        self,
        query: str,
        search_results: List[SearchResult],
        response_strategy: ResponseStrategy,
        query_type: QueryType = QueryType.FACTUAL,
        max_context_length: int = 4000
    ) -> str:
        """
        Generate a response using LangChain with Gemini.
        
        Args:
            query: The user's question
            search_results: List of search results for context
            response_strategy: Strategy for response generation
            query_type: Type of query (factual, analytical, etc.)
            max_context_length: Maximum length of context to include
            
        Returns:
            Generated response string
        """
        if not self._initialized:
            await self.initialize()
            
        logger.info(f"🤖 Generating response for query: '{query[:50]}...'")
        
        if not search_results:
            return "I couldn't find any relevant information to answer your question. Please try rephrasing your query or provide more specific details."
        
        try:
            # Prepare context from search results
            context = self._prepare_context(search_results, max_context_length)
            
            # Select appropriate prompt template
            template_key = self._get_template_key(response_strategy)
            prompt_template = self.prompt_templates.get(template_key, self.prompt_templates["default"])
            
            # Create the prompt
            prompt = prompt_template.format(query=query, context=context)
            
            # Generate response using primary model
            try:
                response = await self._generate_with_model(self.llm, prompt)
            except Exception as e:
                logger.warning(f"Primary model failed: {e}. Trying backup model...")
                response = await self._generate_with_model(self.backup_llm, prompt)
            
            logger.info(f"✅ Response generated successfully ({len(response)} chars)")
            return response
            
        except Exception as e:
            logger.error(f"❌ Response generation failed: {e}")
            return f"I apologize, but I encountered an error while generating a response. Please try again or contact support if the issue persists."
    
    async def _generate_with_model(self, model, prompt: str) -> str:
        """Generate response with a specific model."""
        try:
            # Use LangChain's async invoke
            response = await model.ainvoke(prompt)
            return response.content if hasattr(response, 'content') else str(response)
        except Exception as e:
            logger.error(f"Model generation failed: {e}")
            raise
    
    def _prepare_context(self, search_results: List[SearchResult], max_length: int) -> str:
        """Prepare context string from search results."""
        context_parts = []
        current_length = 0
        
        for i, result in enumerate(search_results):
            # Format source based on type
            source_prefix = "Doc" if result.source_type.name == "DOCUMENT" else "Web"
            
            # Create context entry
            context_entry = f"[{source_prefix} {i+1}] {result.title}\n{result.content}"
            
            # Add source information if available
            if result.source_name:
                context_entry += f"\n(Source: {result.source_name})"
            
            # Check length limit
            if current_length + len(context_entry) > max_length:
                break
                
            context_parts.append(context_entry)
            current_length += len(context_entry)
        
        return "\n\n".join(context_parts)
    
    def _get_template_key(self, response_strategy: ResponseStrategy) -> str:
        """Get the appropriate template key for the response strategy."""
        if response_strategy == ResponseStrategy.RAG_ONLY:
            return "rag_only"
        elif response_strategy == ResponseStrategy.WEB_ONLY:
            return "web_only"
        elif response_strategy == ResponseStrategy.MIXED:
            return "mixed"
        else:
            return "default"
    
    async def generate_streaming_response(
        self,
        query: str,
        search_results: List[SearchResult],
        response_strategy: ResponseStrategy
    ) -> AsyncGenerator[str, None]:
        """
        Generate streaming response (for real-time UI updates).
        
        Note: This is a placeholder for streaming implementation.
        Full streaming requires additional LangChain configuration.
        """
        if not self._initialized:
            await self.initialize()
        
        # For now, generate full response and yield in chunks
        # TODO: Implement true streaming with LangChain streaming callbacks
        response = await self.generate_response(query, search_results, response_strategy)
        
        # Yield response in chunks for streaming effect
        chunk_size = 50
        for i in range(0, len(response), chunk_size):
            yield response[i:i+chunk_size]
            await asyncio.sleep(0.1)  # Small delay for streaming effect

# Factory function
async def create_llm_service() -> LangChainLLMService:
    """Create and initialize the LLM service."""
    service = LangChainLLMService()
    await service.initialize()
    return service

# Global instance for import
llm_service = None

async def get_llm_service() -> LangChainLLMService:
    """Get the global LLM service instance."""
    global llm_service
    if llm_service is None:
        llm_service = await create_llm_service()
    return llm_service 