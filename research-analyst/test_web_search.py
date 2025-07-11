"""
Test script to verify Tavily web search integration.

This script tests the complete web search pipeline:
1. Tavily service initialization
2. Web search functionality
3. Result processing and scoring
4. Integration with the hybrid search system
"""

import asyncio
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

async def test_tavily_service():
    """Test the Tavily service directly."""
    print("🧪 Testing Tavily Service...")
    
    # Check API key
    tavily_api_key = os.getenv('TAVILY_API_KEY')
    if not tavily_api_key:
        print("❌ TAVILY_API_KEY not found in environment")
        return False
    
    print(f"✅ TAVILY_API_KEY found (starts with: {tavily_api_key[:10]}...)")
    
    try:
        # Test Tavily service
        from services.tavily_search_service import create_tavily_service
        from core.models import QueryType
        
        tavily_service = await create_tavily_service()
        print("✅ Tavily service initialized successfully")
        
        # Test web search
        print("🔍 Testing web search...")
        test_query = "artificial intelligence latest developments 2024"
        
        results = await tavily_service.search_web(
            query=test_query,
            max_results=5,
            query_type=QueryType.RECENT,
            include_recent=True
        )
        
        print(f"✅ Web search completed! Found {len(results)} results")
        
        # Display sample results
        if results:
            print("\n📋 Sample Results:")
            for i, result in enumerate(results[:3], 1):
                print(f"\n{i}. {result.title}")
                print(f"   URL: {result.url}")
                print(f"   Source: {result.source_name}")
                print(f"   Relevance: {result.relevance_score:.3f}")
                print(f"   Credibility: {result.credibility_score:.3f}")
                print(f"   Freshness: {result.freshness_score:.3f}")
                print(f"   Content: {result.content[:200]}...")
        
        # Cleanup
        await tavily_service.cleanup()
        
        return True
        
    except Exception as e:
        print(f"❌ Tavily service test failed: {e}")
        return False

async def test_hybrid_search():
    """Test the complete hybrid search pipeline."""
    print("\n🔄 Testing Hybrid Search Pipeline...")
    
    try:
        # Import required services
        from services.search_service import create_search_service
        from services.vector_store import create_vector_store
        from services.database import db_manager
        from core.models import QueryRequest, QueryType
        
        # Initialize database
        await db_manager.initialize()
        
        # Initialize vector store
        qdrant_client = db_manager.get_vector_client()
        vector_store = await create_vector_store(qdrant_client)
        
        # Initialize search service
        search_service = await create_search_service(vector_store)
        
        # Test query
        query_request = QueryRequest(
            query="latest developments in machine learning",
            query_type=QueryType.RECENT,
            max_results=10,
            include_web=True,
            include_documents=True
        )
        
        print(f"🔍 Processing query: '{query_request.query}'")
        response = await search_service.process_query(query_request)
        
        print(f"✅ Query processed successfully!")
        print(f"📊 Results: {response.result_count} total")
        print(f"⏱️  Processing time: {response.processing_time:.2f}s")
        print(f"🎯 Confidence: {response.confidence_score:.3f}")
        print(f"📈 Strategy: {response.response_strategy}")
        print(f"🔍 Query type: {response.query_type}")
        
        # Count document vs web results
        doc_results = len([r for r in response.sources if r.source_type.name == "DOCUMENT"])
        web_results = len([r for r in response.sources if r.source_type.name == "WEB"])
        
        print(f"📄 Document results: {doc_results}")
        print(f"🌐 Web results: {web_results}")
        
        # Show sample response
        if response.answer:
            print(f"\n💬 Sample response: {response.answer[:300]}...")
        
        # Show top sources
        if response.sources:
            print(f"\n📚 Top sources:")
            for i, source in enumerate(response.sources[:3], 1):
                source_type = "🌐" if source.source_type.name == "WEB" else "📄"
                print(f"   {i}. {source_type} {source.title[:60]}...")
                print(f"      Score: {source.relevance_score:.3f} | Source: {source.source_name}")
        
        return True
        
    except Exception as e:
        print(f"❌ Hybrid search test failed: {e}")
        return False

async def test_configuration():
    """Test configuration setup."""
    print("📋 Testing Configuration...")
    
    try:
        from core.config import settings
        
        print(f"✅ Configuration loaded")
        print(f"📊 Max web results: {settings.max_web_results}")
        print(f"📊 Search timeout: {settings.search_timeout}s")
        
        # Test API key validation
        settings.validate_api_keys()
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

async def main():
    """Run all tests."""
    print("🚀 Starting Web Search Integration Tests\n")
    
    # Test configuration
    config_ok = await test_configuration()
    
    if config_ok:
        # Test Tavily service
        tavily_ok = await test_tavily_service()
        
        if tavily_ok:
            # Test hybrid search
            hybrid_ok = await test_hybrid_search()
            
            if hybrid_ok:
                print("\n✅ All tests passed! Your web search integration is working correctly.")
                print("\n🎉 Your Research Assistant now has:")
                print("   📄 Document search from your knowledge base")
                print("   🌐 Real-time web search with Tavily")
                print("   🔗 Intelligent hybrid ranking")
                print("   🤖 Smart response generation with Gemini")
                print("\n🚀 You can now run the full application with:")
                print("   uv run python main.py")
            else:
                print("\n❌ Hybrid search test failed.")
        else:
            print("\n❌ Tavily service test failed.")
            print("Please check your TAVILY_API_KEY and try again.")
    else:
        print("\n❌ Configuration test failed.")

if __name__ == "__main__":
    asyncio.run(main()) 