# Research Assistant - Hybrid RAG Implementation Plan

## Assignment Requirements

### Learning Objectives
- Hybrid search systems
- Real-time information integration
- Production deployment

### Requirements
- PDF document upload and processing
- Real-time web search integration
- Hybrid retrieval (document + web search)
- Source verification and citation
- Response synthesis from multiple sources

### Technical Implementation
Advanced RAG with web integration:
- Document processing pipeline
- Web search API integration (Serper, Bing, etc.)
- Result ranking and relevance scoring
- Source credibility assessment

### Compare Key Retrieval Methods
- **Dense Retrieval** - Semantic search using embeddings (e.g., sentence-transformers)
- **Sparse Retrieval** - Keyword matching
- **Hybrid Retrieval** - Combine dense + sparse scores using
- **Re-ranking** - Use cross-encoders on top-k results for better precision

### Key Indexing Methods
- Vector Indexes
- Text Indexes

### Optimization Tips
- Pre-compute embeddings
- Use approximate nearest neighbour search
- Implement caching for frequent queries

### Production Features
- Response quality monitoring
- User session management
- Indexing techniques

---

## Current Approach - Step-by-Step Plan: Building the Research Assistant (Hybrid RAG)

### Phase 1: Project Setup & File Handling
- **Set up Python project (3.8+) using** `uv`
  - `uv init research-assistant`
  - `uv add streamlit langchain qdrant-client sentence-transformers serper`
- **Add** `data/` folder to upload PDFs
- **Load and extract** from `.pdf` and `.txt` using:
  - `PyPDFLoader`, `TextLoader`

### Phase 2: Dense Document Indexing
- Use `sentence-transformers` (`all-MiniLM-L6-v2` or `bge-small-en`) for local embeddings
- Use `Qdrant` or `FAISS` for vector storage
- Chunk using recursive strategy for best semantic recall
- Store:
  - `chunk`, `source`, `page`, `timestamp`

### Phase 3: Real-Time Web Search Integration
- Use **Serper.dev** or **Bing API**
- Process query → call API → extract top 5–10 URLs
- Scrape summaries with:
  - `newspaper3k`, `BeautifulSoup`, or `serper` snippets

### Phase 4: Hybrid Retrieval Logic
1. **Get local (dense) vector matches** from Qdrant
2. **Get online (sparse) matches** via web search API
3. **Score each result** using:
   - BM25 or TF-IDF (for sparse)
   - Cosine similarity (for dense)
4. **Merge** results by:
   - Normalizing and averaging scores
   - Deduplicating sources

**Bonus**: Add source credibility scoring using known domain trust scores (e.g., `.gov`, `.edu`, `.org` > `.blog`)

### Phase 5: Re-Ranking Layer (Optional but Powerful)
- Use **cross-encoder** (`cross-encoder/ms-marco`) to re-rank top 10 results
- Input: `(query, passage)` pairs
- Output: Refined top-K for final answer

### Phase 6: Response Generation
- Construct system prompt using top K hybrid results
- Use `gemini-pro` or `gpt-4` with context window awareness
- Final answer includes:
  - Synthesized response
  - Source citations (inline or below)
  - Confidence score (optional)

### Phase 7: Optimization & Monitoring
- Precompute embeddings for PDFs on upload
- Cache previous web searches with Redis
- Use ANN (approximate nearest neighbor) in Qdrant for speed
- Log query performance (latency, relevance)
- Streamlit interface with session states

### Method Comparison Table

| Method Type | Use Case |
|-------------|----------|
| Dense Vector | Semantic match |
| Sparse Text index | Exact term, legal/policy search |
| Hybrid Both | Best of both worlds |
| Re-Ranking Post-filter | Precision in final results |

---

## Strategic Feedback on Your Approach

### Strong Points
- Good separation of concerns with phased approach
- Proper consideration of both dense and sparse retrieval
- Inclusion of re-ranking (critical for production quality)
- Thinking about caching and optimization early

### Critical Issues & Improvements

#### 1. Architecture Design Gaps
- Missing **query classification** - different query types need different retrieval strategies
- No **document pre-processing quality gates** - garbage in, garbage out
- Lacking **result freshness scoring** - web content vs document content temporal relevance

#### 2. Technical Implementation Concerns
- `all-MiniLM-L6-v2` is outdated - use `bge-large-en-v1.5` or `e5-large-v2` for better performance
- Single chunking strategy won't work - need **hierarchical chunking** (sentences → paragraphs → sections)
- Missing **query expansion** before retrieval

#### 3. Production-Ready Gaps
- No **failure handling** for API rate limits
- Missing **content deduplication** across sources
- No **hallucination detection** mechanisms

---

## Optimized Step-by-Step Plan

### Phase 1: Smart Foundation Setup

```
Project Structure:
├── core/
│   ├── indexing/     # Document processing
│   ├── retrieval/    # Hybrid search logic  
│   ├── ranking/      # Re-ranking & scoring
│   └── synthesis/    # Response generation
├── services/         # External API integrations
├── utils/           # Caching, monitoring
└── tests/           # Unit & integration tests
```

**Key Dependencies:**
- `sentence-transformers>=2.2.0` with `bge-large-en-v1.5`
- `qdrant-client` with `fastembed` for speed
- `rank-bm25` for sparse retrieval
- `httpx` for async web requests

### Phase 2: Intelligent Document Processing

```python
# Multi-level chunking strategy
1. Extract metadata (title, author, creation date)
2. Detect document structure (headers, sections)
3. Create chunk hierarchy:
   - Level 1: 128 tokens (precise matching)
   - Level 2: 512 tokens (context understanding)
   - Level 3: 2048 tokens (broad context)
4. Quality filtering (remove headers, footers, noise)
```

### Phase 3: Query Intelligence Layer

```python
# Query classification before retrieval
- Factual queries → Dense + Web search
- Analytical queries → Dense + Cross-references
- Recent events → Web search priority
- Technical queries → Sparse + Dense
```

### Phase 4: Advanced Hybrid Retrieval

```python
# Weighted hybrid scoring
def hybrid_score(query, doc):
    dense_score = cosine_similarity(query_emb, doc_emb)
    sparse_score = bm25_score(query, doc)
    freshness_score = calculate_freshness(doc.timestamp)
    
    return (0.6 * dense_score + 
            0.3 * sparse_score + 
            0.1 * freshness_score)
```

### Phase 5: Production-Grade Web Integration
- **Async batch processing** of web searches
- **Rate limiting with exponential backoff**
- **Content quality scoring** (readability, authority)
- **Duplicate detection** using MinHash

### Phase 6: Advanced Re-ranking

```python
# Two-stage re-ranking
1. Fast re-ranking: Cross-encoder on top 20
2. LLM-based relevance scoring on top 10
3. Diversity injection to avoid echo chambers
```

### Phase 7: Smart Response Synthesis

```python
# Context-aware generation
- Detect conflicting information
- Provide confidence intervals
- Include "I don't know" when appropriate
- Structured citations with credibility scores
```

---

## Key Optimizations for Production

### Performance Optimizations
1. **Embedding Caching**: Redis with TTL for computed embeddings
2. **Approximate Search**: Use HNSW with optimal parameters
3. **Batch Processing**: Group similar queries for efficient processing
4. **Streaming Responses**: Start generating while still retrieving

### Quality Assurance
1. **A/B Testing Framework**: Compare retrieval strategies
2. **Relevance Scoring**: Human-in-the-loop feedback
3. **Hallucination Detection**: Cross-reference answers with sources
4. **Citation Accuracy**: Verify all citations link to actual content

### Monitoring & Observability

```python
# Key metrics to track
- Query latency (p95, p99)
- Retrieval accuracy (NDCG@10)
- Source diversity
- Citation accuracy
- User satisfaction scores
```

---

## Architecture Recommendations

### Use This Tech Stack
- **Vector DB**: Qdrant (better than FAISS for production)
- **Embeddings**: `bge-large-en-v1.5` or `e5-large-v2`
- **Re-ranking**: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- **Web Search**: Serper + Bing (redundancy)
- **Caching**: Redis with intelligent TTL
- **Monitoring**: Prometheus + Grafana

### Deployment Strategy
1. **Microservices**: Separate retrieval, ranking, and synthesis
2. **Async Processing**: Use FastAPI with background tasks
3. **Load Balancing**: Handle multiple concurrent users
4. **Graceful Degradation**: Fallback when services fail

This approach will give you a production-ready system that scales and maintains high quality. Focus on getting the retrieval quality right first, then optimize for speed.