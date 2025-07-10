# Research Assistant - Hybrid RAG System

Advanced Research Assistant that combines document analysis with real-time web search using **LangChain**, **Google Gemini**, and **Tavily API**.

## 🚀 Features

- **LangChain Integration**: Unified AI workflow orchestration
- **Hybrid RAG**: Combines document and web search
- **Tavily Web Search**: Real-time web search via LangChain
- **FlashRank**: Ultra-fast cross-encoder re-ranking
- **Multi-level Chunking**: 128/512/2048 tokens for different contexts
- **Multi-model Support**: GPT-4, Gemini Pro with fallbacks
- **Qdrant Vector Database**: Production-grade vector storage
- **Redis Caching**: Intelligent caching with TTL
- **FastAPI Backend**: Modern async web framework
- **Streamlit Frontend**: Simple web interface

## 🛠️ Technology Stack

- **Framework**: LangChain for AI orchestration
- **Vector DB**: Qdrant for embeddings storage
- **Web Search**: Tavily API via LangChain
- **Re-ranking**: FlashRank cross-encoder
- **LLM**: Google Gemini Pro
- **Embeddings**: sentence-transformers/bge-large-en-v1.5
- **Caching**: Redis
- **Package Manager**: uv

## 📋 Prerequisites

- Python 3.12+
- [uv package manager](https://docs.astral.sh/uv/)
- Redis server
- Qdrant vector database
- API keys for Google Gemini and Tavily

## 🔧 Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd research-analyst
```

### 2. Install Dependencies

```bash
# Install all dependencies using uv
uv sync
```

### 3. Environment Configuration

Create a `.env` file in the root directory:

```env
# API Keys (Required)
GEMINI_API_KEY=your_gemini_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here

# Database Configuration
QDRANT_URL=http://localhost:6333
REDIS_URL=redis://localhost:6379

# Development Settings
DEBUG=false
DEVELOPMENT=true
HOST=127.0.0.1
PORT=8000
```

### 4. Start Dependencies

#### Option A: Using Docker (Recommended)
```bash
# Start Qdrant
docker run -p 6333:6333 qdrant/qdrant

# Start Redis
docker run -p 6379:6379 redis:latest
```

#### Option B: Local Installation
- Install [Qdrant](https://qdrant.tech/documentation/guides/installation/)
- Install [Redis](https://redis.io/docs/getting-started/installation/)

## 🚀 Usage

### 1. Test Configuration

```bash
uv run python core/config.py
```

### 2. Start the API Server

```bash
uv run python main.py
```

### 3. Access the API

- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health
- **Configuration**: http://localhost:8000/config (dev mode only)

### 4. Start the Frontend (Coming Soon)

```bash
uv run streamlit run frontend/app.py
```

## 🔑 API Keys Setup

### Google Gemini API Key
1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Create a new API key
3. Add to `.env` file: `GEMINI_API_KEY=your_key_here`

### Tavily API Key
1. Go to [Tavily](https://tavily.com/)
2. Sign up and get your API key
3. Add to `.env` file: `TAVILY_API_KEY=your_key_here`

## 📊 Current Status

**Phase 1: Foundation & Core Setup** ✅
- [x] Project initialization with uv
- [x] FastAPI application structure
- [x] LangChain integration setup
- [x] Configuration management
- [x] Basic logging system
- [x] Pydantic models for data validation

**Phase 2: Database Setup** 🔄 (In Progress)
- [ ] Qdrant vector database setup
- [ ] Document storage schema
- [ ] Basic CRUD operations

**Next Steps:**
- Vector database integration
- Document upload system
- Text processing with LangChain
- Web search with Tavily

## 🧪 Testing

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov

# Run specific test file
uv run pytest tests/test_config.py
```

## 🏗️ Architecture

```
research-analyst/
├── core/                   # Core system components
│   ├── config.py          # Configuration management
│   ├── models.py          # Pydantic data models
│   └── database.py        # Database connections
├── services/              # Business logic services
│   ├── langchain_llm_service.py
│   ├── tavily_search_service.py
│   └── embedding_service.py
├── utils/                 # Utility functions
│   ├── logger.py          # Logging configuration
│   └── helpers.py         # Helper functions
├── api/                   # API endpoints
├── frontend/              # Streamlit frontend
├── tests/                 # Test suites
└── main.py               # FastAPI application
```

## 📚 Documentation

- [API Documentation](http://localhost:8000/docs) (when server is running)
- [LangChain Documentation](https://python.langchain.com/docs/get_started/introduction)
- [Tavily API Documentation](https://docs.tavily.com/)
- [FlashRank Documentation](https://github.com/PrithivirajDamodaran/FlashRank)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

## 🔗 Links

- [LangChain](https://python.langchain.com/)
- [Tavily](https://tavily.com/)
- [FlashRank](https://github.com/PrithivirajDamodaran/FlashRank)
- [Qdrant](https://qdrant.tech/)
- [FastAPI](https://fastapi.tiangolo.com/)
