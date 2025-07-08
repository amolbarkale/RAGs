# RAG System with Qdrant and Google Gemini

A **Retrieval-Augmented Generation (RAG)** system that processes PDF documents and enables intelligent question-answering using vector embeddings and similarity search.

## 🏗️ **System Architecture**

```
PDF Document → Document Loader → Text Splitter → Embeddings → Vector Store → Retriever → Query Processing
```

## 📋 **Prerequisites**

- Python 3.8+
- Docker & Docker Compose
- Google Gemini API Key

## 🚀 **Quick Start**

### 1. **Clone and Setup**
```bash
git clone <repository-url>
cd RAGs
pip install -r requirements.txt
```

### 2. **Environment Setup**
Create a `.env` file in the project root:
```env
GEMINI_API_KEY=your_google_gemini_api_key_here
```

### 3. **Start Qdrant Vector Database**
```bash
docker compose -f docker-compose.db.yml up -d
```

**Qdrant URLs:**
- **API:** http://localhost:6333
- **Dashboard:** http://localhost:6333/dashboard#/welcome

### 4. **Run the RAG System**
```bash
python rag_1.py
```

## 📁 **Project Structure**

```
RAGs/
├── rag_1.py                    # Main RAG implementation
├── requirements.txt            # Python dependencies
├── docker-compose.db.yml       # Qdrant database setup
├── MCP.pdf                     # Source document
├── misogi_syllabus.pdf         # Additional document
├── .env                        # Environment variables
└── README.md                   # This file
```

## 🔄 **RAG System Flow**

### **Phase 1: Document Ingestion**

#### **Step 1: Document Loading**
```python
# Load PDF document
loader = PyPDFLoader(file_path="./MCP.pdf")
docs = loader.load()
```

#### **Step 2: Text Chunking**
```python
# Split documents into manageable chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,      # Size of each chunk
    chunk_overlap=200,    # Overlap between chunks
)
split_docs = text_splitter.split_documents(docs)
```

#### **Step 3: Embedding Generation**
```python
# Create embeddings using Google Gemini
embedding_model = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=api_key
)
```

#### **Step 4: Vector Store Creation**
```python
# Store chunks in Qdrant vector database
vector_store = QdrantVectorStore.from_documents(
    documents=split_docs,
    url="http://localhost:6333",
    collection_name="mcp_collection",
    embedding=embedding_model,
)
```

#### **Step 5: Retriever Setup**
```python
# Create retriever from existing collection
retriever = QdrantVectorStore.from_existing_collection(
    url="http://localhost:6333",
    collection_name="mcp_collection",
    embedding=embedding_model,
)
```

#### **Step 6: Query Processing**
```python
# Query the system
query = "What is the purpose of the MCP?"
relevant_chunks = retriever.similarity_search(query)
```

### **Phase 2: Response Generation** *(TODO)*

## 📊 **Configuration Options**

### **Text Splitter Settings**
- `chunk_size`: 1000 characters per chunk
- `chunk_overlap`: 200 characters overlap for context preservation

### **Embedding Model Options**
- `models/embedding-001` (current)
- `models/text-embedding-004`
- `models/gemini-embedding-exp-03-07`

### **Vector Store Settings**
- **URL:** http://localhost:6333
- **Collection:** mcp_collection

## 🛠️ **Usage Examples**

### **Basic Query**
```python
query = "What is the purpose of the MCP?"
results = retriever.similarity_search(query)
print(results)
```

### **Advanced Query with Limit**
```python
query = "Explain the key features"
results = retriever.similarity_search(query, k=5)  # Get top 5 results
```

## 📝 **Current TODOs**

1. **Extract page content and page numbers** from search results
2. **Create system prompt** with context integration
3. **Add LLM integration** for final answer generation
4. **Implement response formatting**
5. **Add error handling and logging**

## 🔧 **Development Notes**

- **First Run:** Uncomment Step 4 code to create the initial vector store
- **Subsequent Runs:** Use Step 5 to connect to existing collection
- **Multiple Documents:** Add more PDFs to the same collection

## 📋 **Dependencies**

```
python-dotenv==1.1.1
langchain-community==0.3.27
langchain-text-splitters==0.3.8
langchain-google-genai==2.1.6
langchain-qdrant==0.2.0
```

## 🐛 **Troubleshooting**

### **Common Issues:**
1. **Qdrant Connection Error:** Ensure Docker container is running
2. **API Key Error:** Check `.env` file and Gemini API key
3. **Collection Not Found:** Run Step 4 first to create the collection

### **Useful Commands:**
```bash
# Check Qdrant status
docker ps

# View Qdrant logs
docker logs <container-name>

# Restart Qdrant
docker compose -f docker-compose.db.yml restart
```

## 🚀 **Next Steps**

1. Complete the LLM integration for response generation
2. Add support for multiple document types
3. Implement conversation memory
4. Add web interface for easier querying
5. Deploy to production environment