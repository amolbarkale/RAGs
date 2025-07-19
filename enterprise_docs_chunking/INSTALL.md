# Installation Guide - Enterprise Document Chunking RAG System

## 🚀 Quick Start

### 1. Prerequisites
- Python 3.8+ 
- Virtual environment (.venv already created)
- Optional: Docker (for Qdrant server)

### 2. Install Dependencies

```bash
# Activate your virtual environment
# Windows:
.venv\Scripts\activate
# Mac/Linux:
source .venv/bin/activate

# Install all required packages
pip install -r requirements.txt
```

### 3. Test Phase 1 Components

```bash
# Run comprehensive tests
python test_phase1.py
```

Expected output:
- ✅ **Configuration**: Settings load correctly
- ✅ **Document Classification**: Pattern detection working
- ✅ **Embeddings**: HuggingFace models downloading and working
- ⚠️ **Vector Store**: May skip if Qdrant server not running (optional for now)

### 4. Optional: Start Qdrant Server

For full vector store testing:

```bash
# Using Docker (recommended)
docker run -p 6333:6333 qdrant/qdrant

# Or install locally: https://qdrant.tech/documentation/quick-start/
```

## 📦 What Gets Installed

### Core Dependencies
- **LangChain**: Framework orchestration
- **sentence-transformers**: Text embeddings
- **transformers**: HuggingFace models
- **torch**: Deep learning backend
- **langchain-qdrant**: Modern vector store integration

### Document Processing
- **pypdf**: PDF text extraction
- **python-docx**: Word document processing
- **markdown**: Markdown file parsing
- **beautifulsoup4**: HTML/XML processing

### Text Processing
- **nltk**: Natural language toolkit
- **spacy**: Advanced NLP

## 🔧 Troubleshooting

### Common Issues

**1. PyTorch Version Conflicts**
```bash
# If you see torch version errors, try:
pip install torch>=2.2.0 --index-url https://download.pytorch.org/whl/cpu
```

**2. Package Not Found Errors**
```bash
# Ensure you're in the right directory and virtual environment
pip install --upgrade pip
pip install -r requirements.txt
```

**3. Import Errors in Tests**
- This is normal before installation
- Linter errors will resolve after `pip install -r requirements.txt`

**4. Qdrant Connection Errors**
- Vector store tests will gracefully skip if Qdrant isn't running
- This doesn't affect other Phase 1 components

## ✅ Verification

After installation, you should see:

```bash
python test_phase1.py

🚀 STARTING PHASE 1 TESTS
📋 Testing core enterprise document chunking components...

==================================================
TESTING CONFIGURATION
==================================================
✅ Configuration loaded successfully
   Supported document types: 5
     technical_doc: ['pdf', 'md', 'rst', 'txt']
     code_doc: ['py', 'js', 'ts', 'java', 'cpp', 'md']
     ...

==================================================
TESTING DOCUMENT CLASSIFICATION
==================================================
✅ Document classifier created successfully

📄 Document: Technical Documentation
   Classified as: technical_doc
   Confidence: 1.00
   Detected patterns: headers, api_endpoints, code_blocks
   Recommended chunking: semantic
   ...

==================================================
TESTING EMBEDDINGS
==================================================
Initializing embedding model: sentence-transformers/all-MiniLM-L6-v2
Using device: cpu
Model loaded successfully. Embedding dimension: 384
✅ Embedding generator created successfully
   ...

======================================================================
TEST SUMMARY
======================================================================
Configuration            ✅ PASS
Document Classification   ✅ PASS
Embeddings               ✅ PASS
Vector Store             ✅ PASS

Passed: 4/4 tests

🎉 All Phase 1 components are working correctly!
Ready to proceed with Phase 2 (LangChain Integration)
```

## 🎯 Next Steps

Once installation is complete:
1. ✅ **Phase 1 Complete**: Core components working
2. 🚧 **Phase 2 Next**: LangChain integration and chunking strategies
3. 🔄 **Phase 3 After**: Testing and validation

## 💡 Notes

- First run will download HuggingFace models (~100MB)
- Qdrant server is optional for Phase 1 testing
- All components designed to work independently
- Modern LangChain integration for future-proofing 