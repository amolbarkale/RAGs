✅ Phase 1: Project Setup & Folder Structure
Initialize project with uv

Set up standard folder structure:

bash
Copy
Edit
hr_assistant/
├── main.py                # Streamlit UI entry point
├── engine/
│   └── loader.py          # Load PDF, TXT files
│   └── chunking.py        # HR-specific chunking logic
├── rag/
│   └── embed_store.py     # Embedding + vector store
├── data/                  # Place to drop HR files (PDFs, TXTs)
├── .env
└── README.md
Add env vars for GEMINI API Key

Configure uv dependencies

Phase 2: Multi-Format Document Loader
Load .pdf and .txt files from /data

Use PyPDFLoader and TextLoader

Combine output into single documents list

Include filename metadata for traceability

Phase 3: HR-Specific Chunking Strategies
Use recursive + markdown-aware chunkers

Add rule-based chunking for:

"Leave Policy", "Benefits", "Conduct", etc.

Return structured Document chunks with metadata

Phase 4: Embedding & Vector Storage
Use Gemini embeddings

Store in local Qdrant (or FAISS fallback)

Include filename, category, etc. as metadata

Phase 5: Contextual Retrieval Pipeline
Accept user query

Use LangChain retriever to fetch top K relevant chunks

Metadata-aware filtering (e.g., only leave-related docs)

Phase 6: Conversational QA Interface
Create Streamlit UI:

Sidebar filters: category (leave, PF, etc.)

Chatbox input

Display cited answer with metadata (filename, chunk)

Use LangChain ConversationalRetrievalChain

