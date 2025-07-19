# 🔄 Enterprise Document Chunking - System Flow

## 📊 High-Level System Flow

```mermaid
graph TB
    A[📄 Input Documents] --> B{🔍 Document Classifier}
    B --> C1[📖 Technical Doc]
    B --> C2[💻 Code Doc]
    B --> C3[📋 Policy Doc]
    B --> C4[🆘 Support Doc]
    B --> C5[📚 Tutorial]
    
    C1 --> D1[✂️ Semantic Chunking]
    C2 --> D2[✂️ Code-Aware Chunking]
    C3 --> D3[✂️ Hierarchical Chunking]
    C4 --> D1
    C5 --> D3
    
    D1 --> E[🧠 Embedding Generation]
    D2 --> E
    D3 --> E
    
    E --> F[💾 Qdrant Vector Store]
    F --> G[🔍 Semantic Search API]
    
    style A fill:#e1f5fe
    style B fill:#f3e5f5
    style E fill:#e8f5e8
    style F fill:#fff3e0
    style G fill:#e8eaf6
```

## 🔍 Document Classification Flow

```mermaid
graph LR
    A[📄 Document + Filename] --> B[🔍 Pattern Analysis]
    A --> C[📂 Extension Analysis]
    
    B --> D[Headers Detection]
    B --> E[Code Patterns]
    B --> F[Policy Terms]
    B --> G[Tutorial Markers]
    
    C --> H[File Extension Scoring]
    
    D --> I[📊 Content Score 70%]
    E --> I
    F --> I
    G --> I
    
    H --> J[📂 Extension Score 30%]
    
    I --> K[🎯 Combined Scoring]
    J --> K
    
    K --> L{Confidence > 0.7?}
    L -->|Yes| M[✅ Document Type]
    L -->|No| N[❓ Unknown Type]
    
    M --> O[⚙️ Strategy Selection]
    N --> P[🔄 Fallback Strategy]
```

## ✂️ Chunking Strategy Details

### Semantic Chunking Flow
```mermaid
graph TB
    A[📄 Document Content] --> B[🔤 Sentence Splitting]
    B --> C{🧠 Embedder Available?}
    
    C -->|Yes| D[🔢 Generate Embeddings]
    C -->|No| E[📏 Size-Based Boundaries]
    
    D --> F[📊 Similarity Analysis]
    F --> G[🎯 Boundary Detection]
    G --> H[✂️ Smart Split]
    
    E --> I[📏 Size-Based Split]
    
    H --> J[📝 Semantic Chunks]
    I --> J
    
    J --> K[🔗 Add Overlap]
    K --> L[✅ Final Chunks]
```

### Code-Aware Chunking Flow
```mermaid
graph TB
    A[💻 Code Content] --> B[🔍 Language Detection]
    B --> C{Language Type}
    
    C -->|Python| D[🐍 Python AST Parser]
    C -->|JavaScript| E[⚡ JS Pattern Matcher]
    C -->|Other| F[🔧 Generic Code Parser]
    
    D --> G[📥 Extract Imports]
    D --> H[🔧 Extract Functions]
    D --> I[🏗️ Extract Classes]
    
    E --> J[📥 Extract Requires/Imports]
    E --> K[🔧 Extract Functions]
    E --> L[🏗️ Extract Classes]
    
    F --> M[📏 Generic Parsing]
    
    G --> N[📦 Group Imports]
    H --> O[🔧 Preserve Functions]
    I --> P[🏗️ Preserve Classes]
    
    J --> N
    K --> O
    L --> P
    
    M --> Q[📝 Generic Chunks]
    
    N --> R[✅ Code Chunks]
    O --> R
    P --> R
    Q --> R
```

### Hierarchical Chunking Flow
```mermaid
graph TB
    A[📋 Structured Document] --> B[🔍 Header Detection]
    B --> C[📊 Parse Document Structure]
    C --> D[🏗️ Build Section Tree]
    
    D --> E[📝 Section 1]
    D --> F[📝 Section 2]
    D --> G[📝 Section 3]
    
    E --> H[📝 Subsection 1.1]
    E --> I[📝 Subsection 1.2]
    
    F --> J[📝 Subsection 2.1]
    
    H --> K[👨‍👩‍👧‍👦 Parent-Child Links]
    I --> K
    J --> K
    
    K --> L[🔗 Breadcrumb Metadata]
    L --> M[✅ Hierarchical Chunks]
```

## 🧠 Embedding & Storage Pipeline

```mermaid
graph LR
    A[📝 Text Chunks] --> B[🤖 HuggingFace Model]
    B --> C[🔢 384D Vectors]
    
    C --> D[📋 Chunk Metadata]
    D --> E[📦 Combined Payload]
    
    E --> F[🗄️ Qdrant Collection]
    F --> G[📊 Vector Index]
    
    G --> H[🔍 Similarity Search]
    H --> I[📈 Ranked Results]
    
    style B fill:#e8f5e8
    style F fill:#fff3e0
    style H fill:#e8eaf6
```

## 🔍 Search & Retrieval Flow

```mermaid
graph TB
    A[❓ User Query] --> B[🧠 Query Embedding]
    B --> C[🔍 Vector Search]
    
    C --> D{🏷️ Filters Applied?}
    D -->|Yes| E[📋 Metadata Filter]
    D -->|No| F[🔍 Direct Search]
    
    E --> F
    F --> G[📊 Cosine Similarity]
    G --> H[📈 Score Ranking]
    
    H --> I[🎯 Top-K Results]
    I --> J[📝 Chunk Content]
    I --> K[📋 Metadata]
    I --> L[📊 Similarity Score]
    
    J --> M[✅ Search Results]
    K --> M
    L --> M
```

## 📊 Data Transformation Pipeline

```mermaid
graph LR
    A[📄 Raw Document] --> B[🔍 Classification]
    B --> C[📊 Document Type]
    
    C --> D[✂️ Chunking Strategy]
    D --> E[📝 Text Chunks]
    
    E --> F[🧠 Embedding Model]
    F --> G[🔢 Vector Embeddings]
    
    G --> H[📋 Metadata Attachment]
    H --> I[💾 Vector Storage]
    
    I --> J[🔍 Searchable Index]
    
    subgraph "Metadata Flow"
        K[📂 File Info] --> H
        L[🏷️ Chunk Type] --> H
        M[👨‍👩‍👧‍👦 Relationships] --> H
        N[📊 Confidence] --> H
    end
```

## 🎯 Quality Control Flow

```mermaid
graph TB
    A[📝 Generated Chunks] --> B[🔍 Quality Checks]
    
    B --> C{📏 Size Check}
    B --> D{🧠 Coherence Check}
    B --> E{🏗️ Structure Check}
    
    C -->|Pass| F[✅ Size Valid]
    C -->|Fail| G[⚠️ Size Warning]
    
    D -->|Pass| H[✅ Coherent]
    D -->|Fail| I[⚠️ Split Required]
    
    E -->|Pass| J[✅ Structure Preserved]
    E -->|Fail| K[⚠️ Structure Lost]
    
    F --> L[📊 Quality Score]
    G --> L
    H --> L
    I --> L
    J --> L
    K --> L
    
    L --> M{Quality > Threshold?}
    M -->|Yes| N[✅ Accept Chunks]
    M -->|No| O[🔄 Retry Processing]
```

## 📈 Performance Monitoring Flow

```mermaid
graph LR
    A[⏱️ Processing Start] --> B[📊 Metrics Collection]
    
    B --> C[⏱️ Processing Time]
    B --> D[📏 Chunk Count]
    B --> E[🎯 Classification Confidence]
    B --> F[💾 Storage Success]
    
    C --> G[📈 Performance Dashboard]
    D --> G
    E --> G
    F --> G
    
    G --> H[📊 Analytics]
    H --> I[🔄 Optimization Insights]
    
    I --> J[⚙️ Parameter Tuning]
    J --> K[🎯 Improved Performance]
```

## 🔄 Complete System Integration

```mermaid
graph TB
    subgraph "Input Layer"
        A1[📄 PDF Documents]
        A2[📝 Markdown Files]
        A3[💻 Code Files]
        A4[📋 Text Documents]
    end
    
    subgraph "Processing Layer"
        B[🔍 Document Loader]
        C[🎯 Document Classifier]
        D[✂️ Chunking Engine]
        E[🧠 Embedding Generator]
    end
    
    subgraph "Storage Layer"
        F[💾 Qdrant Vector DB]
        G[📊 Metadata Store]
    end
    
    subgraph "API Layer"
        H[🔍 Search API]
        I[📝 Processing API]
        J[📊 Analytics API]
    end
    
    A1 --> B
    A2 --> B
    A3 --> B
    A4 --> B
    
    B --> C
    C --> D
    D --> E
    E --> F
    E --> G
    
    F --> H
    G --> H
    F --> I
    G --> I
    F --> J
    G --> J
    
    style B fill:#e1f5fe
    style C fill:#f3e5f5
    style D fill:#e8f5e8
    style E fill:#fff3e0
    style F fill:#e8eaf6
    style H fill:#f1f8e9
```

---

## 📋 Processing Steps Summary

1. **📄 Document Input**: Multi-format document ingestion
2. **🔍 Classification**: Pattern-based document type detection
3. **⚙️ Strategy Selection**: Mapping document type to chunking strategy
4. **✂️ Adaptive Chunking**: Content-aware chunk generation
5. **🧠 Embedding**: Vector representation generation
6. **💾 Storage**: Persistent vector storage with metadata
7. **🔍 Search**: Semantic similarity search and ranking
8. **📊 Analytics**: Performance monitoring and optimization

Each step is designed to preserve document structure and semantic meaning while optimizing for retrieval accuracy in enterprise knowledge management scenarios. 