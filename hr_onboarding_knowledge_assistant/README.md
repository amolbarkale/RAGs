# 💼 HR Onboarding Knowledge Assistant

An AI-powered assistant that answers HR-related questions from internal documents like policies, leave rules, benefits, and more — using Retrieval-Augmented Generation (RAG) with Gemini and Qdrant.

---

## ✨ Project Setup

### ⚙️ Prerequisites

- Python 3.8+
- [`uv`](https://github.com/astral-sh/uv) package manager  
- Docker & Docker Compose

### 👨‍💻 Setup Instructions

1. **Clone the repo**
   ```bash
   git clone https://github.com/amolbarkale/RAGs/tree/main/hr_onboarding_knowledge_assistant
   cd hr-onboarding-knowledge-assistant
   ```

2. **Initialize the project with uv**
   ```bash
   uv init hr-onboarding-assistant
   uv add streamlit langchain langchain-community langchain-google-genai langchain-qdrant qdrant-client python-dotenv
   ```
   This creates and manages a virtual environment automatically.

3. **Start Qdrant Vector DB**
   ```bash
   docker compose -f docker-compose.db.yml up
   ```
   This runs a Qdrant vector store at http://localhost:6333.

4. **Add your documents**
   Place your `.pdf` or `.txt` HR policy files inside the `data/` folder:
   ```bash
   mkdir -p data/
   # Drop your HR docs here
   ```

5. **Run the assistant UI**
   ```bash
   uv run streamlit run main.py
   ```
   You'll see a local Streamlit app where you can ask questions like:
   - "How many vacation days do I get?"
   - "What's the parental leave policy?"
   - "How do I enroll in health insurance?"

## 🔐 Environment Variables

Create a `.env` file with your Gemini API key:
```env
GEMINI_API_KEY=your_google_generative_ai_key
```

## 📦 Tech Stack

- **LangChain** – for RAG orchestration
- **Google Gemini** – embeddings + chat
- **Qdrant** – vector store (via Docker)
- **Streamlit** – frontend UI
- **Python 3.8+**

## 📝 Note

You only need to ingest documents once. On subsequent runs, it will query Qdrant directly for retrieval.