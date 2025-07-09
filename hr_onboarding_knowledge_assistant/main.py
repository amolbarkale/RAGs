import os
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load environment variables
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

# Define paths
docs_path = Path("data/")
all_files = list(docs_path.glob("*.pdf")) + list(docs_path.glob("*.txt"))

# Load and combine documents
docs = []
for file in all_files:
    if file.suffix == ".pdf":
        loader = PyPDFLoader(str(file))
    elif file.suffix == ".txt":
        loader = TextLoader(str(file), encoding="utf-8")
    else:
        continue
    docs.extend(loader.load())

# Split docs
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
split_docs = splitter.split_documents(docs)

# Embed model
embedding_model = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=api_key
)

# Ingest documents into Qdrant
print("Ingesting docs into Qdrant...")
vector_store = QdrantVectorStore.from_existing_collection(
    documents=split_docs,
    url="http://localhost:6333",
    collection_name="hr_onboarding_collection",
    embedding=embedding_model,
)
print("✅ Ingestion complete!")

# Now create retriever
retriever = QdrantVectorStore.from_existing_collection(
    url="http://localhost:6333",
    collection_name="hr_onboarding_collection",
    embedding=embedding_model,
)

# Set Streamlit UI
st.set_page_config(page_title="HR Assistant", layout="wide")
st.title("💼 HR Onboarding Knowledge Assistant")
st.markdown("Ask any question about leaves, benefits, PF, or onboarding policies:")

query = st.text_input("🔍 Ask your HR question", placeholder="e.g., What is the PF withdrawal process?")

# If query entered
if query:
    with st.spinner("Thinking..."):
        try:
            # Load embedding model
            embedding_model = GoogleGenerativeAIEmbeddings(
                model="models/embedding-001",
                google_api_key=api_key
            )

            # Load retriever from existing Qdrant collection
            retriever = QdrantVectorStore.from_documents(
                url="http://localhost:6333",
                collection_name="hr_collection",
                embedding=embedding_model,
                documents=split_docs, # this is missing
            )

            # Search similar chunks
            relevant_chunks = retriever.similarity_search(query, k=4)

            # Build context string
            context_text = ""
            sources = []
            for doc in relevant_chunks:
                context_text += doc.page_content.strip() + "\n---\n"
                source = doc.metadata.get("source")
                if source and source not in sources:
                    sources.append(source)

            # System prompt template
            SYSTEM_PROMPT = f"""You are a helpful HR assistant.
            Answer the user's question based only on the context below.
            If the answer is not in the context, say "I couldn't find that in the documents."

            Context:
            {context_text}

            Question: {query}
            Helpful Answer:
            """

            # Initialize Gemini chat model
            llm = ChatGoogleGenerativeAI(
                model="gemini-2.0-flash",
                google_api_key=api_key,
                temperature=0.3
            )

            response = llm.invoke(SYSTEM_PROMPT)

            # Display response
            st.markdown("### ✅ Answer")
            st.success(response)

            st.markdown("### 📄 Sources")
            for i, src in enumerate(sources, 1):
                st.markdown(f"- {i}. `{Path(src).name}`")

        except Exception as e:
            st.error(f"⚠️ Error: {e}")
else:
    st.info("Enter your HR-related question above.")
