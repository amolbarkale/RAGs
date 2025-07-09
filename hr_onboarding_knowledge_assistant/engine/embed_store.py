import os
from dotenv import load_dotenv
from typing import List

from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore
# Initialize
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
COLLECTION_NAME = "hr_documents"

# Initialize Gemini embeddings
embedding_model = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=GEMINI_API_KEY
)

# Store chunks in Qdrant
def store_documents(docs: List[Document]):
    vectorstore = QdrantVectorStore.from_documents(
        documents=docs,
        embedding=embedding_model,
        location="http://localhost:6333",
        collection_name=COLLECTION_NAME
    )
    print(f"[✅] Stored {len(docs)} documents in Qdrant.")
    return vectorstore
