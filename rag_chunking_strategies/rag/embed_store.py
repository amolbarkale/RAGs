from langchain.schema import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from typing import List
import os

def create_documents_from_chunks(chunks: List[str]) -> List[Document]:
    """Wrap raw text chunks into LangChain Document objects"""
    return [Document(page_content=chunk) for chunk in chunks]

def embed_and_store(
    docs: List[Document],
    collection_name: str,
    api_key: str,
    qdrant_url: str = "http://localhost:6333"
):
    """Embeds documents and stores them into Qdrant vector DB"""
    embedding_model = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=api_key
    )

    vectorstore = QdrantVectorStore.from_documents(
        documents=docs,
        embedding=embedding_model,
        url=qdrant_url,
        collection_name=collection_name,
    )

    return vectorstore
