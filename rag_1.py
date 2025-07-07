import os
from dotenv import load_dotenv
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from langchain_qdrant import QdrantVectorStore

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

# PHASE 1: Ingesting the PDF document into a vector store
# STEP 1: Load the PDF file (LOADER)
pdf_path = Path(__file__).parent / "./MCP.pdf"

loader = PyPDFLoader(file_path = pdf_path)

docs = loader.load() # create list of docs

# STEP 2: Split the documents into smaller chunks (CHUNKING)
text_splitter  = RecursiveCharacterTextSplitter(
    chunk_size = 1000,  # size of each chunk
    chunk_overlap = 200,  # overlap between chunks
)

split_docs = text_splitter.split_documents(documents = docs)
# print('docs:', len(docs))
# print('split_docs:', len(split_docs))

# STEP 3: Create embeddings for the chunks (EMBEDDING)
embedding_model = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001", # "models/text-embedding-004" # or "models/gemini-embedding-exp-03-07"
    google_api_key=api_key
)

# STEP 4: Store the chunks in a vector store (VECTOR STORE)
# vector_store = QdrantVectorStore.from_documents(
#     documents=split_docs,
#     url="http://localhost:6333",  # URL of the Qdrant server
#     collection_name="mcp_collection",  # name of the collection in Qdrant
#     embedding=embedding_model,
# )
print("Ingestion DONE!")

# STEP 5: Create a retriever from the existing collection in Qdrant
retriever = QdrantVectorStore.from_existing_collection(
    url="http://localhost:6333",  # URL of the Qdrant server
    collection_name="mcp_collection",  # name of the collection in Qdrant
    embedding=embedding_model,
)

# STEP 6: Query the retriever
query = "What is the purpose of the MCP?"
# It will create vector embeddings for the query and search the Qdrant collection
relevant_chunks = retriever.similarity_search(query)  # k = number of results to return
print('Relevant chnks:', relevant_chunks)

# TODO: 
# 1] extract the 'page_content' and 'page' numebr from the search results
# 2] add it in the system prompt
# 3] and then pass it to the LLM for generating the final answer

SYSTEM_PROMPT = f"""You are a helpful assistant that answers questions based on the provided context.
The context is extracted from the availbal context.

Context:
{relevant_chunks}
"""