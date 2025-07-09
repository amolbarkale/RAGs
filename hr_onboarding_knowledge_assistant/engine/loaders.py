from langchain_community.document_loaders import PyPDFLoader, TextLoader
from pathlib import Path

def load_documents_from_data_folder(folder_path="data"):
    documents = []
    data_path = Path(folder_path)

    for file_path in data_path.glob("*"):
        if file_path.suffix == ".pdf":
            loader = PyPDFLoader(str(file_path))
        elif file_path.suffix == ".txt":
            loader = TextLoader(str(file_path))
        else:
            continue

        docs = loader.load()
        documents.extend(docs)

    return documents
