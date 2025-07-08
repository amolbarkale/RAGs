from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_pdf(path):
    """
    Load a PDF file and return raw text from all pages.
    """
    loader = PyPDFLoader(str(path))
    docs = loader.load()
    return "\n".join([doc.page_content for doc in docs])


def get_chunks(text, strategy="Recursive", chunk_size=1000, chunk_overlap=200):
    """
    Apply chunking strategy to the input text.
    Supports: 'Recursive' and 'Markdown-aware'
    """
    if strategy == "Recursive":
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        return splitter.split_text(text)

    elif strategy == "Markdown-aware":
        sections = text.split('\n## ')
        chunks = []
        for section in sections:
            if len(section) <= chunk_size:
                chunks.append(section)
            else:
                paragraphs = section.split('\n\n')
                current_chunk = ""
                for para in paragraphs:
                    if len(current_chunk) + len(para) <= chunk_size:
                        current_chunk += para + "\n\n"
                    else:
                        chunks.append(current_chunk.strip())
                        current_chunk = para + "\n\n"
                if current_chunk:
                    chunks.append(current_chunk.strip())
        return chunks

    # Fallback if unknown strategy
    return [text]
