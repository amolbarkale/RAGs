import os
import re
from typing import List

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_file(file_path: str) -> str:
    """Load content from PDF or TXT file"""
    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        text = "\n".join([doc.page_content for doc in docs])
    elif file_path.endswith(".txt"):
        loader = TextLoader(file_path)
        text = loader.load().page_content
    else:
        raise ValueError("Unsupported file format. Only PDF and TXT are allowed.")
    return text


def recursive_chunking(text: str, chunk_size=1000, chunk_overlap=200) -> List[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    return splitter.split_text(text)


def markdown_aware_chunking(text: str, max_chunk_size=1000) -> List[str]:
    """Chunk markdown-like text by respecting section headers"""
    sections = text.split('\n## ')
    chunks = []

    for section in sections:
        if len(section) <= max_chunk_size:
            chunks.append(section.strip())
        else:
            paras = section.split('\n\n')
            current = ""
            for para in paras:
                if len(current) + len(para) <= max_chunk_size:
                    current += para + "\n\n"
                else:
                    chunks.append(current.strip())
                    current = para + "\n\n"
            if current:
                chunks.append(current.strip())

    return chunks


def rule_based_hr_chunking(text: str) -> List[str]:
    """Chunk HR policy content based on known section headers"""
    patterns = [
        "LEAVE POLICY", "BENEFITS", "COMPANY POLICY", "CODE OF CONDUCT",
        "ATTENDANCE", "REIMBURSEMENTS", "WORK FROM HOME", "OVERTIME",
        "TERMINATION POLICY", "GRIEVANCE", "PAYROLL", "PERFORMANCE REVIEW"
    ]

    # Match all headings (case-insensitive)
    regex = '|'.join([f"({re.escape(p)})" for p in patterns])
    matches = list(re.finditer(regex, text.upper()))

    if not matches:
        return [text]

    chunks = []
    for i in range(len(matches)):
        start = matches[i].start()
        end = matches[i+1].start() if i + 1 < len(matches) else len(text)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

    return chunks


def get_chunks(text: str, strategy: str = "recursive") -> List[str]:
    """Select and run the appropriate chunking strategy"""
    if strategy == "recursive":
        return recursive_chunking(text)
    elif strategy == "markdown":
        return markdown_aware_chunking(text)
    elif strategy == "rule_based":
        return rule_based_hr_chunking(text)
    else:
        raise ValueError(f"Unknown chunking strategy: {strategy}")
