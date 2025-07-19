"""
Document Loaders for Different File Types

This module provides loaders for various document formats commonly found
in enterprise environments: PDF, Markdown, Text files, and more.
"""

import os
import re
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
from dataclasses import dataclass

import pypdf
import markdown
from bs4 import BeautifulSoup

from .config import DATA_DIRS


@dataclass
class LoadedDocument:
    """Represents a loaded document with metadata"""
    content: str
    filename: str
    file_type: str
    metadata: Dict[str, Any]
    
    @property
    def file_extension(self) -> str:
        return Path(self.filename).suffix.lower().lstrip('.')


class BaseDocumentLoader:
    """Base class for all document loaders"""
    
    def __init__(self):
        self.supported_extensions = []
    
    def can_load(self, filepath: Union[str, Path]) -> bool:
        """Check if this loader can handle the file"""
        extension = Path(filepath).suffix.lower().lstrip('.')
        return extension in self.supported_extensions
    
    def load(self, filepath: Union[str, Path]) -> LoadedDocument:
        """Load document from file"""
        raise NotImplementedError


class PDFLoader(BaseDocumentLoader):
    """Loader for PDF documents"""
    
    def __init__(self):
        super().__init__()
        self.supported_extensions = ['pdf']
    
    def load(self, filepath: Union[str, Path]) -> LoadedDocument:
        """Load PDF document"""
        filepath = Path(filepath)
        
        try:
            with open(filepath, 'rb') as file:
                pdf_reader = pypdf.PdfReader(file)
                
                # Extract text from all pages
                text_content = []
                for page_num, page in enumerate(pdf_reader.pages):
                    page_text = page.extract_text()
                    if page_text.strip():
                        text_content.append(f"--- Page {page_num + 1} ---\n{page_text}")
                
                content = "\n\n".join(text_content)
                
                # Extract metadata
                metadata = {
                    "source": str(filepath),
                    "total_pages": len(pdf_reader.pages),
                    "file_size": filepath.stat().st_size,
                    "loader": "PDFLoader"
                }
                
                # Add PDF metadata if available
                if pdf_reader.metadata:
                    pdf_meta = pdf_reader.metadata
                    metadata.update({
                        "title": pdf_meta.get("/Title", ""),
                        "author": pdf_meta.get("/Author", ""),
                        "subject": pdf_meta.get("/Subject", ""),
                        "creator": pdf_meta.get("/Creator", ""),
                        "producer": pdf_meta.get("/Producer", ""),
                        "creation_date": str(pdf_meta.get("/CreationDate", "")),
                    })
                
                return LoadedDocument(
                    content=content,
                    filename=filepath.name,
                    file_type="pdf",
                    metadata=metadata
                )
                
        except Exception as e:
            raise Exception(f"Error loading PDF {filepath}: {str(e)}")


class MarkdownLoader(BaseDocumentLoader):
    """Loader for Markdown documents"""
    
    def __init__(self):
        super().__init__()
        self.supported_extensions = ['md', 'markdown', 'mdown', 'mkd']
    
    def load(self, filepath: Union[str, Path]) -> LoadedDocument:
        """Load Markdown document"""
        filepath = Path(filepath)
        
        try:
            with open(filepath, 'r', encoding='utf-8') as file:
                content = file.read()
            
            # Parse markdown structure
            metadata = self._extract_markdown_metadata(content)
            metadata.update({
                "source": str(filepath),
                "file_size": filepath.stat().st_size,
                "loader": "MarkdownLoader",
                "encoding": "utf-8"
            })
            
            return LoadedDocument(
                content=content,
                filename=filepath.name,
                file_type="markdown",
                metadata=metadata
            )
            
        except UnicodeDecodeError:
            # Try different encodings
            for encoding in ['latin-1', 'cp1252', 'iso-8859-1']:
                try:
                    with open(filepath, 'r', encoding=encoding) as file:
                        content = file.read()
                    
                    metadata = self._extract_markdown_metadata(content)
                    metadata.update({
                        "source": str(filepath),
                        "file_size": filepath.stat().st_size,
                        "loader": "MarkdownLoader",
                        "encoding": encoding
                    })
                    
                    return LoadedDocument(
                        content=content,
                        filename=filepath.name,
                        file_type="markdown",
                        metadata=metadata
                    )
                    
                except UnicodeDecodeError:
                    continue
            
            raise Exception(f"Could not decode markdown file {filepath}")
        except Exception as e:
            raise Exception(f"Error loading Markdown {filepath}: {str(e)}")
    
    def _extract_markdown_metadata(self, content: str) -> Dict[str, Any]:
        """Extract metadata from markdown content"""
        metadata = {}
        
        # Count headers at different levels
        headers = re.findall(r'^(#{1,6})\s+(.+)$', content, re.MULTILINE)
        metadata["header_count"] = len(headers)
        metadata["max_header_level"] = max([len(h[0]) for h in headers]) if headers else 0
        
        # Extract first header as potential title
        if headers:
            metadata["title"] = headers[0][1].strip()
        
        # Count code blocks
        code_blocks = re.findall(r'```[\s\S]*?```', content)
        metadata["code_block_count"] = len(code_blocks)
        
        # Count links
        links = re.findall(r'\[([^\]]+)\]\(([^)]+)\)', content)
        metadata["link_count"] = len(links)
        
        # Detect if it has YAML front matter
        if content.startswith('---'):
            yaml_match = re.match(r'^---\n(.*?)\n---', content, re.DOTALL)
            if yaml_match:
                metadata["has_yaml_frontmatter"] = True
        
        return metadata


class TextLoader(BaseDocumentLoader):
    """Loader for plain text documents"""
    
    def __init__(self):
        super().__init__()
        self.supported_extensions = ['txt', 'text', 'log', 'csv']
    
    def load(self, filepath: Union[str, Path]) -> LoadedDocument:
        """Load text document"""
        filepath = Path(filepath)
        
        try:
            # Try UTF-8 first
            with open(filepath, 'r', encoding='utf-8') as file:
                content = file.read()
            encoding = 'utf-8'
            
        except UnicodeDecodeError:
            # Try other common encodings
            for encoding in ['latin-1', 'cp1252', 'iso-8859-1']:
                try:
                    with open(filepath, 'r', encoding=encoding) as file:
                        content = file.read()
                    break
                except UnicodeDecodeError:
                    continue
            else:
                raise Exception(f"Could not decode text file {filepath}")
        
        # Analyze text structure
        metadata = self._analyze_text_structure(content)
        metadata.update({
            "source": str(filepath),
            "file_size": filepath.stat().st_size,
            "loader": "TextLoader",
            "encoding": encoding
        })
        
        return LoadedDocument(
            content=content,
            filename=filepath.name,
            file_type="text",
            metadata=metadata
        )
    
    def _analyze_text_structure(self, content: str) -> Dict[str, Any]:
        """Analyze structure of text content"""
        lines = content.split('\n')
        
        metadata = {
            "line_count": len(lines),
            "character_count": len(content),
            "word_count": len(content.split()),
            "empty_lines": sum(1 for line in lines if not line.strip()),
            "avg_line_length": sum(len(line) for line in lines) / len(lines) if lines else 0
        }
        
        # Detect if it might be CSV
        if content.count(',') > len(lines) * 2:  # Rough heuristic
            metadata["possible_csv"] = True
            
            # Try to detect delimiter
            first_line = lines[0] if lines else ""
            if ',' in first_line:
                metadata["csv_delimiter"] = ","
            elif '\t' in first_line:
                metadata["csv_delimiter"] = "\t"
            elif ';' in first_line:
                metadata["csv_delimiter"] = ";"
        
        # Detect log file patterns
        if any(re.search(r'\d{4}-\d{2}-\d{2}', line) for line in lines[:10]):
            metadata["possible_log_file"] = True
        
        return metadata


class CodeLoader(BaseDocumentLoader):
    """Loader for source code files"""
    
    def __init__(self):
        super().__init__()
        self.supported_extensions = [
            'py', 'js', 'ts', 'jsx', 'tsx', 'java', 'cpp', 'c', 'h', 
            'cs', 'php', 'rb', 'go', 'rs', 'swift', 'kt', 'scala'
        ]
    
    def load(self, filepath: Union[str, Path]) -> LoadedDocument:
        """Load source code file"""
        filepath = Path(filepath)
        extension = filepath.suffix.lower().lstrip('.')
        
        try:
            with open(filepath, 'r', encoding='utf-8') as file:
                content = file.read()
            
            metadata = self._analyze_code_structure(content, extension)
            metadata.update({
                "source": str(filepath),
                "file_size": filepath.stat().st_size,
                "loader": "CodeLoader",
                "language": extension,
                "encoding": "utf-8"
            })
            
            return LoadedDocument(
                content=content,
                filename=filepath.name,
                file_type="code",
                metadata=metadata
            )
            
        except UnicodeDecodeError:
            # Try different encoding for older code files
            for encoding in ['latin-1', 'cp1252']:
                try:
                    with open(filepath, 'r', encoding=encoding) as file:
                        content = file.read()
                    
                    metadata = self._analyze_code_structure(content, extension)
                    metadata.update({
                        "source": str(filepath),
                        "file_size": filepath.stat().st_size,
                        "loader": "CodeLoader",
                        "language": extension,
                        "encoding": encoding
                    })
                    
                    return LoadedDocument(
                        content=content,
                        filename=filepath.name,
                        file_type="code",
                        metadata=metadata
                    )
                    
                except UnicodeDecodeError:
                    continue
            
            raise Exception(f"Could not decode code file {filepath}")
        except Exception as e:
            raise Exception(f"Error loading code file {filepath}: {str(e)}")
    
    def _analyze_code_structure(self, content: str, language: str) -> Dict[str, Any]:
        """Analyze code structure"""
        lines = content.split('\n')
        metadata = {
            "line_count": len(lines),
            "language": language
        }
        
        # Language-specific analysis
        if language == 'py':
            metadata.update(self._analyze_python_code(content))
        elif language in ['js', 'ts', 'jsx', 'tsx']:
            metadata.update(self._analyze_javascript_code(content))
        elif language == 'java':
            metadata.update(self._analyze_java_code(content))
        
        # Count comments
        comment_patterns = {
            'py': r'#.*$',
            'js': r'//.*$',
            'ts': r'//.*$',
            'java': r'//.*$',
            'cpp': r'//.*$',
            'c': r'//.*$'
        }
        
        if language in comment_patterns:
            comments = re.findall(comment_patterns[language], content, re.MULTILINE)
            metadata["comment_count"] = len(comments)
        
        return metadata
    
    def _analyze_python_code(self, content: str) -> Dict[str, Any]:
        """Analyze Python-specific code structure"""
        metadata = {}
        
        # Count functions and classes
        functions = re.findall(r'^def\s+(\w+)', content, re.MULTILINE)
        classes = re.findall(r'^class\s+(\w+)', content, re.MULTILINE)
        imports = re.findall(r'^(?:import|from)\s+(\w+)', content, re.MULTILINE)
        
        metadata.update({
            "function_count": len(functions),
            "class_count": len(classes),
            "import_count": len(imports),
            "function_names": functions[:10],  # First 10 functions
            "class_names": classes[:10],  # First 10 classes
        })
        
        return metadata
    
    def _analyze_javascript_code(self, content: str) -> Dict[str, Any]:
        """Analyze JavaScript/TypeScript code structure"""
        metadata = {}
        
        # Count different function declarations
        function_decls = re.findall(r'function\s+(\w+)', content)
        arrow_functions = re.findall(r'const\s+(\w+)\s*=.*?=>', content)
        classes = re.findall(r'class\s+(\w+)', content)
        
        metadata.update({
            "function_declarations": len(function_decls),
            "arrow_functions": len(arrow_functions),
            "class_count": len(classes),
            "total_functions": len(function_decls) + len(arrow_functions)
        })
        
        return metadata
    
    def _analyze_java_code(self, content: str) -> Dict[str, Any]:
        """Analyze Java code structure"""
        metadata = {}
        
        methods = re.findall(r'(?:public|private|protected)?\s*(?:static)?\s*\w+\s+(\w+)\s*\(', content)
        classes = re.findall(r'(?:public|private)?\s*class\s+(\w+)', content)
        interfaces = re.findall(r'(?:public)?\s*interface\s+(\w+)', content)
        
        metadata.update({
            "method_count": len(methods),
            "class_count": len(classes),
            "interface_count": len(interfaces)
        })
        
        return metadata


class DocumentLoaderFactory:
    """Factory for creating appropriate document loaders"""
    
    def __init__(self):
        self.loaders = [
            PDFLoader(),
            MarkdownLoader(), 
            CodeLoader(),
            TextLoader()  # TextLoader should be last (fallback)
        ]
    
    def get_loader(self, filepath: Union[str, Path]) -> BaseDocumentLoader:
        """Get appropriate loader for file"""
        for loader in self.loaders:
            if loader.can_load(filepath):
                return loader
        
        # Default to text loader
        return TextLoader()
    
    def load_document(self, filepath: Union[str, Path]) -> LoadedDocument:
        """Load document using appropriate loader"""
        loader = self.get_loader(filepath)
        return loader.load(filepath)
    
    def load_documents_from_directory(self, 
                                    directory: Union[str, Path],
                                    recursive: bool = True,
                                    file_pattern: Optional[str] = None) -> List[LoadedDocument]:
        """Load all documents from a directory"""
        directory = Path(directory)
        documents = []
        
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")
        
        # Get file pattern
        pattern = file_pattern or "*"
        
        # Find files
        if recursive:
            files = directory.rglob(pattern)
        else:
            files = directory.glob(pattern)
        
        # Load each file
        for filepath in files:
            if filepath.is_file():
                try:
                    doc = self.load_document(filepath)
                    documents.append(doc)
                    print(f"✅ Loaded: {filepath.name}")
                except Exception as e:
                    print(f"⚠️  Failed to load {filepath.name}: {e}")
        
        print(f"📁 Loaded {len(documents)} documents from {directory}")
        return documents


# Convenience functions
def load_document(filepath: Union[str, Path]) -> LoadedDocument:
    """Load a single document"""
    factory = DocumentLoaderFactory()
    return factory.load_document(filepath)


def load_documents_from_directory(directory: Union[str, Path], 
                                recursive: bool = True) -> List[LoadedDocument]:
    """Load all documents from directory"""
    factory = DocumentLoaderFactory()
    return factory.load_documents_from_directory(directory, recursive)


def create_sample_documents():
    """Create sample documents for testing"""
    sample_dir = DATA_DIRS["samples"]
    sample_dir.mkdir(parents=True, exist_ok=True)
    
    samples = {
        "api_documentation.md": """
# User Management API

## Overview
This API provides endpoints for managing user accounts in the system.

## Authentication
All endpoints require JWT token authentication.

## Endpoints

### GET /api/users
Returns a list of all users.

**Response:**
```json
{
  "users": [
    {"id": 1, "name": "John Doe", "email": "john@example.com"}
  ]
}
```

### POST /api/users
Creates a new user account.
        """,
        
        "user_service.py": """
import bcrypt
import jwt
from datetime import datetime, timedelta

class UserService:
    def __init__(self, db_connection):
        self.db = db_connection
        
    def create_user(self, email, password, name):
        '''Create a new user account'''
        hashed_password = self.hash_password(password)
        user_id = self.db.insert_user(email, hashed_password, name)
        return user_id
    
    def authenticate_user(self, email, password):
        '''Authenticate user credentials'''
        user = self.db.get_user_by_email(email)
        if user and self.verify_password(password, user.password_hash):
            return self.generate_token(user.id)
        return None
    
    def hash_password(self, password):
        return bcrypt.hashpw(password.encode(), bcrypt.gensalt())
        """,
        
        "privacy_policy.txt": """
PRIVACY POLICY

1. INFORMATION WE COLLECT
We collect information you provide directly to us, such as when you create an account, make a purchase, or contact us for support.

2. HOW WE USE YOUR INFORMATION
We use the information we collect to provide, maintain, and improve our services.

3. INFORMATION SHARING
We do not sell, trade, or otherwise transfer your personal information to third parties without your consent.

4. DATA SECURITY
We implement appropriate security measures to protect your personal information against unauthorized access.
        """
    }
    
    for filename, content in samples.items():
        filepath = sample_dir / filename
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content.strip())
    
    print(f"📝 Created {len(samples)} sample documents in {sample_dir}")
    return list(samples.keys()) 