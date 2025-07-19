"""
Chunking Strategies Module for Adaptive Document Processing

This module implements different chunking strategies based on document types:
- Semantic: Embedding-based similarity splitting
- Code-Aware: Preserves functions, classes, and code structure  
- Hierarchical: Section-based with parent-child relationships
"""

import re
import ast
import uuid
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from abc import ABC, abstractmethod

from .document_classifier import DocumentType
from .vector_store import DocumentChunk
from .config import CHUNKING_CONFIG


@dataclass
class ChunkMetadata:
    """Extended metadata for chunks"""
    chunk_type: str
    parent_id: Optional[str] = None
    children_ids: List[str] = None
    section_level: Optional[int] = None
    code_language: Optional[str] = None
    function_name: Optional[str] = None
    
    def __post_init__(self):
        if self.children_ids is None:
            self.children_ids = []


class BaseChunker(ABC):
    """Abstract base class for all chunking strategies"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
    
    @abstractmethod
    def chunk(self, content: str, doc_type: DocumentType, metadata: Optional[Dict] = None) -> List[DocumentChunk]:
        """Chunk document content based on strategy"""
        pass
    
    def _create_chunk(self, content: str, chunk_metadata: ChunkMetadata, 
                     source_metadata: Optional[Dict] = None) -> DocumentChunk:
        """Create a DocumentChunk with proper metadata"""
        metadata = {
            "chunk_type": chunk_metadata.chunk_type,
            "doc_type": source_metadata.get("doc_type", "unknown") if source_metadata else "unknown",
            **(source_metadata or {})
        }
        
        # Add chunk-specific metadata
        if chunk_metadata.parent_id:
            metadata["parent_id"] = chunk_metadata.parent_id
        if chunk_metadata.children_ids:
            metadata["children_ids"] = chunk_metadata.children_ids
        if chunk_metadata.section_level:
            metadata["section_level"] = chunk_metadata.section_level
        if chunk_metadata.code_language:
            metadata["code_language"] = chunk_metadata.code_language
        if chunk_metadata.function_name:
            metadata["function_name"] = chunk_metadata.function_name
        
        return DocumentChunk.create(content=content, **metadata)


class SemanticChunker(BaseChunker):
    """
    Semantic chunking based on embedding similarity and sentence boundaries
    Splits text where semantic meaning changes significantly
    """
    
    def __init__(self, config: Dict[str, Any], embedder=None):
        super().__init__(config)
        self.chunk_size = config.get("chunk_size", 500)
        self.chunk_overlap = config.get("chunk_overlap", 50)
        self.similarity_threshold = config.get("similarity_threshold", 0.8)
        self.embedder = embedder  # Will be injected when available
    
    def chunk(self, content: str, doc_type: DocumentType, metadata: Optional[Dict] = None) -> List[DocumentChunk]:
        """Chunk content using semantic similarity"""
        
        # Split into sentences first
        sentences = self._split_into_sentences(content)
        
        if len(sentences) <= 1:
            # Single sentence or empty content
            chunk_meta = ChunkMetadata(chunk_type="semantic_single")
            return [self._create_chunk(content, chunk_meta, metadata)]
        
        # If embedder is available, use semantic boundaries
        if self.embedder:
            boundaries = self._find_semantic_boundaries(sentences)
        else:
            # Fallback to size-based chunking
            boundaries = self._find_size_based_boundaries(sentences)
        
        # Create chunks based on boundaries
        chunks = []
        start_idx = 0
        
        for boundary_idx in boundaries + [len(sentences)]:
            if boundary_idx > start_idx:
                chunk_sentences = sentences[start_idx:boundary_idx]
                chunk_content = " ".join(chunk_sentences)
                
                # Add overlap from previous chunk
                if chunks and self.chunk_overlap > 0:
                    overlap_content = self._get_overlap_content(chunks[-1].content)
                    if overlap_content:
                        chunk_content = overlap_content + " " + chunk_content
                
                chunk_meta = ChunkMetadata(
                    chunk_type="semantic",
                    section_level=self._detect_section_level(chunk_content)
                )
                chunks.append(self._create_chunk(chunk_content, chunk_meta, metadata))
                start_idx = boundary_idx
        
        return chunks
    
    def _split_into_sentences(self, content: str) -> List[str]:
        """Split content into sentences using regex"""
        # Simple sentence splitting (can be enhanced with nltk/spacy)
        sentence_pattern = r'(?<=[.!?])\s+'
        sentences = re.split(sentence_pattern, content.strip())
        return [s.strip() for s in sentences if s.strip()]
    
    def _find_semantic_boundaries(self, sentences: List[str]) -> List[int]:
        """Find semantic boundaries using embedding similarity"""
        if not self.embedder or len(sentences) < 2:
            return []
        
        try:
            # Get embeddings for each sentence
            embeddings = self.embedder.encode_text(sentences)
            boundaries = []
            
            # Compare consecutive sentences
            for i in range(len(embeddings) - 1):
                similarity = self.embedder.compute_similarity(embeddings[i], embeddings[i + 1])
                
                # If similarity drops below threshold, mark as boundary
                if similarity < self.similarity_threshold:
                    boundaries.append(i + 1)
            
            return boundaries
            
        except Exception as e:
            print(f"Warning: Semantic boundary detection failed: {e}")
            return self._find_size_based_boundaries(sentences)
    
    def _find_size_based_boundaries(self, sentences: List[str]) -> List[int]:
        """Fallback: Find boundaries based on chunk size"""
        boundaries = []
        current_size = 0
        
        for i, sentence in enumerate(sentences):
            current_size += len(sentence)
            
            if current_size >= self.chunk_size and i < len(sentences) - 1:
                boundaries.append(i + 1)
                current_size = 0
        
        return boundaries
    
    def _get_overlap_content(self, previous_chunk: str) -> str:
        """Get overlap content from previous chunk"""
        words = previous_chunk.split()
        if len(words) <= self.chunk_overlap:
            return ""
        
        overlap_words = words[-self.chunk_overlap:]
        return " ".join(overlap_words)
    
    def _detect_section_level(self, content: str) -> Optional[int]:
        """Detect markdown header level in content"""
        header_match = re.search(r'^(#{1,6})\s+', content, re.MULTILINE)
        return len(header_match.group(1)) if header_match else None


class CodeAwareChunker(BaseChunker):
    """
    Code-aware chunking that preserves code structure
    Keeps functions, classes, and imports together
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.chunk_size = config.get("chunk_size", 800)
        self.preserve_functions = config.get("preserve_functions", True)
        self.keep_imports = config.get("keep_imports", True)
    
    def chunk(self, content: str, doc_type: DocumentType, metadata: Optional[Dict] = None) -> List[DocumentChunk]:
        """Chunk content preserving code structure"""
        
        # Detect programming language
        language = self._detect_language(content, metadata)
        
        if language == "python":
            return self._chunk_python_code(content, metadata)
        elif language in ["javascript", "typescript"]:
            return self._chunk_javascript_code(content, metadata)
        else:
            # Generic code chunking
            return self._chunk_generic_code(content, metadata)
    
    def _detect_language(self, content: str, metadata: Optional[Dict] = None) -> str:
        """Detect programming language from content and metadata"""
        if metadata and "filename" in metadata:
            filename = metadata["filename"].lower()
            if filename.endswith('.py'):
                return "python"
            elif filename.endswith(('.js', '.jsx')):
                return "javascript"
            elif filename.endswith(('.ts', '.tsx')):
                return "typescript"
        
        # Content-based detection
        if re.search(r'\bdef\s+\w+\s*\(', content):
            return "python"
        elif re.search(r'\bfunction\s+\w+\s*\(', content):
            return "javascript"
        elif re.search(r'\bclass\s+\w+\s*{', content):
            return "javascript"
        
        return "generic"
    
    def _chunk_python_code(self, content: str, metadata: Optional[Dict] = None) -> List[DocumentChunk]:
        """Chunk Python code preserving functions and classes"""
        chunks = []
        
        try:
            # Parse Python AST
            tree = ast.parse(content)
            
            # Extract imports
            imports = self._extract_python_imports(tree, content)
            if imports:
                chunk_meta = ChunkMetadata(
                    chunk_type="code_imports",
                    code_language="python"
                )
                chunks.append(self._create_chunk(imports, chunk_meta, metadata))
            
            # Extract functions and classes
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                    func_content = self._extract_node_content(node, content)
                    if func_content:
                        chunk_meta = ChunkMetadata(
                            chunk_type="code_function" if isinstance(node, ast.FunctionDef) else "code_class",
                            code_language="python",
                            function_name=node.name
                        )
                        chunks.append(self._create_chunk(func_content, chunk_meta, metadata))
            
            # If no functions/classes found, fall back to generic chunking
            if len(chunks) <= 1:  # Only imports or nothing
                return self._chunk_generic_code(content, metadata)
            
            return chunks
            
        except SyntaxError:
            # If parsing fails, fall back to generic chunking
            return self._chunk_generic_code(content, metadata)
    
    def _extract_python_imports(self, tree: ast.AST, content: str) -> str:
        """Extract import statements from Python AST"""
        import_lines = []
        lines = content.split('\n')
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                if hasattr(node, 'lineno') and node.lineno <= len(lines):
                    import_lines.append(lines[node.lineno - 1])
        
        return '\n'.join(import_lines) if import_lines else ""
    
    def _extract_node_content(self, node: ast.AST, content: str) -> str:
        """Extract content for a specific AST node"""
        lines = content.split('\n')
        
        if hasattr(node, 'lineno') and hasattr(node, 'end_lineno'):
            start_line = node.lineno - 1
            end_line = node.end_lineno if node.end_lineno else len(lines)
            return '\n'.join(lines[start_line:end_line])
        
        return ""
    
    def _chunk_javascript_code(self, content: str, metadata: Optional[Dict] = None) -> List[DocumentChunk]:
        """Chunk JavaScript/TypeScript code"""
        chunks = []
        
        # Extract imports/requires
        import_pattern = r'(?:import.*?from.*?[\'"][^\'"]*[\'"]|const.*?=\s*require\([\'"][^\'"]*[\'"]\));?'
        imports = re.findall(import_pattern, content, re.MULTILINE)
        
        if imports:
            import_content = '\n'.join(imports)
            chunk_meta = ChunkMetadata(
                chunk_type="code_imports",
                code_language="javascript"
            )
            chunks.append(self._create_chunk(import_content, chunk_meta, metadata))
        
        # Extract functions
        function_pattern = r'(?:function\s+\w+\s*\([^)]*\)\s*{[^}]*}|const\s+\w+\s*=\s*\([^)]*\)\s*=>\s*{[^}]*})'
        functions = re.findall(function_pattern, content, re.DOTALL)
        
        for func in functions:
            # Extract function name
            name_match = re.search(r'(?:function\s+(\w+)|const\s+(\w+))', func)
            func_name = name_match.group(1) or name_match.group(2) if name_match else "anonymous"
            
            chunk_meta = ChunkMetadata(
                chunk_type="code_function",
                code_language="javascript",
                function_name=func_name
            )
            chunks.append(self._create_chunk(func, chunk_meta, metadata))
        
        # If no functions found, fall back to generic
        if len(chunks) <= 1:
            return self._chunk_generic_code(content, metadata)
        
        return chunks
    
    def _chunk_generic_code(self, content: str, metadata: Optional[Dict] = None) -> List[DocumentChunk]:
        """Generic code chunking by size with code block preservation"""
        # Split by code blocks first
        code_block_pattern = r'```[\s\S]*?```'
        blocks = re.split(f'({code_block_pattern})', content)
        
        chunks = []
        current_chunk = ""
        
        for block in blocks:
            if re.match(code_block_pattern, block):
                # This is a code block - keep it together
                if len(current_chunk) + len(block) > self.chunk_size and current_chunk:
                    # Save current chunk and start new one
                    chunk_meta = ChunkMetadata(chunk_type="code_generic")
                    chunks.append(self._create_chunk(current_chunk.strip(), chunk_meta, metadata))
                    current_chunk = block
                else:
                    current_chunk += block
            else:
                # Regular text - can be split
                if len(current_chunk) + len(block) > self.chunk_size and current_chunk:
                    chunk_meta = ChunkMetadata(chunk_type="code_generic")
                    chunks.append(self._create_chunk(current_chunk.strip(), chunk_meta, metadata))
                    current_chunk = block
                else:
                    current_chunk += block
        
        # Add final chunk
        if current_chunk.strip():
            chunk_meta = ChunkMetadata(chunk_type="code_generic")
            chunks.append(self._create_chunk(current_chunk.strip(), chunk_meta, metadata))
        
        return chunks


class HierarchicalChunker(BaseChunker):
    """
    Hierarchical chunking that preserves document structure
    Creates parent-child relationships based on headers and sections
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.respect_headers = config.get("respect_headers", True)
        self.max_depth = config.get("max_depth", 3)
        self.min_chunk_size = config.get("min_chunk_size", 200)
    
    def chunk(self, content: str, doc_type: DocumentType, metadata: Optional[Dict] = None) -> List[DocumentChunk]:
        """Chunk content preserving hierarchical structure"""
        
        # Parse document structure
        sections = self._parse_sections(content)
        
        if not sections:
            # No structure found, create single chunk
            chunk_meta = ChunkMetadata(chunk_type="hierarchical_single")
            return [self._create_chunk(content, chunk_meta, metadata)]
        
        # Build hierarchical chunks
        chunks = []
        section_map = {}  # Maps section IDs to chunk IDs
        
        for section in sections:
            chunk = self._create_section_chunk(section, metadata, section_map)
            chunks.append(chunk)
            section_map[section["id"]] = chunk.id
        
        # Set parent-child relationships
        self._set_chunk_relationships(chunks, sections)
        
        return chunks
    
    def _parse_sections(self, content: str) -> List[Dict[str, Any]]:
        """Parse document into hierarchical sections"""
        sections = []
        lines = content.split('\n')
        current_section = None
        section_counter = 0
        
        for line_num, line in enumerate(lines):
            header_match = re.match(r'^(#{1,6})\s+(.+)$', line.strip())
            
            if header_match:
                # Found a header - start new section
                if current_section and current_section["content"].strip():
                    sections.append(current_section)
                
                level = len(header_match.group(1))
                title = header_match.group(2).strip()
                
                current_section = {
                    "id": f"section_{section_counter}",
                    "level": level,
                    "title": title,
                    "content": line + '\n',
                    "start_line": line_num,
                    "parent_id": self._find_parent_section(sections, level)
                }
                section_counter += 1
            else:
                # Add line to current section
                if current_section:
                    current_section["content"] += line + '\n'
                else:
                    # Content before first header
                    if not sections:
                        sections.append({
                            "id": f"section_{section_counter}",
                            "level": 0,
                            "title": "Introduction",
                            "content": line + '\n',
                            "start_line": line_num,
                            "parent_id": None
                        })
                        section_counter += 1
                    else:
                        sections[-1]["content"] += line + '\n'
        
        # Add final section
        if current_section and current_section["content"].strip():
            sections.append(current_section)
        
        return sections
    
    def _find_parent_section(self, sections: List[Dict], current_level: int) -> Optional[str]:
        """Find the parent section for the current header level"""
        for section in reversed(sections):
            if section["level"] < current_level:
                return section["id"]
        return None
    
    def _create_section_chunk(self, section: Dict, metadata: Optional[Dict], section_map: Dict) -> DocumentChunk:
        """Create a chunk from a section"""
        chunk_meta = ChunkMetadata(
            chunk_type="hierarchical_section",
            section_level=section["level"],
            parent_id=section.get("parent_id")
        )
        
        # Add section-specific metadata
        section_metadata = {
            "section_id": section["id"],
            "section_title": section["title"],
            "section_level": section["level"],
            **(metadata or {})
        }
        
        return self._create_chunk(section["content"].strip(), chunk_meta, section_metadata)
    
    def _set_chunk_relationships(self, chunks: List[DocumentChunk], sections: List[Dict]):
        """Set parent-child relationships between chunks"""
        # Create mapping from section ID to chunk
        section_to_chunk = {}
        for chunk, section in zip(chunks, sections):
            section_to_chunk[section["id"]] = chunk
        
        # Set children IDs for parent chunks
        for chunk, section in zip(chunks, sections):
            if section.get("parent_id") and section["parent_id"] in section_to_chunk:
                parent_chunk = section_to_chunk[section["parent_id"]]
                if hasattr(parent_chunk.metadata, "children_ids"):
                    parent_chunk.metadata["children_ids"].append(chunk.id)
                else:
                    parent_chunk.metadata["children_ids"] = [chunk.id]


class ChunkingStrategyFactory:
    """Factory for creating chunking strategies based on document type"""
    
    @staticmethod
    def create_chunker(strategy: str, embedder=None) -> BaseChunker:
        """Create appropriate chunker based on strategy name"""
        
        if strategy == "semantic":
            config = CHUNKING_CONFIG["semantic"]
            return SemanticChunker(config, embedder)
        
        elif strategy == "code_aware":
            config = CHUNKING_CONFIG["code_aware"]
            return CodeAwareChunker(config)
        
        elif strategy == "hierarchical":
            config = CHUNKING_CONFIG["hierarchical"]
            return HierarchicalChunker(config)
        
        else:
            # Default to semantic
            config = CHUNKING_CONFIG["semantic"]
            return SemanticChunker(config, embedder)


def chunk_document(content: str, 
                  doc_type: DocumentType, 
                  strategy: str = None,
                  embedder=None,
                  metadata: Optional[Dict] = None) -> List[DocumentChunk]:
    """
    Convenience function to chunk a document using appropriate strategy
    
    Args:
        content: Document content
        doc_type: Detected document type
        strategy: Chunking strategy (if None, determined from doc_type)
        embedder: Embedding model for semantic chunking
        metadata: Additional metadata for chunks
        
    Returns:
        List of DocumentChunk objects
    """
    
    # Determine strategy if not provided
    if strategy is None:
        from .document_classifier import create_classifier
        classifier = create_classifier()
        strategy = classifier.get_chunking_strategy(doc_type)
    
    # Create chunker and process
    chunker = ChunkingStrategyFactory.create_chunker(strategy, embedder)
    return chunker.chunk(content, doc_type, metadata) 