"""
Document Classification Module for Adaptive Chunking

This module implements rule-based and pattern-based classification
to determine the appropriate chunking strategy for different document types.
"""

import re
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum

from .config import DOCUMENT_TYPES, CLASSIFICATION_CONFIG


class DocumentType(Enum):
    """Enumeration of supported document types"""
    TECHNICAL_DOC = "technical_doc"
    CODE_DOC = "code_doc"
    POLICY_DOC = "policy_doc"
    SUPPORT_DOC = "support_doc"
    TUTORIAL = "tutorial"
    UNKNOWN = "unknown"


@dataclass
class ClassificationResult:
    """Result of document classification"""
    document_type: DocumentType
    confidence: float
    detected_patterns: List[str]
    metadata: Dict[str, Any]


class DocumentClassifier:
    """
    Classifies documents based on content patterns and metadata
    to determine the optimal chunking strategy.
    """
    
    def __init__(self):
        self.confidence_threshold = CLASSIFICATION_CONFIG["confidence_threshold"]
        self.fallback_strategy = CLASSIFICATION_CONFIG["fallback_strategy"]
        
        # Compile regex patterns for efficiency
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Compile regex patterns for content detection"""
        self.patterns = {
            # Code patterns
            "code_blocks": re.compile(r'```[\s\S]*?```|`[^`]+`'),
            "function_def": re.compile(r'(def |function |class |public |private |protected )'),
            "imports": re.compile(r'(import |from |#include |require\()'),
            "code_comments": re.compile(r'(//|#|/\*|\*/)'),
            
            # Documentation patterns
            "headers": re.compile(r'^#{1,6}\s+.+$', re.MULTILINE),
            "api_endpoints": re.compile(r'(GET|POST|PUT|DELETE|PATCH)\s+/\w+'),
            "code_syntax": re.compile(r'\.(py|js|ts|java|cpp|html|css)$'),
            
            # Policy/Support patterns
            "policy_terms": re.compile(r'(policy|procedure|compliance|regulation|requirement)', re.IGNORECASE),
            "support_terms": re.compile(r'(issue|ticket|problem|solution|troubleshoot)', re.IGNORECASE),
            "step_by_step": re.compile(r'(step \d+|^\d+\.|first|second|third|next|then)', re.IGNORECASE),
            
            # Tutorial patterns
            "tutorial_terms": re.compile(r'(tutorial|guide|how to|walkthrough|example)', re.IGNORECASE),
            "numbered_steps": re.compile(r'^\s*\d+\.', re.MULTILINE),
        }
    
    def classify_document(self, 
                         content: str, 
                         filename: Optional[str] = None,
                         metadata: Optional[Dict] = None) -> ClassificationResult:
        """
        Classify a document based on content and metadata
        
        Args:
            content: Document text content
            filename: Optional filename for extension-based classification
            metadata: Optional metadata dictionary
            
        Returns:
            ClassificationResult with type, confidence, and detected patterns
        """
        detected_patterns = []
        scores = {doc_type: 0.0 for doc_type in DocumentType}
        
        # File extension analysis
        if filename:
            file_ext = Path(filename).suffix.lower().lstrip('.')
            extension_score = self._classify_by_extension(file_ext)
            for doc_type, score in extension_score.items():
                scores[doc_type] += score * 0.3  # 30% weight for extension
        
        # Content pattern analysis
        content_scores = self._analyze_content_patterns(content)
        for doc_type, (score, patterns) in content_scores.items():
            scores[doc_type] += score * 0.7  # 70% weight for content
            detected_patterns.extend(patterns)
        
        # Determine best classification
        best_type = max(scores.keys(), key=lambda k: scores[k])
        confidence = scores[best_type]
        
        # Apply confidence threshold
        if confidence < self.confidence_threshold:
            best_type = DocumentType.UNKNOWN
            confidence = 0.0
        
        return ClassificationResult(
            document_type=best_type,
            confidence=confidence,
            detected_patterns=list(set(detected_patterns)),
            metadata={
                "filename": filename,
                "content_length": len(content),
                "all_scores": {k.value: v for k, v in scores.items()},
                **(metadata or {})
            }
        )
    
    def _classify_by_extension(self, extension: str) -> Dict[DocumentType, float]:
        """Classify based on file extension"""
        scores = {doc_type: 0.0 for doc_type in DocumentType}
        
        for doc_type_str, extensions in DOCUMENT_TYPES.items():
            if extension in extensions:
                doc_type = DocumentType(doc_type_str)
                scores[doc_type] = 1.0
                break
        
        return scores
    
    def _analyze_content_patterns(self, content: str) -> Dict[DocumentType, Tuple[float, List[str]]]:
        """Analyze content patterns to determine document type"""
        scores = {doc_type: (0.0, []) for doc_type in DocumentType}
        
        # Count pattern matches
        pattern_counts = {}
        for pattern_name, pattern in self.patterns.items():
            matches = pattern.findall(content)
            pattern_counts[pattern_name] = len(matches)
        
        # Calculate scores for each document type
        
        # Technical Documentation
        tech_score = 0.0
        tech_patterns = []
        if pattern_counts["headers"] > 0:
            tech_score += 0.3
            tech_patterns.append("headers")
        if pattern_counts["api_endpoints"] > 0:
            tech_score += 0.4
            tech_patterns.append("api_endpoints")
        if pattern_counts["code_blocks"] > 0:
            tech_score += 0.3
            tech_patterns.append("code_blocks")
        scores[DocumentType.TECHNICAL_DOC] = (tech_score, tech_patterns)
        
        # Code Documentation
        code_score = 0.0
        code_patterns = []
        if pattern_counts["function_def"] > 0:
            code_score += 0.4
            code_patterns.append("function_definitions")
        if pattern_counts["imports"] > 0:
            code_score += 0.3
            code_patterns.append("imports")
        if pattern_counts["code_comments"] > 2:
            code_score += 0.3
            code_patterns.append("code_comments")
        scores[DocumentType.CODE_DOC] = (code_score, code_patterns)
        
        # Policy Documentation
        policy_score = 0.0
        policy_patterns = []
        if pattern_counts["policy_terms"] > 0:
            policy_score += 0.6
            policy_patterns.append("policy_terms")
        if pattern_counts["headers"] > 0:
            policy_score += 0.2
            policy_patterns.append("structured_headers")
        scores[DocumentType.POLICY_DOC] = (policy_score, policy_patterns)
        
        # Support Documentation
        support_score = 0.0
        support_patterns = []
        if pattern_counts["support_terms"] > 0:
            support_score += 0.5
            support_patterns.append("support_terms")
        if pattern_counts["step_by_step"] > 0:
            support_score += 0.3
            support_patterns.append("troubleshooting_steps")
        scores[DocumentType.SUPPORT_DOC] = (support_score, support_patterns)
        
        # Tutorial
        tutorial_score = 0.0
        tutorial_patterns = []
        if pattern_counts["tutorial_terms"] > 0:
            tutorial_score += 0.4
            tutorial_patterns.append("tutorial_terms")
        if pattern_counts["numbered_steps"] > 2:
            tutorial_score += 0.4
            tutorial_patterns.append("numbered_steps")
        if pattern_counts["code_blocks"] > 0:
            tutorial_score += 0.2
            tutorial_patterns.append("example_code")
        scores[DocumentType.TUTORIAL] = (tutorial_score, tutorial_patterns)
        
        return scores
    
    def get_chunking_strategy(self, doc_type: DocumentType) -> str:
        """Get recommended chunking strategy for document type"""
        strategy_mapping = {
            DocumentType.TECHNICAL_DOC: "semantic",
            DocumentType.CODE_DOC: "code_aware", 
            DocumentType.POLICY_DOC: "hierarchical",
            DocumentType.SUPPORT_DOC: "semantic",
            DocumentType.TUTORIAL: "hierarchical",
            DocumentType.UNKNOWN: "semantic"  # fallback
        }
        return strategy_mapping.get(doc_type, "semantic")


def create_classifier() -> DocumentClassifier:
    """Factory function to create a document classifier"""
    return DocumentClassifier() 