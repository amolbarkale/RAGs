#!/usr/bin/env python3
"""
Enterprise Document Chunking RAG System - Interactive Demo

This demo showcases the complete document processing pipeline:
1. Document classification and adaptive chunking
2. Embedding generation and vector storage
3. Intelligent search and retrieval
4. Support for multiple document types
"""

import sys
import os
import time
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def print_banner():
    """Print demo banner"""
    print("="*70)
    print("  🏢 ENTERPRISE DOCUMENT CHUNKING RAG SYSTEM - DEMO")
    print("="*70)
    print("📚 Intelligent document processing with adaptive chunking strategies")
    print("🔍 Smart classification • ✂️ Context-aware chunking • 🧠 Semantic search")
    print("="*70)


def demo_document_classification():
    """Demo document classification capabilities"""
    print("\n🔍 DOCUMENT CLASSIFICATION DEMO")
    print("-" * 40)
    
    try:
        from document_classifier import create_classifier
        
        classifier = create_classifier()
        
        # Sample documents for each type
        samples = {
            "API Documentation": """
            # User Management API
            
            ## Authentication Endpoints
            
            ### POST /api/auth/login
            Authenticates a user and returns a JWT token.
            
            **Request:**
            ```json
            {"username": "user@example.com", "password": "secret"}
            ```
            """,
            
            "Python Code": """
            import pandas as pd
            from sklearn.metrics import accuracy_score
            
            def evaluate_model(y_true, y_pred):
                '''Calculate model performance metrics'''
                accuracy = accuracy_score(y_true, y_pred)
                return {'accuracy': accuracy}
            
            class DataProcessor:
                def __init__(self, config):
                    self.config = config
            """,
            
            "Company Policy": """
            # Data Privacy Policy
            
            ## Requirements
            
            1. All personal data must be encrypted at rest
            2. Access to customer information requires authorization
            3. Data retention period: 7 years for financial records
            4. GDPR compliance is mandatory for EU customers
            """,
            
            "Support Guide": """
            # Troubleshooting Login Issues
            
            ## Problem: User cannot access their account
            
            **Step 1:** Verify username and password are correct
            **Step 2:** Check if account is locked
            **Step 3:** Clear browser cache and cookies
            **Step 4:** Contact IT support if issue persists
            """,
            
            "Tutorial": """
            # Getting Started with Our API
            
            ## Step 1: Installation
            First, install the required packages:
            ```bash
            pip install our-api-client
            ```
            
            ## Step 2: Authentication
            Set up your API credentials:
            ```python
            client = APIClient(api_key='your-key')
            ```
            
            ## Step 3: First API Call
            Make your first request...
            """
        }
        
        print("📋 Classifying different document types...\n")
        
        for doc_name, content in samples.items():
            result = classifier.classify_document(content, f"{doc_name.lower().replace(' ', '_')}.md")
            
            print(f"📄 {doc_name}")
            print(f"   ➤ Type: {result.document_type.value}")
            print(f"   ➤ Confidence: {result.confidence:.2f}")
            print(f"   ➤ Patterns: {', '.join(result.detected_patterns)}")
            print(f"   ➤ Chunking Strategy: {classifier.get_chunking_strategy(result.document_type)}")
            print()
        
        return True
        
    except Exception as e:
        print(f"❌ Classification demo failed: {e}")
        return False


def demo_chunking_strategies():
    """Demo different chunking strategies"""
    print("\n✂️ CHUNKING STRATEGIES DEMO")
    print("-" * 40)
    
    try:
        from chunking_strategies import chunk_document
        from document_classifier import DocumentType
        
        # Demonstrate each chunking strategy
        examples = {
            "Semantic Chunking (Technical Docs)": {
                "content": """
                Machine learning enables computers to learn from data without explicit programming.
                
                Deep learning is a subset of machine learning that uses neural networks with multiple layers.
                These networks can automatically discover patterns in data.
                
                Natural language processing allows computers to understand and generate human language.
                Applications include chatbots, translation, and sentiment analysis.
                """,
                "doc_type": DocumentType.TECHNICAL_DOC,
                "strategy": "semantic"
            },
            
            "Code-Aware Chunking (Code Docs)": {
                "content": """
                import numpy as np
                from sklearn.linear_model import LogisticRegression
                
                def train_model(X, y):
                    '''Train a logistic regression model'''
                    model = LogisticRegression()
                    model.fit(X, y)
                    return model
                
                def predict(model, X):
                    '''Make predictions using trained model'''
                    return model.predict(X)
                
                class ModelEvaluator:
                    def __init__(self, metrics=['accuracy', 'precision']):
                        self.metrics = metrics
                """,
                "doc_type": DocumentType.CODE_DOC,
                "strategy": "code_aware"
            },
            
            "Hierarchical Chunking (Policy Docs)": {
                "content": """
                # Security Policy
                
                ## Access Control
                ### User Authentication
                All users must authenticate using multi-factor authentication.
                
                ### Password Requirements
                Passwords must be at least 12 characters long.
                
                ## Data Protection
                ### Encryption Standards
                All data must be encrypted using AES-256.
                
                ### Backup Procedures
                Daily backups are required for all critical systems.
                """,
                "doc_type": DocumentType.POLICY_DOC,
                "strategy": "hierarchical"
            }
        }
        
        for example_name, example_data in examples.items():
            print(f"🔧 {example_name}")
            
            chunks = chunk_document(
                content=example_data["content"],
                doc_type=example_data["doc_type"],
                strategy=example_data["strategy"]
            )
            
            print(f"   ➤ Created {len(chunks)} chunks using {example_data['strategy']} strategy")
            
            for i, chunk in enumerate(chunks, 1):
                chunk_type = chunk.metadata.get('chunk_type', 'unknown')
                print(f"   📝 Chunk {i} ({chunk_type}): {len(chunk.content)} chars")
                
                # Show special metadata
                if 'function_name' in chunk.metadata:
                    print(f"      └─ Function: {chunk.metadata['function_name']}")
                if 'section_level' in chunk.metadata:
                    print(f"      └─ Section Level: {chunk.metadata['section_level']}")
                if 'code_language' in chunk.metadata:
                    print(f"      └─ Language: {chunk.metadata['code_language']}")
            
            print()
        
        return True
        
    except Exception as e:
        print(f"❌ Chunking demo failed: {e}")
        return False


def demo_end_to_end_processing():
    """Demo complete end-to-end document processing"""
    print("\n🚀 END-TO-END PROCESSING DEMO")
    print("-" * 40)
    
    try:
        from document_processor import create_document_processor
        
        # Create processor
        print("🔄 Initializing document processor...")
        processor = create_document_processor()
        
        # Sample enterprise documents
        enterprise_docs = [
            ("API Guide", """
            # Payment Processing API
            
            ## Overview
            Our payment API allows secure processing of credit card transactions.
            
            ## Authentication
            Use JWT tokens for API authentication.
            
            ## Endpoints
            
            ### POST /api/payments
            Process a payment transaction.
            
            **Request:**
            ```json
            {
                "amount": 29.99,
                "currency": "USD",
                "card_token": "tok_visa_4242"
            }
            ```
            """),
            
            ("Payment Service", """
            import stripe
            from decimal import Decimal
            
            class PaymentService:
                def __init__(self, api_key):
                    stripe.api_key = api_key
                
                def process_payment(self, amount, currency, card_token):
                    '''Process payment using Stripe'''
                    try:
                        charge = stripe.Charge.create(
                            amount=int(amount * 100),  # Convert to cents
                            currency=currency,
                            source=card_token
                        )
                        return {'success': True, 'charge_id': charge.id}
                    except stripe.error.StripeError as e:
                        return {'success': False, 'error': str(e)}
            """),
            
            ("Payment Policy", """
            # Payment Processing Policy
            
            ## Security Requirements
            
            ### PCI Compliance
            All payment processing must comply with PCI DSS standards.
            
            ### Data Protection
            Credit card information must never be stored in plain text.
            
            ## Fraud Prevention
            
            ### Risk Assessment
            All transactions over $500 require manual review.
            
            ### Monitoring
            Suspicious transaction patterns must be flagged automatically.
            """)
        ]
        
        print(f"📄 Processing {len(enterprise_docs)} enterprise documents...\n")
        
        # Process documents
        results = []
        for doc_name, content in enterprise_docs:
            print(f"📄 Processing: {doc_name}")
            
            result = processor.process_document(
                content=content,
                filename=f"{doc_name.lower().replace(' ', '_')}.md"
            )
            
            results.append(result)
            
            if result.error:
                print(f"   ❌ Error: {result.error}")
            else:
                print(f"   ✅ Success: {result.chunks_stored} chunks stored")
                print(f"   📊 Type: {result.classification.document_type.value}")
                print(f"   ⏱️  Time: {result.processing_time:.2f}s")
            print()
        
        # Demo search functionality
        print("🔍 SEARCH DEMONSTRATION")
        print("-" * 30)
        
        search_queries = [
            "How to process payments?",
            "API authentication methods", 
            "PCI compliance requirements",
            "Stripe payment integration",
            "Fraud prevention policies"
        ]
        
        for query in search_queries:
            print(f"\n❓ Query: '{query}'")
            
            search_results = processor.search_documents(query, top_k=2)
            
            if search_results:
                for i, (chunk, score) in enumerate(search_results, 1):
                    doc_type = chunk.metadata.get('doc_type', 'unknown')
                    chunk_type = chunk.metadata.get('chunk_type', 'unknown')
                    
                    print(f"   {i}. Score: {score:.3f} | Type: {doc_type} | Chunk: {chunk_type}")
                    print(f"      Preview: {chunk.content[:80]}...")
            else:
                print("   No results found")
        
        # Show statistics
        stats = processor.get_processing_stats()
        print(f"\n📊 SYSTEM STATISTICS")
        print(f"   Total chunks stored: {stats['collection_info'].get('points_count', 0)}")
        print(f"   Embedding model: {stats['embedding_model']}")
        print(f"   Vector dimension: {stats['embedding_dimension']}")
        
        return True
        
    except Exception as e:
        print(f"❌ End-to-end demo failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def demo_document_loading():
    """Demo document loading from files"""
    print("\n📁 DOCUMENT LOADING DEMO")
    print("-" * 40)
    
    try:
        from document_loaders import DocumentLoaderFactory, create_sample_documents
        
        # Create sample documents
        print("📝 Creating sample documents...")
        sample_files = create_sample_documents()
        
        # Load documents
        print("\n📂 Loading documents using appropriate loaders...")
        factory = DocumentLoaderFactory()
        
        sample_dir = Path("data/samples")
        if sample_dir.exists():
            for sample_file in sample_files:
                filepath = sample_dir / sample_file
                if filepath.exists():
                    try:
                        doc = factory.load_document(filepath)
                        
                        print(f"\n📄 {doc.filename}")
                        print(f"   ➤ Type: {doc.file_type}")
                        print(f"   ➤ Size: {doc.metadata.get('file_size', 0)} bytes")
                        print(f"   ➤ Loader: {doc.metadata.get('loader', 'unknown')}")
                        
                        # Show type-specific metadata
                        if doc.file_type == "markdown":
                            print(f"   ➤ Headers: {doc.metadata.get('header_count', 0)}")
                            print(f"   ➤ Code blocks: {doc.metadata.get('code_block_count', 0)}")
                        elif doc.file_type == "code":
                            print(f"   ➤ Language: {doc.metadata.get('language', 'unknown')}")
                            print(f"   ➤ Functions: {doc.metadata.get('function_count', 0)}")
                        
                        print(f"   ➤ Content preview: {doc.content[:100]}...")
                        
                    except Exception as e:
                        print(f"   ❌ Failed to load {sample_file}: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Document loading demo failed: {e}")
        return False


def interactive_query_demo():
    """Interactive query demonstration"""
    print("\n💬 INTERACTIVE QUERY DEMO")
    print("-" * 40)
    print("Type queries to search the knowledge base (type 'quit' to exit)")
    
    try:
        from document_processor import create_document_processor
        
        processor = create_document_processor()
        
        while True:
            query = input("\n🔍 Enter your query: ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                break
            
            if not query:
                continue
            
            print(f"\n🔄 Searching for: '{query}'")
            
            results = processor.search_documents(query, top_k=3)
            
            if results:
                print(f"✅ Found {len(results)} relevant results:")
                
                for i, (chunk, score) in enumerate(results, 1):
                    doc_type = chunk.metadata.get('doc_type', 'unknown')
                    filename = chunk.metadata.get('filename', 'unknown')
                    
                    print(f"\n{i}. Score: {score:.3f} | Type: {doc_type} | File: {filename}")
                    print(f"   Content: {chunk.content[:200]}...")
                    
                    if len(chunk.content) > 200:
                        print("   [Content truncated...]")
            else:
                print("❌ No relevant results found")
        
        print("\n👋 Thanks for using the interactive demo!")
        return True
        
    except Exception as e:
        print(f"❌ Interactive demo failed: {e}")
        return False


def run_complete_demo():
    """Run the complete demonstration"""
    print_banner()
    
    demos = [
        ("Document Classification", demo_document_classification),
        ("Chunking Strategies", demo_chunking_strategies),
        ("End-to-End Processing", demo_end_to_end_processing),
        ("Document Loading", demo_document_loading)
    ]
    
    results = []
    
    for demo_name, demo_func in demos:
        print(f"\n🎯 Starting demo: {demo_name}")
        
        try:
            success = demo_func()
            results.append((demo_name, success))
            
            if success:
                print(f"✅ {demo_name} demo completed successfully!")
            else:
                print(f"❌ {demo_name} demo failed!")
                
        except Exception as e:
            print(f"❌ {demo_name} demo crashed: {e}")
            results.append((demo_name, False))
    
    # Summary
    print("\n" + "="*70)
    print("DEMO SUMMARY")
    print("="*70)
    
    successful = sum(1 for _, success in results if success)
    
    for demo_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{demo_name:<25} {status}")
    
    print(f"\nCompleted: {successful}/{len(results)} demos")
    
    if successful == len(results):
        print("\n🎉 All demos completed successfully!")
        print("🚀 The Enterprise Document Chunking RAG System is fully operational!")
        
        # Offer interactive demo
        response = input("\nWould you like to try the interactive query demo? (y/n): ").strip().lower()
        if response in ['y', 'yes']:
            interactive_query_demo()
    else:
        print("\n⚠️  Some demos failed. Check the error messages above.")
        print("💡 Make sure all dependencies are installed and Qdrant server is running.")


if __name__ == "__main__":
    try:
        run_complete_demo()
    except KeyboardInterrupt:
        print("\n\n👋 Demo interrupted by user. Goodbye!")
    except Exception as e:
        print(f"\n❌ Demo crashed: {e}")
        import traceback
        traceback.print_exc() 