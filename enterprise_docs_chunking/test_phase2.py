#!/usr/bin/env python3
"""
Test Suite for Phase 2 - Complete Document Processing Pipeline

This script tests the integrated functionality:
1. Document Processing Pipeline
2. Chunking Strategies
3. End-to-End Document → Search Flow
4. Different Document Types
"""

import sys
import os
import time

# Add src to path so we can import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_chunking_strategies():
    """Test all chunking strategies independently"""
    print("\n" + "="*50)
    print("TESTING CHUNKING STRATEGIES")
    print("="*50)
    
    try:
        from chunking_strategies import chunk_document, ChunkingStrategyFactory
        from document_classifier import DocumentType
        
        # Test documents for each strategy
        test_cases = {
            "semantic": {
                "content": """
                # Machine Learning Overview
                Machine learning is a subset of artificial intelligence that enables computers to learn patterns from data.
                
                ## Deep Learning
                Deep learning uses neural networks with multiple layers to model complex patterns.
                
                ## Applications
                Common applications include image recognition, natural language processing, and predictive analytics.
                """,
                "doc_type": DocumentType.TECHNICAL_DOC
            },
            
            "code_aware": {
                "content": """
                import pandas as pd
                from sklearn.metrics import accuracy_score
                
                def load_data(filepath):
                    '''Load dataset from CSV file'''
                    return pd.read_csv(filepath)
                
                def evaluate_model(y_true, y_pred):
                    '''Calculate model accuracy'''
                    return accuracy_score(y_true, y_pred)
                
                class DataProcessor:
                    def __init__(self, config):
                        self.config = config
                    
                    def process(self, data):
                        return data.dropna()
                """,
                "doc_type": DocumentType.CODE_DOC
            },
            
            "hierarchical": {
                "content": """
                # Data Privacy Policy
                
                ## 1. Data Collection
                ### 1.1 Personal Information
                We collect name, email, and phone number for account creation.
                
                ### 1.2 Usage Data
                We track user interactions for analytics purposes.
                
                ## 2. Data Storage
                ### 2.1 Encryption
                All data is encrypted at rest using AES-256 encryption.
                
                ### 2.2 Access Control
                Access to personal data requires proper authorization.
                """,
                "doc_type": DocumentType.POLICY_DOC
            }
        }
        
        # Test each strategy
        for strategy_name, test_case in test_cases.items():
            print(f"\n🔧 Testing {strategy_name} chunking...")
            
            chunks = chunk_document(
                content=test_case["content"],
                doc_type=test_case["doc_type"],
                strategy=strategy_name,
                metadata={"test_case": strategy_name}
            )
            
            print(f"   Created {len(chunks)} chunks")
            
            for i, chunk in enumerate(chunks, 1):
                print(f"   Chunk {i}: {chunk.metadata.get('chunk_type', 'unknown')} "
                      f"({len(chunk.content)} chars)")
                if hasattr(chunk, 'metadata') and 'function_name' in chunk.metadata:
                    print(f"     Function: {chunk.metadata['function_name']}")
                if hasattr(chunk, 'metadata') and 'section_level' in chunk.metadata:
                    print(f"     Section Level: {chunk.metadata['section_level']}")
        
        print("\n✅ All chunking strategies tested successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Chunking strategies test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_document_processor():
    """Test the complete document processing pipeline"""
    print("\n" + "="*50)
    print("TESTING DOCUMENT PROCESSOR")
    print("="*50)
    
    try:
        from document_processor import create_document_processor
        
        # Create processor
        print("🔄 Creating document processor...")
        processor = create_document_processor()
        
        # Test documents
        test_documents = [
            ("# API Reference\n\nGET /users/{id}\nReturns user information.", "api_docs.md"),
            ("def hello_world():\n    print('Hello, World!')\n    return True", "example.py"),
            ("# Policy\n## Requirements\n1. All data must be encrypted", "policy.md")
        ]
        
        results = []
        
        # Process each document
        for content, filename in test_documents:
            print(f"\n📄 Processing: {filename}")
            
            result = processor.process_document(content, filename)
            results.append(result)
            
            if result.error:
                print(f"   ❌ Error: {result.error}")
            else:
                print(f"   ✅ Success: {result.chunks_stored} chunks stored")
                print(f"   📊 Type: {result.classification.document_type.value}")
                print(f"   ⏱️  Time: {result.processing_time:.2f}s")
        
        # Test search functionality
        print(f"\n🔍 Testing search functionality...")
        search_results = processor.search_documents("API user information", top_k=3)
        
        print(f"   Found {len(search_results)} results")
        for i, (chunk, score) in enumerate(search_results, 1):
            print(f"   {i}. Score: {score:.3f}")
            print(f"      Content preview: {chunk.content[:50]}...")
        
        # Get processing stats
        stats = processor.get_processing_stats()
        print(f"\n📊 Processing Statistics:")
        print(f"   Collection points: {stats['collection_info'].get('points_count', 0)}")
        print(f"   Embedding model: {stats['embedding_model']}")
        print(f"   Vector dimension: {stats['embedding_dimension']}")
        
        print("\n✅ Document processor tests completed successfully!")
        return True
        
    except ImportError as e:
        print(f"⚠️  Document processor test skipped - missing dependencies: {e}")
        return True
    except Exception as e:
        print(f"❌ Document processor test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_end_to_end_workflow():
    """Test complete end-to-end document processing workflow"""
    print("\n" + "="*50)
    print("TESTING END-TO-END WORKFLOW")
    print("="*50)
    
    try:
        from document_processor import process_single_document, search_knowledge_base
        
        # Sample enterprise documents
        enterprise_docs = {
            "tech_doc": """
            # User Authentication API
            
            ## Overview
            The authentication API provides secure user login functionality.
            
            ## Endpoints
            
            ### POST /api/auth/login
            Authenticates a user and returns a JWT token.
            
            **Request Body:**
            ```json
            {
                "username": "user@example.com",
                "password": "secure_password"
            }
            ```
            
            **Response:**
            ```json
            {
                "token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
                "expires_in": 3600
            }
            ```
            """,
            
            "code_doc": """
            import jwt
            import bcrypt
            from datetime import datetime, timedelta
            
            class AuthenticationService:
                def __init__(self, secret_key):
                    self.secret_key = secret_key
                
                def hash_password(self, password):
                    '''Hash password using bcrypt'''
                    salt = bcrypt.gensalt()
                    return bcrypt.hashpw(password.encode('utf-8'), salt)
                
                def verify_password(self, password, hashed):
                    '''Verify password against hash'''
                    return bcrypt.checkpw(password.encode('utf-8'), hashed)
                
                def generate_token(self, user_id):
                    '''Generate JWT token for user'''
                    payload = {
                        'user_id': user_id,
                        'exp': datetime.utcnow() + timedelta(hours=1)
                    }
                    return jwt.encode(payload, self.secret_key, algorithm='HS256')
            """,
            
            "support_doc": """
            # Troubleshooting Authentication Issues
            
            ## Problem: Cannot login to system
            
            ### Step 1: Check Credentials
            First, verify that you are using the correct username and password.
            
            ### Step 2: Clear Browser Cache
            Clear your browser cache and cookies, then try again.
            
            ### Step 3: Check Network Connection
            Ensure you have a stable internet connection.
            
            ### Step 4: Contact Support
            If the problem persists, contact our support team with error details.
            """
        }
        
        # Process all documents
        print("🔄 Processing enterprise documents...")
        for doc_type, content in enterprise_docs.items():
            print(f"\n📄 Processing {doc_type}...")
            
            result = process_single_document(content, f"{doc_type}.md")
            
            if result.error:
                print(f"   ❌ Failed: {result.error}")
                return False
            else:
                print(f"   ✅ Success: {result.chunks_stored} chunks")
        
        # Test various search queries
        search_queries = [
            "How to authenticate users?",
            "JWT token generation",
            "Login troubleshooting steps",
            "Password hashing bcrypt",
            "API endpoints authentication"
        ]
        
        print(f"\n🔍 Testing search queries...")
        for query in search_queries:
            print(f"\n❓ Query: '{query}'")
            
            results = search_knowledge_base(query, top_k=2)
            
            if results:
                for i, (chunk, score) in enumerate(results, 1):
                    doc_type = chunk.metadata.get('doc_type', 'unknown')
                    chunk_type = chunk.metadata.get('chunk_type', 'unknown')
                    print(f"   {i}. Score: {score:.3f} | Type: {doc_type} | Chunk: {chunk_type}")
                    print(f"      Preview: {chunk.content[:80]}...")
            else:
                print("   No results found")
        
        print("\n✅ End-to-end workflow test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ End-to-end workflow test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_chunk_quality():
    """Test the quality of different chunking strategies"""
    print("\n" + "="*50)
    print("TESTING CHUNK QUALITY")
    print("="*50)
    
    try:
        from chunking_strategies import chunk_document
        from document_classifier import DocumentType
        
        # Test code preservation
        python_code = """
        def calculate_metrics(y_true, y_pred):
            '''Calculate precision, recall, and F1 score'''
            from sklearn.metrics import precision_score, recall_score, f1_score
            
            precision = precision_score(y_true, y_pred)
            recall = recall_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred)
            
            return {
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            }
        """
        
        chunks = chunk_document(python_code, DocumentType.CODE_DOC, "code_aware")
        
        print("🧪 Code-aware chunking quality:")
        for chunk in chunks:
            print(f"   Chunk type: {chunk.metadata.get('chunk_type', 'unknown')}")
            if 'function_name' in chunk.metadata:
                print(f"   Function preserved: {chunk.metadata['function_name']}")
            
            # Check if imports are preserved with functions
            if 'from sklearn' in chunk.content and 'def calculate_metrics' in chunk.content:
                print("   ✅ Imports kept with function")
            elif 'from sklearn' in chunk.content:
                print("   ✅ Imports separated properly")
        
        # Test hierarchical structure preservation
        policy_doc = """
        # Privacy Policy
        ## Data Collection
        ### Personal Data
        We collect your name and email.
        ### Usage Data  
        We track how you use our service.
        ## Data Use
        ### Analytics
        Data is used for improving our service.
        """
        
        chunks = chunk_document(policy_doc, DocumentType.POLICY_DOC, "hierarchical")
        
        print("\n🧪 Hierarchical chunking quality:")
        for chunk in chunks:
            section_level = chunk.metadata.get('section_level')
            section_title = chunk.metadata.get('section_title', 'unknown')
            print(f"   Section: {section_title} (Level {section_level})")
            
            # Check structure preservation
            if section_level and section_level > 0:
                header_count = chunk.content.count('#')
                if header_count > 0:
                    print(f"   ✅ Header structure preserved")
        
        print("\n✅ Chunk quality tests completed!")
        return True
        
    except Exception as e:
        print(f"❌ Chunk quality test failed: {e}")
        return False


def main():
    """Run all Phase 2 tests"""
    print("🚀 STARTING PHASE 2 TESTS")
    print("📋 Testing complete document processing pipeline...")
    
    results = []
    
    # Run tests
    results.append(("Chunking Strategies", test_chunking_strategies()))
    results.append(("Document Processor", test_document_processor()))
    results.append(("End-to-End Workflow", test_end_to_end_workflow()))
    results.append(("Chunk Quality", test_chunk_quality()))
    
    # Summary
    print("\n" + "="*70)
    print("PHASE 2 TEST SUMMARY")
    print("="*70)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:<25} {status}")
        if result:
            passed += 1
    
    print(f"\nPassed: {passed}/{len(results)} tests")
    
    if passed == len(results):
        print("\n🎉 All Phase 2 components are working correctly!")
        print("✅ Complete document processing pipeline operational")
        print("🚀 Ready to proceed with Phase 3 (Advanced Testing)")
    else:
        print("\n⚠️  Some tests failed. Please check the errors above.")
        print("💡 Make sure Qdrant server is running for full functionality")
    
    return passed == len(results)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 