#!/usr/bin/env python3
"""
Script to export vector database content to a JSON file.
This is useful for debugging, analysis, and backup purposes.
"""

import os
import sys
import json
import time
from datetime import datetime
from typing import Dict, List, Any, Optional

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import Django settings
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "zproject.settings")
import django
django.setup()

from zproject.vector_db_mock import VectorStore


def export_vector_db_to_json(output_file: str = None) -> None:
    """
    Export the vector database content to a JSON file
    
    Args:
        output_file: Path to output file. If None, a timestamped file will be created.
    """
    # Try to get the shared vector database instance
    try:
        from zerver.lib.ai_agents_openai import get_shared_vector_store
        vector_db = get_shared_vector_store()
        print("Using shared vector store instance")
    except (ImportError, AttributeError):
        # Fallback to creating a new instance
        vector_db = VectorStore()
        print("WARNING: Using a new vector store instance, may not contain actual data")
    
    # Get all collections
    collections_data = {}
    
    # Debug information about the vector database instance
    print(f"Vector store type: {type(vector_db).__name__}")
    print(f"Vector store ID: {id(vector_db)}")
    
    # Print directory of vector_db to see all attributes and methods
    print("\nVector store attributes:")
    for attr in dir(vector_db):
        if not attr.startswith('__'):
            try:
                value = getattr(vector_db, attr)
                if not callable(value):
                    print(f"  {attr}: {type(value).__name__}")
            except Exception as e:
                print(f"  {attr}: Error accessing - {e}")
    
    if hasattr(vector_db, 'collections'):
        print(f"\nCollections found: {list(vector_db.collections.keys())}")
        
        for collection_name, documents in vector_db.collections.items():
            print(f"\nCollection '{collection_name}' has {len(documents)} documents")
            
            # Convert documents to serializable format
            serializable_docs = []
            
            for i, doc in enumerate(documents):
                # Debug individual documents
                print(f"  Document {i+1}:")
                print(f"    Content: {doc.get('content', '')[:50]}...")
                print(f"    Metadata keys: {list(doc.get('metadata', {}).keys())}")
                
                # Create a copy of document for serialization
                serializable_doc = {
                    "content": doc.get("content", ""),
                    "metadata": doc.get("metadata", {})
                }
                
                # Omit embedding vectors to keep file size manageable
                # but store embedding dimensions if available
                embedding = doc.get("embedding", None)
                if embedding:
                    serializable_doc["embedding_dimensions"] = len(embedding)
                    print(f"    Embedding dimensions: {len(embedding)}")
                
                serializable_docs.append(serializable_doc)
            
            collections_data[collection_name] = serializable_docs
    else:
        print("\nNo 'collections' attribute found in vector store")
    
    # Prepare export data
    export_data = {
        "export_timestamp": datetime.now().isoformat(),
        "collections": collections_data
    }
    
    # Generate output filename if not provided
    if not output_file:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_file = f"vector_db_export_{timestamp}.json"
    
    # Write to file
    with open(output_file, 'w') as f:
        json.dump(export_data, f, indent=2)
    
    print(f"Vector database exported to {output_file}")
    print(f"Collections found: {list(collections_data.keys())}")
    print(f"Total documents: {sum(len(docs) for docs in collections_data.values())}")


if __name__ == "__main__":
    # Check if output file is specified as command line argument
    output_file = sys.argv[1] if len(sys.argv) > 1 else None
    export_vector_db_to_json(output_file)
