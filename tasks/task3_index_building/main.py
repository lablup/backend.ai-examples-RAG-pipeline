#!/usr/bin/env python3
"""
Task 3: Index Building
- Load processed documents and embeddings from Task 2
- Build FAISS vector index for semantic search
- Build BM25 index for keyword search
- Save indexes with metadata for hybrid retrieval

Environment Variables:
- PROCESSED_DIR: Input directory with processed documents
- INDEX_DIR: Output directory for search indexes
- HUGGINGFACE_TOKEN: HuggingFace API token (optional)

Requirements: pip install -r requirements.txt
"""

import logging
import os
import pickle
from pathlib import Path

import numpy as np
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_config():
    """Load configuration from environment variables"""
    config = {
        'processed_dir': Path(os.getenv('PROCESSED_DIR', '../processed')),
        'index_dir': Path(os.getenv('INDEX_DIR', '../indexes')),
        'huggingface_token': os.getenv('HUGGINGFACE_TOKEN', ''),
    }
    
    # Create index directory
    config['index_dir'].mkdir(parents=True, exist_ok=True)
    
    return config

def load_processed_data(processed_dir: Path):
    """Load processed documents, embeddings, and config"""
    # Load documents
    docs_path = processed_dir / "documents.pkl"
    if not docs_path.exists():
        raise FileNotFoundError("documents.pkl not found. Run task2_document_processing.py first.")
    
    with open(docs_path, 'rb') as f:
        documents = pickle.load(f)
    
    # Load embeddings
    embeddings_path = processed_dir / "embeddings.pkl"
    if not embeddings_path.exists():
        raise FileNotFoundError("embeddings.pkl not found. Run task2_document_processing.py first.")
    
    with open(embeddings_path, 'rb') as f:
        doc_embeddings = pickle.load(f)
    
    # Load embedding config
    config_path = processed_dir / "embedding_config.pkl"
    if not config_path.exists():
        raise FileNotFoundError("embedding_config.pkl not found. Run task2_document_processing.py first.")
    
    with open(config_path, 'rb') as f:
        embedding_config = pickle.load(f)
    
    return documents, doc_embeddings, embedding_config

def initialize_embedding_model(embedding_config, config):
    """Initialize the same embedding model used in processing"""
    # Set HuggingFace token if provided
    if config['huggingface_token']:
        os.environ['HF_TOKEN'] = config['huggingface_token']
    
    return HuggingFaceEmbeddings(
        model_name=embedding_config['embed_model']
    )

def build_faiss_index(documents, doc_embeddings, embeddings, index_dir: Path):
    """Build and save FAISS vector index"""
    logger.info("Building FAISS index...")
    
    try:
        # Create FAISS index from documents and pre-computed embeddings
        # Convert embeddings to numpy array
        embeddings_array = np.array(doc_embeddings, dtype=np.float32)
        
        # Create FAISS index using from_embeddings method
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        
        # Create the vectorstore with pre-computed embeddings
        vectorstore = FAISS.from_embeddings(
            text_embeddings=list(zip(texts, embeddings_array)),
            embedding=embeddings,
            metadatas=metadatas
        )
        
        # Save the FAISS index
        faiss_path = index_dir / "faiss"
        vectorstore.save_local(str(faiss_path))
        logger.info(f"FAISS index saved to {faiss_path}")
        
        return vectorstore
        
    except Exception as e:
        logger.error(f"Error building FAISS index: {e}")
        raise

def build_bm25_index(documents, index_dir: Path):
    """Build and save BM25 index"""
    logger.info("Building BM25 index...")
    
    try:
        # Create BM25 retriever
        bm25_retriever = BM25Retriever.from_documents(documents)
        
        # Save BM25 index
        bm25_path = index_dir / "bm25.pkl"
        with open(bm25_path, 'wb') as f:
            pickle.dump(bm25_retriever, f)
        logger.info(f"BM25 index saved to {bm25_path}")
        
        return bm25_retriever
        
    except Exception as e:
        logger.error(f"Error building BM25 index: {e}")
        raise

def main():
    """Main function to build indexes"""
    config = load_config()
    processed_dir = config['processed_dir']
    index_dir = config['index_dir']
    
    # Load processed data
    try:
        documents, doc_embeddings, embedding_config = load_processed_data(processed_dir)
        logger.info(f"Loaded {len(documents)} documents with {len(doc_embeddings)} embeddings")
    except FileNotFoundError as e:
        logger.error(str(e))
        return
    
    # Initialize embedding model
    try:
        embeddings = initialize_embedding_model(embedding_config, config)
        logger.info(f"Initialized embedding model: {embedding_config['embed_model']}")
    except Exception as e:
        logger.error(f"Failed to initialize embedding model: {e}")
        return
    
    # Build FAISS index
    try:
        faiss_index = build_faiss_index(documents, doc_embeddings, embeddings, index_dir)
    except Exception as e:
        logger.error(f"Failed to build FAISS index: {e}")
        return
    
    # Build BM25 index  
    try:
        bm25_index = build_bm25_index(documents, index_dir)
    except Exception as e:
        logger.error(f"Failed to build BM25 index: {e}")
        return
    
    # Save index metadata
    index_metadata = {
        'num_documents': len(documents),
        'embedding_config': embedding_config,
        'faiss_path': str(index_dir / "faiss"),
        'bm25_path': str(index_dir / "bm25.pkl")
    }
    
    metadata_path = index_dir / "index_metadata.pkl"
    with open(metadata_path, 'wb') as f:
        pickle.dump(index_metadata, f)
    logger.info(f"Index metadata saved to {metadata_path}")
    
    logger.info("Index building completed successfully")

if __name__ == "__main__":
    main()