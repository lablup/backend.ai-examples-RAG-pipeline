#!/usr/bin/env python3
"""
Task 2: Document Processing
- Load cleaned text files from Task 1
- Split documents into chunks with configurable overlap
- Generate embeddings using HuggingFace models
- Extract metadata from source markers
- Save processed documents and embeddings for indexing

Environment Variables:
- CACHE_DIR: Input directory with cleaned text files
- PROCESSED_DIR: Output directory for processed documents
- EMBED_MODEL: HuggingFace embedding model name (required)
- CHUNK_SIZE: Text chunk size for splitting
- CHUNK_OVERLAP: Overlap between consecutive chunks
- HUGGINGFACE_TOKEN: HuggingFace API token (optional)

Requirements: pip install -r requirements.txt
"""

import logging
import os
import pickle
import re
from pathlib import Path
from typing import Any, Dict, List

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_config():
    """Load configuration from environment variables"""
    config = {
        'cache_dir': Path(os.getenv('CACHE_DIR', '../cleaned')),
        'processed_dir': Path(os.getenv('PROCESSED_DIR', '../processed')),
        'chunk_size': int(os.getenv('CHUNK_SIZE', '180')),
        'chunk_overlap': int(os.getenv('CHUNK_OVERLAP', '80')),
        # local embedding (HuggingFace)
        'embed_model': os.getenv('EMBED_MODEL', ''),
        
        # remote embedding (OpenAI-style, e.g. https://.../v1)
        'embed_endpoint': os.getenv('EMBEDDING_SERVICE_ENDPOINT', ''),
        'embed_model_alias': os.getenv('EMBEDDING_API_KEY', 'kure'),
        'huggingface_token': os.getenv('HUGGINGFACE_TOKEN', ''),
    }
    
    # Create processed directory
    config['processed_dir'].mkdir(parents=True, exist_ok=True)
    
    return config

def initialize_embeddings(config):
    """Initialize embedding model

    - USE_LOCAL_EMBEDDINGS=true  → HuggingFaceEmbeddings (local)
    - USE_LOCAL_EMBEDDINGS=false → OpenAIEmbeddings (remote /v1/embeddings)
    """

    use_local = os.getenv("USE_LOCAL_EMBEDDINGS", "false").lower() == "true"

    # Set HuggingFace token if provided
    if use_local:
        os.environ['HF_TOKEN'] = config['huggingface_token']
        logger.info(f"Using HuggingFace embeddings: {config['embed_model']}")
        return HuggingFaceEmbeddings(
            model_name=config['embed_model']
        )
    # Set Backend.AI embedding service endpoint
    else:
        logger.info(f"Using Backend.AI Embedding Service at {config['embed_endpoint']}")
        logger.info(f"Embedding model alias: {config['embed_model_alias']}")
        # Remote embedding caller
        emb =  OpenAIEmbeddings(
            base_url=config['embed_endpoint'].rstrip("/"),       # ex) https://embedding-kure-v1.asia03.app.backend.ai/v1
            api_key="dummy-key",                     # token 필요 없음
            model=config['embed_model_alias']        # "kure"
        )
        # 디버그용
        try:
            from openai import OpenAI

            # langchain-openai 0.2.x 기준 내부 client 접근
            client = emb.client  # or emb._client depending on version
            logger.info(f"[DEBUG] OpenAIEmbeddings base_url = {client._client.base_url}")
        except Exception as e:
            logger.warning(f"[DEBUG] Could not inspect base_url: {e}")

        return emb


def build_documents_from_txt(txt_path: Path, chunk_size: int, chunk_overlap: int) -> List[Document]:
    """Convert text file to chunked documents"""
    text = txt_path.read_text(encoding="utf-8")
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    
    chunks = splitter.split_text(text)
    docs: List[Document] = []
    
    for chunk in chunks:
        # Extract metadata from source markers
        meta: Dict[str, Any] = {}
        content = chunk
        
        # Look for source markers like [SOURCE=native PAGE=1 FILE=filename.pdf]
        source_match = re.search(r"\[SOURCE=(\w+)\s+PAGE=(\d+)\s+FILE=([^\]]+)\]", chunk)
        if source_match:
            meta["extracted_by"] = source_match.group(1)
            meta["page"] = int(source_match.group(2))
            meta["source_file"] = source_match.group(3)
            # Remove the source marker from content
            content = chunk.replace(source_match.group(0), "").strip()
        else:
            # Fallback to filename if no source marker
            meta["source_file"] = txt_path.stem + ".pdf"
        
        if content.strip():  # Only add non-empty chunks
            docs.append(Document(page_content=content, metadata=meta))
    
    return docs

def main():
    """Main function to process documents and generate embeddings"""
    config = load_config()
    cache_dir = config['cache_dir']
    processed_dir = config['processed_dir']
    
    # Load manifest of text files
    manifest_path = cache_dir / "manifest.txt"
    if not manifest_path.exists():
        logger.error("No manifest.txt found. Run task1_data_ingestion.py first.")
        return
    
    # Read text file paths from manifest
    with open(manifest_path, 'r') as f:
        txt_paths = [Path(line.strip()) for line in f if line.strip()]
    
    logger.info(f"Processing {len(txt_paths)} text files")
    
    # Process all text files into documents
    all_docs = []
    for txt_path in txt_paths:
        if not txt_path.exists():
            logger.warning(f"Text file not found: {txt_path}")
            continue
        
        try:
            docs = build_documents_from_txt(
                txt_path, 
                config['chunk_size'], 
                config['chunk_overlap']
            )
            all_docs.extend(docs)
            logger.info(f"Processed {txt_path}: {len(docs)} chunks")
        except Exception as e:
            logger.error(f"Error processing {txt_path}: {e}")
    
    logger.info(f"Total documents: {len(all_docs)}")
    
    # Initialize embeddings
    try:
        embeddings = initialize_embeddings(config)
    except Exception as e:
        logger.error(f"Failed to initialize embeddings: {e}")
        return
    
    # Save processed documents
    docs_path = processed_dir / "documents.pkl"
    with open(docs_path, 'wb') as f:
        pickle.dump(all_docs, f)
    logger.info(f"Saved {len(all_docs)} documents to {docs_path}")
    
    # Generate and save embeddings for all documents
    logger.info("Generating embeddings...")
    try:
        texts = [doc.page_content for doc in all_docs]
        doc_embeddings = embeddings.embed_documents(texts)
        
        embeddings_path = processed_dir / "embeddings.pkl"
        with open(embeddings_path, 'wb') as f:
            pickle.dump(doc_embeddings, f)
        logger.info(f"Saved embeddings to {embeddings_path}")
        
        # Save embedding config for next task
        config_path = processed_dir / "embedding_config.pkl"
        embedding_config = {
            'embed_model': config['embed_model'],
            'embedding_dimension': len(doc_embeddings[0]) if doc_embeddings else 0
        }
        with open(config_path, 'wb') as f:
            pickle.dump(embedding_config, f)
        logger.info(f"Saved embedding config to {config_path}")
        
    except Exception as e:
        logger.error(f"Error generating embeddings: {e}")
        return

if __name__ == "__main__":
    main()