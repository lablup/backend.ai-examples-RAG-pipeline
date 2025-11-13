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
- USE_LOCAL_EMBEDDINGS: "true"/"false" to control local vs remote embeddings
- EMBED_MODEL: HuggingFace embedding model name (for local mode)
- EMBEDDING_SERVICE_ENDPOINT: Base URL of OpenAI-style embedding API (for remote mode)
- EMBED_MODEL_ALIAS: Model name for OpenAI-style API (for remote mode)

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
from langchain_openai import OpenAIEmbeddings  # ← 추가

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config():
    """Load configuration from environment variables"""
    config = {
        'processed_dir': Path(os.getenv('PROCESSED_DIR', '../processed')),
        'index_dir': Path(os.getenv('INDEX_DIR', '../indexes')),
        'huggingface_token': os.getenv('HUGGINGFACE_TOKEN', ''),
        # local embedding (HuggingFace)
        'embed_model': os.getenv('EMBED_MODEL', ''),
        # remote embedding (OpenAI-style, e.g. https://.../v1)
        'embed_endpoint': os.getenv('EMBEDDING_SERVICE_ENDPOINT', '').rstrip('/'),
        'embed_model_alias': os.getenv('EMBED_MODEL_ALIAS', 'kure'),
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
    """
    Initialize the same embedding model used in processing.

    - USE_LOCAL_EMBEDDINGS=true  → HuggingFaceEmbeddings (local)
    - USE_LOCAL_EMBEDDINGS=false → OpenAIEmbeddings (remote /v1/embeddings)
    """

    use_local = os.getenv("USE_LOCAL_EMBEDDINGS", "false").lower() == "true"

    if use_local:
        # 1) 로컬 HuggingFaceEmbeddings 사용
        if config['huggingface_token']:
            os.environ['HF_TOKEN'] = config['huggingface_token']

        # Task 2에서 저장한 값 우선, 없으면 ENV 값 사용
        model_name = embedding_config.get('embed_model') or config['embed_model']
        if not model_name:
            raise ValueError("USE_LOCAL_EMBEDDINGS=true 인데 EMBED_MODEL 또는 embedding_config['embed_model']가 없습니다.")

        logger.info(f"Using local HuggingFaceEmbeddings: {model_name}")

        return HuggingFaceEmbeddings(
            model_name=model_name
        )

    else:
        # 2) 원격 OpenAI 스타일 Embeddings 사용
        # Task 2에서 저장한 값 우선, 없으면 ENV 값 사용
        endpoint = embedding_config.get('embed_endpoint') or config['embed_endpoint']
        model_alias = embedding_config.get('embed_model_alias') or config['embed_model_alias']

        if not endpoint:
            raise ValueError(
                "USE_LOCAL_EMBEDDINGS=false 인데 EMBEDDING_SERVICE_ENDPOINT 또는 embedding_config['embed_endpoint']가 없습니다."
            )

        endpoint = endpoint.rstrip('/')
        logger.info(f"Using remote embedding endpoint: {endpoint}/v1/embeddings")
        logger.info(f"Embedding model alias: {model_alias}")

        emb = OpenAIEmbeddings(
            base_url=endpoint,      # 예: https://embedding-kure-v1.asia03.app.backend.ai/v1
            api_key="dummy-key",    # 토큰 필요 없음
            model=model_alias       # 예: "kure"
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


def build_faiss_index(documents, doc_embeddings, embeddings, index_dir: Path):
    """Build and save FAISS vector index"""
    logger.info("Building FAISS index...")

    try:
        # pre-computed embeddings → numpy array
        embeddings_array = np.array(doc_embeddings, dtype=np.float32)

        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]

        # FAISS.from_embeddings: (text, embedding) 쌍과 embedding function 넣기
        vectorstore = FAISS.from_embeddings(
            text_embeddings=list(zip(texts, embeddings_array)),
            embedding=embeddings,
            metadatas=metadatas,
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
        bm25_retriever = BM25Retriever.from_documents(documents)

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

    # Initialize embedding model (local or remote)
    try:
        embeddings = initialize_embedding_model(embedding_config, config)
        logger.info("Initialized embedding model (local/remote) successfully")
    except Exception as e:
        logger.error(f"Failed to initialize embedding model: {e}")
        return

    # Build FAISS index
    try:
        _ = build_faiss_index(documents, doc_embeddings, embeddings, index_dir)
    except Exception as e:
        logger.error(f"Failed to build FAISS index: {e}")
        return

    # Build BM25 index
    try:
        _ = build_bm25_index(documents, index_dir)
    except Exception as e:
        logger.error(f"Failed to build BM25 index: {e}")
        return

    # Save index metadata (embedding_config 그대로 저장)
    index_metadata = {
        'num_documents': len(documents),
        'embedding_config': embedding_config,
        'faiss_path': str(index_dir / "faiss"),
        'bm25_path': str(index_dir / "bm25.pkl"),
    }

    metadata_path = index_dir / "index_metadata.pkl"
    with open(metadata_path, 'wb') as f:
        pickle.dump(index_metadata, f)
    logger.info(f"Index metadata saved to {metadata_path}")

    logger.info("Index building completed successfully")


if __name__ == "__main__":
    main()
