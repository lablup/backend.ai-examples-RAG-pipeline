#!/usr/bin/env python3
"""
Task 4: Query Processing
- Load FAISS and BM25 indexes from Task 3
- Implement hybrid retrieval combining semantic and keyword search
- Apply optional cross-encoder reranking for improved relevance
- Optional rerank: local CrossEncoder OR remote API (/v1/rerank)
- Save retrieved context and sources for response generation

Environment Variables:
- INDEX_DIR: Input directory with search indexes
- QUERY_DIR: Output directory for query results
- QUERY: Query string to process (required)
- TOP_K: Number of documents to retrieve initially
- FINAL_K: Number of final documents after reranking
- RERANK_POOL: Maximum documents to consider for reranking
- USE_RERANK: Enable/disable reranking (true/false)
- RERANK_MODEL: Cross-encoder reranking model name (required if USE_RERANK=true)
- HUGGINGFACE_TOKEN: HuggingFace API token (optional)

Requirements: pip install -r requirements.txt
"""

import json
import logging
import os
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import requests
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import (
    OpenAIEmbeddings,  # for vectorstore loading with remote embedding if needed
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================
# 1. Config
# ============================================================

def load_config():
    """Load environment variables"""
    config = {
        'index_dir': Path(os.getenv('INDEX_DIR', '../indexes')),
        'query_dir': Path(os.getenv('QUERY_DIR', '../query_results')),
        'top_k': int(os.getenv('TOP_K', '32')),
        'final_k': int(os.getenv('FINAL_K', '6')),
        'rerank_pool': int(os.getenv('RERANK_POOL', '40')),

        # Rerank settings
        'use_rerank': os.getenv('USE_RERANK', 'true').lower() == 'true',
        'use_local_rerank': os.getenv('USE_LOCAL_RERANK', 'false').lower() == 'true',
        'rerank_model': os.getenv('RERANK_MODEL', ''),
        'rerank_endpoint': os.getenv('RERANK_SERVICE_ENDPOINT', '').rstrip('/'),

        # Query
        'query': os.getenv('QUERY', ''),
    }

    config['query_dir'].mkdir(parents=True, exist_ok=True)
    return config


# ============================================================
# 2. Load Indexes (FAISS + BM25)
# ============================================================

def load_indexes(index_dir: Path):
    metadata_path = index_dir / "index_metadata.pkl"
    if not metadata_path.exists():
        raise FileNotFoundError("index_metadata.pkl not found.")

    with open(metadata_path, 'rb') as f:
        index_metadata = pickle.load(f)

    embedding_config = index_metadata['embedding_config']

    # -------------------------------
    # Load Embedding Model
    # (Needed only for FAISS vectorstore load)
    # -------------------------------
    if 'embed_model' in embedding_config:
        # Local HuggingFace embedding
        embeddings = HuggingFaceEmbeddings(
            model_name=embedding_config['embed_model']
        )
    else:
        # Remote embedding service (OpenAI Embedding API)
        embeddings = OpenAIEmbeddings(
            base_url=embedding_config['embed_endpoint'].rstrip("/"),
            api_key="dummy-key",
            model=embedding_config['embed_model_alias']
        )

    # -------------------------------
    # Load FAISS
    # -------------------------------
    faiss_path = Path(index_metadata['faiss_path'])
    vectorstore = FAISS.load_local(
        str(faiss_path),
        embeddings,
        allow_dangerous_deserialization=True
    )
    logger.info("FAISS index loaded successfully")

    # -------------------------------
    # Load BM25
    # -------------------------------
    bm25_path = Path(index_metadata['bm25_path'])
    with open(bm25_path, 'rb') as f:
        bm25_retriever = pickle.load(f)
    logger.info("BM25 index loaded successfully")

    return vectorstore, bm25_retriever, index_metadata


# ============================================================
# 3. Hybrid Retriever (BM25 + FAISS)
# ============================================================

def _retrieve_docs(retriever, query: str, k: int):
    if hasattr(retriever, "k"):
        retriever.k = k

    if hasattr(retriever, "invoke"):
        return retriever.invoke(query)
    return retriever.get_relevant_documents(query) or []


class HybridRetriever:
    def __init__(self, bm25_retriever, vector_retriever, weights=(0.5, 0.5), k=32, pool_k=40):
        self.bm25 = bm25_retriever
        self.vector = vector_retriever
        self.w_bm25, self.w_vector = weights
        self.k = k
        self.pool_k = pool_k

    def get_relevant_documents(self, query: str) -> List[Document]:
        bm25_docs = _retrieve_docs(self.bm25, query, min(self.pool_k, self.k))
        vector_docs = _retrieve_docs(self.vector, query, min(self.pool_k, self.pool_k))

        scores = defaultdict(float)

        for rank, doc in enumerate(bm25_docs):
            scores[id(doc)] += self.w_bm25 / (rank + 1)

        for rank, doc in enumerate(vector_docs):
            scores[id(doc)] += self.w_vector / (rank + 1)

        unique_docs = {id(doc): doc for doc in bm25_docs + vector_docs}
        sorted_docs = sorted(unique_docs.values(), key=lambda d: scores[id(d)], reverse=True)

        return sorted_docs[:self.pool_k]


# ============================================================
# 4. Rerank (Local / Remote)
# ============================================================

def initialize_local_reranker(model_name: str):
    try:
        from FlagEmbedding import FlagReranker
        logger.info(f"Loading local reranker: {model_name}")
        return FlagReranker(model_name, use_fp16=False, device="cpu")
    except Exception as e:
        logger.error(f"Failed to load local reranker: {e}")
        return None


def remote_rerank_call(endpoint: str, query: str, docs: List[Document], top_k: int):
    """
    Calls remote rerank API: POST /v1/rerank
    """
    payload = {
        "query": query,
        "documents": [d.page_content for d in docs],
        "top_n": top_k,
    }

    try:
        r = requests.post(endpoint, json=payload, timeout=120)
        r.raise_for_status()
        data = r.json()

        # API returns ranked: [{"index": idx, "relevance_score": x}, ...]
        ranked_indices = [item["index"] for item in data.get("data", [])]

        return [docs[i] for i in ranked_indices]

    except Exception as e:
        logger.error(f"Remote rerank API call failed: {e}")
        return docs[:top_k]


def local_rerank(query: str, documents: List[Document], reranker, top_k: int):
    if not reranker:
        return documents[:top_k]
    try:
        pairs = [[query, d.page_content] for d in documents]
        scores = reranker.compute_score(pairs)
        ranked = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in ranked[:top_k]]
    except:
        return documents[:top_k]


# ============================================================
# 5. Main
# ============================================================

def main():
    config = load_config()

    if not config['query']:
        logger.error("QUERY is required.")
        return

    query = config['query']
    logger.info(f"Processing query: {query}")

    vectorstore, bm25_retriever, index_metadata = load_indexes(config['index_dir'])

    vector_retriever = vectorstore.as_retriever(
        search_kwargs={"k": min(config['top_k'], config['rerank_pool'])}
    )

    hybrid_retriever = HybridRetriever(
        bm25_retriever=bm25_retriever,
        vector_retriever=vector_retriever,
        weights=(0.5, 0.5),
        k=config['top_k'],
        pool_k=config['rerank_pool'],
    )

    # Step 1: Hybrid Retrieval
    candidate_docs = hybrid_retriever.get_relevant_documents(query)
    logger.info(f"Retrieved {len(candidate_docs)} candidates")

    # Step 2: Rerank (local OR remote)
    if config['use_rerank']:
        logger.info("Reranking enabled")

        if config['use_local_rerank']:
            reranker = initialize_local_reranker(config['rerank_model'])
            final_docs = local_rerank(query, candidate_docs, reranker, config['final_k'])
        else:
            final_docs = remote_rerank_call(
                endpoint=config['rerank_endpoint'],
                query=query,
                docs=candidate_docs,
                top_k=config['final_k']
            )
    else:
        final_docs = candidate_docs[:config['final_k']]

    # Save Results
    results = {
        "query": query,
        "final_docs": len(final_docs),
        "candidate_docs": len(candidate_docs),
        "documents": [
            {
                "rank": i + 1,
                "content": d.page_content,
                "metadata": d.metadata
            }
            for i, d in enumerate(final_docs)
        ],
    }

    results_path = config['query_dir'] / "query_results.json"
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    logger.info(f"Saved results to {results_path}")

    # ✅ Task5 를 위한 pickle 저장 (retrieved_documents.pkl)
    docs_path = config['query_dir'] / "retrieved_documents.pkl"
    with open(docs_path, 'wb') as f:
        pickle.dump(final_docs, f)
    logger.info(f"Retrieved documents saved to {docs_path}")


if __name__ == "__main__":
    main()