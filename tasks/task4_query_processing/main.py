#!/usr/bin/env python3
"""
Task 4: Query Processing
- Load FAISS and BM25 indexes from Task 3
- Implement hybrid retrieval combining semantic and keyword search
- Apply optional cross-encoder reranking for improved relevance
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

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_config():
    """Load configuration from environment variables"""
    config = {
        'index_dir': Path(os.getenv('INDEX_DIR', '../indexes')),
        'query_dir': Path(os.getenv('QUERY_DIR', '../query_results')),
        'top_k': int(os.getenv('TOP_K', '32')),
        'final_k': int(os.getenv('FINAL_K', '6')),
        'rerank_pool': int(os.getenv('RERANK_POOL', '40')),
        'use_rerank': os.getenv('USE_RERANK', 'true').lower() == 'true',
        'rerank_model': os.getenv('RERANK_MODEL', ''),
        'query': os.getenv('QUERY', ''),
        'huggingface_token': os.getenv('HUGGINGFACE_TOKEN', ''),
    }
    
    # Create query results directory
    config['query_dir'].mkdir(parents=True, exist_ok=True)
    
    return config

def load_indexes(index_dir: Path, config):
    """Load FAISS and BM25 indexes"""
    # Load index metadata
    metadata_path = index_dir / "index_metadata.pkl"
    if not metadata_path.exists():
        raise FileNotFoundError("index_metadata.pkl not found. Run task3_index_building.py first.")
    
    with open(metadata_path, 'rb') as f:
        index_metadata = pickle.load(f)
    
    # Initialize embedding model
    embedding_config = index_metadata['embedding_config']
    if config['huggingface_token']:
        os.environ['HF_TOKEN'] = config['huggingface_token']
    
    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_config['embed_model']
    )
    
    # Load FAISS index
    faiss_path = Path(index_metadata['faiss_path'])
    if not faiss_path.exists():
        raise FileNotFoundError(f"FAISS index not found at {faiss_path}")
    
    vectorstore = FAISS.load_local(str(faiss_path), embeddings, allow_dangerous_deserialization=True)
    logger.info("FAISS index loaded successfully")
    
    # Load BM25 index
    bm25_path = Path(index_metadata['bm25_path'])
    if not bm25_path.exists():
        raise FileNotFoundError(f"BM25 index not found at {bm25_path}")
    
    with open(bm25_path, 'rb') as f:
        bm25_retriever = pickle.load(f)
    logger.info("BM25 index loaded successfully")
    
    return vectorstore, bm25_retriever, index_metadata

def _retrieve_docs(retriever, query: str, k: int):
    """Helper function to retrieve documents with proper k setting"""
    if hasattr(retriever, "k"):
        retriever.k = k
    
    if hasattr(retriever, "invoke"):
        docs = retriever.invoke(query)
    else:
        docs = retriever.get_relevant_documents(query)
    
    return docs or []

class HybridRetriever:
    """Hybrid retriever combining BM25 and FAISS"""
    
    def __init__(self, bm25_retriever, vector_retriever, weights=(0.5, 0.5), k=32, pool_k=40):
        self.bm25 = bm25_retriever
        self.vector = vector_retriever
        self.w_bm25, self.w_vector = weights
        self.k = k
        self.pool_k = pool_k
    
    def get_relevant_documents(self, query: str) -> List[Document]:
        """Retrieve documents using hybrid approach"""
        # Get candidates from both retrievers
        bm25_docs = _retrieve_docs(self.bm25, query, min(self.pool_k, self.k))
        vector_docs = _retrieve_docs(self.vector, query, min(self.pool_k, self.pool_k))
        
        # Calculate reciprocal rank scores
        scores = defaultdict(float)
        
        for rank, doc in enumerate(bm25_docs):
            scores[id(doc)] += self.w_bm25 / (rank + 1)
        
        for rank, doc in enumerate(vector_docs):
            scores[id(doc)] += self.w_vector / (rank + 1)
        
        # Combine and deduplicate
        unique_docs = {id(doc): doc for doc in bm25_docs + vector_docs}
        
        # Sort by combined scores
        sorted_docs = sorted(
            unique_docs.values(), 
            key=lambda d: scores[id(d)], 
            reverse=True
        )
        
        return sorted_docs[:self.pool_k]
    
    def invoke(self, query: str) -> List[Document]:
        """LangChain-style invoke method"""
        return self.get_relevant_documents(query)

def initialize_reranker(rerank_model: str):
    """Initialize cross-encoder reranker"""
    try:
        from sentence_transformers import CrossEncoder
        reranker = CrossEncoder(rerank_model)
        logger.info(f"Reranker loaded: {rerank_model}")
        return reranker
    except Exception as e:
        logger.error(f"Failed to load reranker: {e}")
        return None

def rerank_documents(query: str, documents: List[Document], reranker, top_k: int) -> List[Document]:
    """Rerank documents using cross-encoder"""
    if not reranker or not documents:
        return documents[:top_k]
    
    # Create query-document pairs
    pairs = [(query, doc.page_content) for doc in documents]
    
    # Get reranking scores
    scores = reranker.predict(pairs)
    
    # Sort by scores
    ranked_docs = sorted(
        zip(documents, scores), 
        key=lambda x: x[1], 
        reverse=True
    )
    
    return [doc for doc, score in ranked_docs[:top_k]]

def main():
    """Main function to process query and retrieve context"""
    config = load_config()
    
    if not config['query']:
        logger.error("No query provided. Set QUERY environment variable.")
        return
    
    query = config['query']
    logger.info(f"Processing query: {query}")
    
    # Load indexes
    try:
        vectorstore, bm25_retriever, index_metadata = load_indexes(config['index_dir'], config)
    except Exception as e:
        logger.error(f"Failed to load indexes: {e}")
        return
    
    # Create hybrid retriever
    vector_retriever = vectorstore.as_retriever(
        search_kwargs={"k": min(config['top_k'], config['rerank_pool'])}
    )
    
    hybrid_retriever = HybridRetriever(
        bm25_retriever=bm25_retriever,
        vector_retriever=vector_retriever,
        weights=(0.5, 0.5),
        k=config['top_k'],
        pool_k=config['rerank_pool']
    )
    
    # Retrieve candidate documents
    logger.info("Retrieving candidate documents...")
    candidate_docs = hybrid_retriever.get_relevant_documents(query)
    logger.info(f"Retrieved {len(candidate_docs)} candidate documents")
    
    # Apply reranking if enabled
    if config['use_rerank']:
        logger.info("Applying reranking...")
        reranker = initialize_reranker(config['rerank_model'])
        if reranker:
            final_docs = rerank_documents(query, candidate_docs, reranker, config['final_k'])
            logger.info(f"Reranked to {len(final_docs)} final documents")
        else:
            final_docs = candidate_docs[:config['final_k']]
    else:
        final_docs = candidate_docs[:config['final_k']]
    
    # Prepare results
    results = {
        'query': query,
        'num_candidates': len(candidate_docs),
        'num_final': len(final_docs),
        'config': {
            'top_k': config['top_k'],
            'final_k': config['final_k'],
            'rerank_pool': config['rerank_pool'],
            'use_rerank': config['use_rerank'],
            'rerank_model': config['rerank_model']
        },
        'documents': []
    }
    
    # Convert documents to serializable format
    for i, doc in enumerate(final_docs):
        doc_data = {
            'rank': i + 1,
            'content': doc.page_content,
            'metadata': dict(doc.metadata),
            'preview': doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
        }
        results['documents'].append(doc_data)
    
    # Save results
    results_path = config['query_dir'] / "query_results.json"
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logger.info(f"Query results saved to {results_path}")
    
    # Save documents for next task (pickle format for easy loading)
    docs_path = config['query_dir'] / "retrieved_documents.pkl"
    with open(docs_path, 'wb') as f:
        pickle.dump(final_docs, f)
    logger.info(f"Retrieved documents saved to {docs_path}")
    
    # Print summary
    logger.info(f"Query processing completed:")
    logger.info(f"  - Query: {query}")
    logger.info(f"  - Candidates retrieved: {len(candidate_docs)}")
    logger.info(f"  - Final documents: {len(final_docs)}")
    logger.info(f"  - Reranking: {'enabled' if config['use_rerank'] else 'disabled'}")

if __name__ == "__main__":
    main()