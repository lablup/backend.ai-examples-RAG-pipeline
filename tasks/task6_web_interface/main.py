#!/usr/bin/env python3
"""
Task 6: Web Interface for RAG Comparison
- Interactive Gradio web interface for real-time RAG evaluation
- Side-by-side comparison of RAG vs non-RAG responses
- Live query processing with hybrid retrieval and reranking
- Document source display with transparency
- Configurable system prompts and model parameters
- No Docker dependencies - runs as standalone Python application

Environment Variables:
- INDEX_DIR: Input directory with search indexes from Task 3
- MODEL_ENDPOINT: LLM API endpoint URL (required)
- MODEL_NAME: LLM model name (required)
- API_KEY: API key for LLM service
- TOKENIZER_MODEL: Tokenizer model for token counting (required)
- TEMPERATURE: LLM temperature for response generation
- MAX_TOKENS: Maximum tokens for response
- MODEL_CTX_LIMIT: Model context window limit
- TOP_K: Number of documents to retrieve initially
- FINAL_K: Number of final documents after reranking
- RERANK_POOL: Maximum documents to consider for reranking
- USE_RERANK: Enable/disable reranking (true/false)
- RERANK_MODEL: Cross-encoder reranking model name (required if USE_RERANK=true)
- PER_DOC_CAP: Maximum tokens per document in context
- SAFETY_MARGIN: Token counting safety margin
- RAG_SYSTEM_PROMPT: System prompt for RAG responses
- NON_RAG_SYSTEM_PROMPT: System prompt for non-RAG responses
- HUGGINGFACE_TOKEN: HuggingFace API token (optional)

Requirements: pip install -r requirements.txt
"""

import os
import pickle
import json
import re
from pathlib import Path
from typing import List, Optional, Tuple
from functools import lru_cache
import logging

import gradio as gr
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for loaded components
vectorstore = None
bm25_retriever = None
reranker = None
tokenizer = None
llm_client = None

def load_config():
    """Load configuration from environment variables"""
    return {
        'index_dir': Path(os.getenv('INDEX_DIR', '../indexes')),
        'model_endpoint': os.getenv('MODEL_ENDPOINT', ''),
        'model_name': os.getenv('MODEL_NAME', ''),
        'api_key': os.getenv('API_KEY', 'dummy-key'),
        'temperature': float(os.getenv('TEMPERATURE', '0.2')),
        'max_tokens': int(os.getenv('MAX_TOKENS', '3000')),
        'model_ctx_limit': int(os.getenv('MODEL_CTX_LIMIT', '4096')),
        'tokenizer_model': os.getenv('TOKENIZER_MODEL', ''),
        'top_k': int(os.getenv('TOP_K', '32')),
        'final_k': int(os.getenv('FINAL_K', '6')),
        'rerank_pool': int(os.getenv('RERANK_POOL', '40')),
        'use_rerank': os.getenv('USE_RERANK', 'true').lower() == 'true',
        'rerank_model': os.getenv('RERANK_MODEL', ''),
        'per_doc_cap': int(os.getenv('PER_DOC_CAP', '320')),
        'safety_margin': int(os.getenv('SAFETY_MARGIN', '256')),
        'huggingface_token': os.getenv('HUGGINGFACE_TOKEN', ''),
        'rag_system_prompt': os.getenv('RAG_SYSTEM_PROMPT', '당신은 Backend.AI 전문가이며, 근거 중심 어시스턴트입니다. 아래 컨텍스트의 근거에 한해서만 답변하세요. 근거가 부족하면 \'근거 부족\'이라고 말하세요. 마지막에 반드시 출처 파일명, 소제목을 함께 제공하세요. 순서가 필요하면 순서를 정확하게 언급하세요.'),
        'non_rag_system_prompt': os.getenv('NON_RAG_SYSTEM_PROMPT', '당신은 Backend.AI 전문 어시스턴트입니다. 간결하고 사실 기반으로 답변하세요.'),
    }

def initialize_tokenizer(tokenizer_model: str):
    """Initialize tokenizer for token counting"""
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_model, use_fast=False)
        logger.info(f"Tokenizer loaded: {tokenizer_model}")
        return tokenizer
    except Exception as e:
        logger.warning(f"Failed to load tokenizer, using fallback: {e}")
        return None

@lru_cache(maxsize=100000)
def _encode(text: str):
    """Cached tokenization"""
    if tokenizer is None:
        return list(range(max(1, len(text) // 4)))
    return tokenizer.encode(text, add_special_tokens=False)

def count_tokens(text: str) -> int:
    """Count tokens in text"""
    return len(_encode(text))

def trim_text_to_tokens(text: str, max_tokens: int) -> str:
    """Trim text to maximum token count"""
    ids = _encode(text)
    if len(ids) <= max_tokens:
        return text
    
    if tokenizer is None:
        return text[:max_tokens * 4]
    
    return tokenizer.decode(ids[:max_tokens], skip_special_tokens=True)

def dedupe_spaces(text: str) -> str:
    """Clean up extra spaces and newlines"""
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def load_indexes(config):
    """Load pre-built indexes and models"""
    global vectorstore, bm25_retriever, reranker, tokenizer, llm_client
    
    # Load index metadata
    index_dir = config['index_dir']
    metadata_path = index_dir / "index_metadata.pkl"
    
    if not metadata_path.exists():
        raise FileNotFoundError(f"Index metadata not found at {metadata_path}. Run tasks 1-3 first.")
    
    with open(metadata_path, 'rb') as f:
        index_metadata = pickle.load(f)
    
    # Initialize embedding model
    embedding_config = index_metadata['embedding_config']
    if config['huggingface_token']:
        os.environ['HF_TOKEN'] = config['huggingface_token']
    embeddings = HuggingFaceEmbeddings(model_name=embedding_config['embed_model'])
    
    # Load FAISS index
    faiss_path = Path(index_metadata['faiss_path'])
    vectorstore = FAISS.load_local(str(faiss_path), embeddings, allow_dangerous_deserialization=True)
    
    # Load BM25 index
    bm25_path = Path(index_metadata['bm25_path'])
    with open(bm25_path, 'rb') as f:
        bm25_retriever = pickle.load(f)
    
    # Initialize reranker
    if config['use_rerank']:
        try:
            from sentence_transformers import CrossEncoder
            reranker = CrossEncoder(config['rerank_model'])
        except Exception as e:
            logger.warning(f"Failed to load reranker: {e}")
            reranker = None
    
    # Initialize tokenizer
    tokenizer = initialize_tokenizer(config['tokenizer_model'])
    
    # Initialize LLM client
    llm_client = ChatOpenAI(
        model=config['model_name'],
        openai_api_base=config['model_endpoint'],
        openai_api_key=config['api_key'],
        temperature=config['temperature'],
        max_tokens=config['max_tokens'],
        timeout=120,
    )
    
    logger.info("All components loaded successfully")

def retrieve_documents(query: str, config) -> List[Document]:
    """Retrieve relevant documents using hybrid approach"""
    # BM25 retrieval
    bm25_retriever.k = min(config['top_k'], config['rerank_pool'])
    bm25_docs = bm25_retriever.invoke(query)
    
    # Vector retrieval
    vector_docs = vectorstore.as_retriever(
        search_kwargs={"k": min(config['top_k'], config['rerank_pool'])}
    ).invoke(query)
    
    # Combine with reciprocal rank fusion
    from collections import defaultdict
    scores = defaultdict(float)
    
    for rank, doc in enumerate(bm25_docs):
        scores[id(doc)] += 0.5 / (rank + 1)
    
    for rank, doc in enumerate(vector_docs):
        scores[id(doc)] += 0.5 / (rank + 1)
    
    # Deduplicate and sort
    unique_docs = {id(doc): doc for doc in bm25_docs + vector_docs}
    sorted_docs = sorted(unique_docs.values(), key=lambda d: scores[id(d)], reverse=True)
    
    # Apply reranking if available
    if reranker and sorted_docs:
        candidates = sorted_docs[:config['rerank_pool']]
        pairs = [(query, doc.page_content) for doc in candidates]
        rerank_scores = reranker.predict(pairs)
        reranked = sorted(zip(candidates, rerank_scores), key=lambda x: x[1], reverse=True)
        final_docs = [doc for doc, score in reranked[:config['final_k']]]
    else:
        final_docs = sorted_docs[:config['final_k']]
    
    return final_docs

def pack_context(docs: List[Document], config) -> str:
    """Pack document context for RAG response"""
    blocks = []
    for doc in docs:
        head = f"[{doc.metadata.get('source_file','?')}#p{doc.metadata.get('page','?')}]\n"
        body = dedupe_spaces(doc.page_content)
        head_tokens = count_tokens(head)
        remain_for_body = max(8, config['per_doc_cap'] - head_tokens)
        body = trim_text_to_tokens(body, remain_for_body)
        blocks.append(head + body)
    
    # Binary search to fit within token budget
    system_prompt = config['rag_system_prompt']
    user_prefix = f"질문: [QUERY]\n\n[컨텍스트]\n"
    
    hard_cap = max(512, config['model_ctx_limit'] - config['max_tokens'] - config['safety_margin'])
    
    used_text = ""
    for block in blocks:
        candidate = f"{user_prefix}{(used_text + '\n\n') if used_text else ''}{block}"
        total_tokens = count_tokens(system_prompt) + count_tokens(candidate)
        
        if total_tokens <= hard_cap:
            used_text = f"{used_text}\n\n{block}" if used_text else block
        else:
            break
    
    return used_text

def generate_responses(query: str, config) -> Tuple[str, str, str, int]:
    """Generate both RAG and non-RAG responses"""
    if not query.strip():
        return "질문을 입력해주세요.", "질문을 입력해주세요.", "질문이 없습니다.", 0
    
    
    try:
        # Retrieve documents for RAG
        context_docs = retrieve_documents(query, config)
        
        # Generate RAG response
        context = pack_context(context_docs, config)
        rag_system_prompt = config['rag_system_prompt']
        rag_user_prompt = f"질문: {query}\n\n[컨텍스트]\n{context}\n\n규칙: 사실 확인 가능한 문장만 작성."
        
        rag_messages = [
            {"role": "system", "content": rag_system_prompt},
            {"role": "user", "content": rag_user_prompt},
        ]
        
        rag_response = llm_client.invoke(rag_messages).content
        
        # Generate non-RAG response
        non_rag_system_prompt = config['non_rag_system_prompt']
        non_rag_user_prompt = f"질문: {query}"
        
        non_rag_messages = [
            {"role": "system", "content": non_rag_system_prompt},
            {"role": "user", "content": non_rag_user_prompt},
        ]
        
        non_rag_response = llm_client.invoke(non_rag_messages).content
        
        # Prepare sources info
        sources_info = "\n".join([
            f"• {doc.metadata.get('source_file', '?')}#p{doc.metadata.get('page', '?')}"
            for doc in context_docs
        ])
        
        return rag_response, non_rag_response, sources_info, len(context_docs)
        
    except Exception as e:
        error_msg = f"오류가 발생했습니다: {str(e)}"
        return error_msg, error_msg, "오류로 인해 소스를 불러올 수 없습니다.", 0

def create_interface():
    """Create Gradio interface"""
    config = load_config()
    
    # Load components at startup
    try:
        load_indexes(config)
        status_message = "✅ 모든 구성요소가 성공적으로 로드되었습니다."
    except Exception as e:
        status_message = f"❌ 로드 실패: {str(e)}\n작업 1-3을 먼저 실행하여 인덱스를 생성하세요."
    
    # Define the main function for Gradio
    def compare_responses(query: str):
        return generate_responses(query, config)
    
    # Create Gradio interface
    with gr.Blocks(title="Korean RAG Comparison", theme=gr.themes.Soft()) as demo:
        gr.HTML("<h1 style='text-align: center;'>🔍 Korean RAG vs Non-RAG Comparison</h1>")
        gr.HTML(f"<p style='text-align: center;'>{status_message}</p>")
        
        with gr.Row():
            query_input = gr.Textbox(
                label="질문을 입력하세요",
                placeholder="예: Backend.AI에서 모델 서비스를 생성하는 방법은?",
                lines=3,
                max_lines=5
            )
        
        submit_btn = gr.Button("답변 생성", variant="primary", size="lg")
        
        with gr.Row(equal_height=True):
            with gr.Column():
                gr.HTML("<h3>🔍 RAG Response (문서 기반)</h3>")
                rag_output = gr.Textbox(
                    label="RAG 응답",
                    lines=15,
                    max_lines=20,
                    show_copy_button=True
                )
            
            with gr.Column():
                gr.HTML("<h3>❌ Non-RAG Response (학습 데이터만)</h3>")
                non_rag_output = gr.Textbox(
                    label="Non-RAG 응답",
                    lines=15,
                    max_lines=20,
                    show_copy_button=True
                )
        
        with gr.Row():
            sources_output = gr.Textbox(
                label="사용된 문서 소스",
                lines=5,
                max_lines=10
            )
            
            doc_count = gr.Number(
                label="검색된 문서 수",
                precision=0
            )
        
        # Examples
        gr.Examples(
            examples=[
                ["Backend.AI에서 제한된 사용자만 모델 서비스에 접근하도록 설정하는 방법은?"],
                ["Backend.AI에서 모델 서빙을 위한 토큰 발급 절차는?"],
                ["Backend.AI Control Panel에서 사용자 권한을 관리하는 방법은?"],
            ],
            inputs=[query_input]
        )
        
        # Event handlers
        submit_btn.click(
            fn=compare_responses,
            inputs=[query_input],
            outputs=[rag_output, non_rag_output, sources_output, doc_count]
        )
        
        query_input.submit(
            fn=compare_responses,
            inputs=[query_input],
            outputs=[rag_output, non_rag_output, sources_output, doc_count]
        )
    
    return demo

def main():
    """Main function to launch the web interface"""
    demo = create_interface()
    
    # Launch the interface
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        inbrowser=False,
        show_error=True,
    )

if __name__ == "__main__":
    main()