#!/usr/bin/env python3
"""
Task 5: Response Generation
- Load retrieved documents and context from Task 4
- Generate RAG and non-RAG responses using configurable LLM endpoint
- Apply smart context packing with precise token counting
- Compare responses side-by-side with detailed analysis
- Save results with comprehensive metrics and metadata

Environment Variables:
- QUERY_DIR: Input directory with query results
- RESPONSE_DIR: Output directory for generated responses
- MODEL_ENDPOINT: LLM API endpoint URL (required)
- MODEL_NAME: LLM model name (required) 
- API_KEY: API key for LLM service
- TOKENIZER_MODEL: Tokenizer model for token counting (required)
- TEMPERATURE: LLM temperature for response generation
- MAX_TOKENS: Maximum tokens for response
- MODEL_CTX_LIMIT: Model context window limit
- PER_DOC_CAP: Maximum tokens per document in context
- SAFETY_MARGIN: Token counting safety margin
- RAG_SYSTEM_PROMPT: System prompt for RAG responses
- NON_RAG_SYSTEM_PROMPT: System prompt for non-RAG responses
- QUERY: Query string (optional, can be loaded from Task 4)
- HUGGINGFACE_TOKEN: HuggingFace API token (optional)

Requirements: pip install -r requirements.txt
"""

import os
import pickle
import json
import re
from pathlib import Path
from typing import List, Optional
from functools import lru_cache
import logging

from langchain_openai import ChatOpenAI
from langchain_core.documents import Document

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_config():
    """Load configuration from environment variables"""
    config = {
        'query_dir': Path(os.getenv('QUERY_DIR', '../query_results')),
        'response_dir': Path(os.getenv('RESPONSE_DIR', '../responses')),
        'model_endpoint': os.getenv('MODEL_ENDPOINT', ''),
        'model_name': os.getenv('MODEL_NAME', ''),
        'api_key': os.getenv('API_KEY', 'dummy-key'),
        'temperature': float(os.getenv('TEMPERATURE', '0.2')),
        'max_tokens': int(os.getenv('MAX_TOKENS', '3000')),
        'model_ctx_limit': int(os.getenv('MODEL_CTX_LIMIT', '4096')),
        'tokenizer_model': os.getenv('TOKENIZER_MODEL', ''),
        'per_doc_cap': int(os.getenv('PER_DOC_CAP', '320')),
        'safety_margin': int(os.getenv('SAFETY_MARGIN', '256')),
        'query': os.getenv('QUERY', ''),
        'huggingface_token': os.getenv('HUGGINGFACE_TOKEN', ''),
        'rag_system_prompt': os.getenv('RAG_SYSTEM_PROMPT', '당신은 Backend.AI 전문가이며, 근거 중심 어시스턴트입니다. 아래 컨텍스트의 근거에 한해서만 답변하세요. 근거가 부족하면 \'근거 부족\'이라고 말하세요. 마지막에 반드시 출처 파일명, 소제목을 함께 제공하세요. 순서가 필요하면 순서를 정확하게 언급하세요.'),
        'non_rag_system_prompt': os.getenv('NON_RAG_SYSTEM_PROMPT', '당신은 Backend.AI 전문 어시스턴트입니다. 간결하고 사실 기반으로 답변하세요.'),
    }
    
    # Create response directory
    config['response_dir'].mkdir(parents=True, exist_ok=True)
    
    return config

def initialize_tokenizer(tokenizer_model: str):
    """Initialize tokenizer for token counting"""
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_model, use_fast=False)
        logger.info(f"Tokenizer loaded: {tokenizer_model}")
        return tokenizer
    except Exception as e:
        logger.error(f"Failed to load tokenizer: {e}")
        return None

# Global tokenizer instance for caching
_tokenizer = None

def get_tokenizer(tokenizer_model: str):
    """Get or initialize tokenizer"""
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = initialize_tokenizer(tokenizer_model)
    return _tokenizer

@lru_cache(maxsize=100000)
def _encode(text: str, tokenizer):
    """Cached tokenization"""
    if tokenizer is None:
        # Fallback approximation
        return list(range(max(1, len(text) // 4)))
    return tokenizer.encode(text, add_special_tokens=False)

def count_tokens(text: str, tokenizer) -> int:
    """Count tokens in text"""
    return len(_encode(text, tokenizer))

def trim_text_to_tokens(text: str, max_tokens: int, tokenizer) -> str:
    """Trim text to maximum token count"""
    ids = _encode(text, tokenizer)
    if len(ids) <= max_tokens:
        return text
    
    if tokenizer is None:
        # Fallback: approximate character-based trimming
        return text[:max_tokens * 4]
    
    return tokenizer.decode(ids[:max_tokens], skip_special_tokens=True)

def dedupe_spaces(text: str) -> str:
    """Clean up extra spaces and newlines"""
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def pack_context(docs: List[Document], token_budget: int, per_doc_cap: int, tokenizer) -> str:
    """Pack document context within token budget"""
    used_text = ""
    
    for doc in docs:
        # Create document header
        head = f"[{doc.metadata.get('source_file', '?')}#p{doc.metadata.get('page', '?')}]\n"
        head_tokens = count_tokens(head, tokenizer)
        
        # Calculate remaining tokens for content
        remain_for_body = max(8, per_doc_cap - head_tokens)
        
        # Clean and trim document content
        body = dedupe_spaces(doc.page_content)
        body = trim_text_to_tokens(body, remain_for_body, tokenizer)
        
        block = head + body
        
        # Check if adding this block exceeds budget
        candidate = ((used_text + "\n\n") if used_text else "") + block
        if count_tokens(candidate, tokenizer) > token_budget:
            # Try to fit partial block
            if used_text:
                remaining_budget = max(16, token_budget - count_tokens(used_text, tokenizer))
                trimmed_block = trim_text_to_tokens(block, remaining_budget, tokenizer)
                if trimmed_block.strip():
                    used_text = used_text + "\n\n" + trimmed_block
            break
        
        used_text = candidate
    
    return used_text

def _message_token_count(system_prompt: str, user_content: str, tokenizer) -> int:
    """Count total tokens in message"""
    return count_tokens(system_prompt, tokenizer) + count_tokens(user_content, tokenizer)

def _pack_blocks(ctx_docs: List[Document], per_doc_cap_tokens: int = 320) -> List[str]:
    """Pack documents into blocks with token-based trimming (matching notebook implementation)"""
    blocks = []
    for d in ctx_docs:
        head = f"[{d.metadata.get('source_file','?')}#p{d.metadata.get('page','?')}]\n"
        body = dedupe_spaces(d.page_content)
        head_tok = count_tokens(head, get_tokenizer(""))
        remain_for_body = max(8, per_doc_cap_tokens - head_tok)
        body = trim_text_to_tokens(body, remain_for_body, get_tokenizer(""))
        blocks.append(head + body)
    return blocks

def _fit_context_with_binary_search(system_prompt: str, user_prefix: str, blocks: List[str], 
                                    model_ctx_limit: int, response_budget: int, safety: int, tokenizer) -> str:
    """Binary search approach to fit context within token limits (matching notebook implementation)"""
    hard_cap = max(512, model_ctx_limit - response_budget - safety)
    
    used_text = ""
    for blk in blocks:
        candidate = f"{user_prefix}{(used_text + '\n\n') if used_text else ''}{blk}"
        t = _message_token_count(system_prompt, candidate, tokenizer)
        if t <= hard_cap:
            used_text = f"{used_text}\n\n{blk}" if used_text else blk
        else:
            # Binary search on the last block
            lo, hi = 0, len(blk)
            best = ""
            while lo <= hi:
                mid = (lo + hi) // 2
                trial = blk[:mid]
                candidate = f"{user_prefix}{(used_text + '\n\n') if used_text else ''}{trial}"
                tt = _message_token_count(system_prompt, candidate, tokenizer)
                if tt <= hard_cap:
                    best = trial
                    lo = mid + 1
                else:
                    hi = mid - 1
            if best.strip():
                used_text = f"{used_text}\n\n{best}" if used_text else best
            break
    return used_text

def synthesize_with_rag(query: str, context_docs: List[Document], config, tokenizer) -> str:
    """Generate response using RAG approach (matching notebook implementation)"""
    # Initialize LLM client
    llm = ChatOpenAI(
        model=config['model_name'],
        openai_api_base=config['model_endpoint'],
        openai_api_key=config['api_key'],
        temperature=config['temperature'],
        max_tokens=config['max_tokens'],
        timeout=120,
    )
    
    # System prompt from configuration
    system_prompt = config['rag_system_prompt']
    
    user_prefix = f"질문: {query}\n\n[컨텍스트]\n"
    
    # Pack context blocks with token-based trimming
    blocks = _pack_blocks(context_docs, config['per_doc_cap'])
    
    # Use binary search to fit context within limits
    safe_context = _fit_context_with_binary_search(
        system_prompt, user_prefix, blocks,
        config['model_ctx_limit'], config['max_tokens'], config['safety_margin'], tokenizer
    )
    
    user_prompt = f"{user_prefix}{safe_context}\n\n규칙: 사실 확인 가능한 문장만 작성."
    
    # Generate response
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    
    try:
        logger.info("Generating RAG response...")
        response = llm.invoke(messages)
        return response.content
    except Exception as e:
        logger.error(f"Error generating RAG response: {e}")
        return f"Error generating RAG response: {str(e)}"

def synthesize_without_rag(query: str, config) -> str:
    """Generate response without RAG context (baseline comparison)"""
    # Initialize LLM client
    llm = ChatOpenAI(
        model=config['model_name'],
        openai_api_base=config['model_endpoint'],
        openai_api_key=config['api_key'],
        temperature=config['temperature'],
        max_tokens=config['max_tokens'],
        timeout=120,
    )
    
    # System prompt from configuration
    system_prompt = config['non_rag_system_prompt']
    user_prompt = f"질문: {query}"
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    
    try:
        logger.info("Generating non-RAG response...")
        response = llm.invoke(messages)
        return response.content
    except Exception as e:
        logger.error(f"Error generating non-RAG response: {e}")
        return f"Error generating non-RAG response: {str(e)}"

def load_query_results(query_dir: Path):
    """Load query results and retrieved documents"""
    # Load retrieved documents
    docs_path = query_dir / "retrieved_documents.pkl"
    if not docs_path.exists():
        raise FileNotFoundError("retrieved_documents.pkl not found. Run task4_query_processing.py first.")
    
    with open(docs_path, 'rb') as f:
        documents = pickle.load(f)
    
    # Load query results metadata
    results_path = query_dir / "query_results.json"
    query_results = None
    if results_path.exists():
        with open(results_path, 'r', encoding='utf-8') as f:
            query_results = json.load(f)
    
    return documents, query_results

def main():
    """Main function to generate response"""
    config = load_config()
    
    # Use query from config or load from results
    query = config['query']
    
    # Load retrieved documents
    try:
        context_docs, query_results = load_query_results(config['query_dir'])
        logger.info(f"Loaded {len(context_docs)} context documents")
        
        # Use query from results if not provided in config
        if not query and query_results:
            query = query_results['query']
            
    except FileNotFoundError as e:
        logger.error(str(e))
        return
    
    if not query:
        logger.error("No query found. Set QUERY environment variable or run query processing first.")
        return
    
    # Initialize tokenizer
    tokenizer = get_tokenizer(config['tokenizer_model'])
    if tokenizer is None:
        logger.warning("Using fallback token counting")
    
    # Generate both RAG and non-RAG responses for comparison
    try:
        logger.info("Generating responses with RAG and without RAG for comparison...")
        
        # RAG response (using retrieved context)
        rag_response = synthesize_with_rag(query, context_docs, config, tokenizer)
        logger.info("RAG response generated successfully")
        
        # Non-RAG response (baseline)
        non_rag_response = synthesize_without_rag(query, config)
        logger.info("Non-RAG response generated successfully")
        
    except Exception as e:
        logger.error(f"Failed to generate responses: {e}")
        return
    
    # Prepare final results with comparison
    final_results = {
        'query': query,
        'rag_response': rag_response,
        'non_rag_response': non_rag_response,
        'context_documents': len(context_docs),
        'comparison': {
            'rag_length_chars': len(rag_response),
            'non_rag_length_chars': len(non_rag_response),
            'rag_length_words': len(rag_response.split()),
            'non_rag_length_words': len(non_rag_response.split()),
            'length_ratio': len(rag_response.split()) / len(non_rag_response.split()) if non_rag_response else 0,
            'context_documents_used': len(context_docs),
            'rag_has_citations': '[' in rag_response and '#p' in rag_response,
            'rag_mentions_backend': 'backend.ai' in rag_response.lower(),
            'non_rag_mentions_backend': 'backend.ai' in non_rag_response.lower(),
            'rag_has_steps': any(word in rag_response.lower() for word in ['단계', '순서', '방법', '절차']),
            'non_rag_has_steps': any(word in non_rag_response.lower() for word in ['단계', '순서', '방법', '절차']),
        },
        'config': {
            'model_endpoint': config['model_endpoint'],
            'model_name': config['model_name'],
            'temperature': config['temperature'],
            'max_tokens': config['max_tokens'],
            'model_ctx_limit': config['model_ctx_limit'],
        },
        'context_preview': []
    }
    
    # Add context preview
    for i, doc in enumerate(context_docs[:3]):  # First 3 docs
        preview = {
            'rank': i + 1,
            'source': f"{doc.metadata.get('source_file', '?')}#p{doc.metadata.get('page', '?')}",
            'preview': doc.page_content[:150] + "..." if len(doc.page_content) > 150 else doc.page_content
        }
        final_results['context_preview'].append(preview)
    
    # Save response
    response_path = config['response_dir'] / "response.json"
    with open(response_path, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, ensure_ascii=False, indent=2)
    logger.info(f"Response saved to {response_path}")
    
    # Print results with clear side-by-side comparison
    print("=" * 100)
    print("QUERY:")
    print(query)
    print("=" * 100)
    
    print("\n🔍 RESPONSE WITH RAG (Using Retrieved Documents):")
    print("-" * 100)
    print(rag_response)
    
    print("\n" + "=" * 100)
    print("\n❌ RESPONSE WITHOUT RAG (No Context, Model Knowledge Only):")
    print("-" * 100)
    print(non_rag_response)
    
    print("\n" + "=" * 100)
    print("📋 QUICK COMPARISON:")
    print(f"• RAG Response: {len(rag_response.split())} words ({len(rag_response)} chars)")
    print(f"• Non-RAG Response: {len(non_rag_response.split())} words ({len(non_rag_response)} chars)")
    print(f"• Documents Used: {len(context_docs)} retrieved documents")
    
    # Check for key differences
    rag_has_sources = '[' in rag_response and '#p' in rag_response
    print(f"• Source Citations: {'✅ RAG includes sources' if rag_has_sources else '❌ No sources in either'}")
    
    print(f"\n💡 Key Difference: RAG response is based on actual documents, Non-RAG relies only on training data")

if __name__ == "__main__":
    main()