#!/usr/bin/env python3
"""
Task 7: RAG vs Non-RAG Evaluation
- Systematic evaluation framework with comprehensive metrics
- Automatic scoring using ROUGE, BERTScore, and custom metrics  
- Interactive web interface for single question testing and manual evaluation
- Batch evaluation mode for full dataset processing
- Statistical analysis with category breakdown and performance comparison
- Export results with timestamps for tracking improvements

Environment Variables:
- INDEX_DIR: Input directory with search indexes from Task 3
- EVAL_DATASET_PATH: Path to evaluation dataset JSON file
- RESULTS_DIR: Output directory for evaluation results
- MODEL_ENDPOINT: LLM API endpoint URL (required, e.g. https://.../v1 without /v1)
- MODEL_NAME: LLM model name (required)
- API_KEY: API key for LLM service
- TEMPERATURE: LLM temperature for response generation
- MAX_TOKENS: Maximum tokens for response
- FINAL_K: Number of final documents after reranking
- USE_RERANK: Enable/disable reranking (true/false)
- USE_LOCAL_RERANK: Use local reranker (true) or remote rerank API (false)
- RERANK_MODEL: Local reranker model name (e.g. BAAI/bge-reranker-v2-m3) if USE_LOCAL_RERANK=true
- RERANK_SERVICE_ENDPOINT: Remote rerank API endpoint (e.g. https://.../v1/rerank) if USE_LOCAL_RERANK=false
- RAG_SYSTEM_PROMPT: System prompt for RAG responses
- NON_RAG_SYSTEM_PROMPT: System prompt for non-RAG responses
- HUGGINGFACE_TOKEN: HuggingFace API token (optional)

Requirements: pip install -r requirements.txt
"""

import json
import logging
import os
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import gradio as gr
import numpy as np
import pandas as pd
import requests
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Evaluation metrics
try:
    from bert_score import score as bert_score
    from rouge_score import rouge_scorer
    METRICS_AVAILABLE = True
except ImportError:
    METRICS_AVAILABLE = False
    logging.warning("Some evaluation metrics not available. Install rouge-score and bert-score for full functionality.")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables
vectorstore = None
bm25_retriever = None
reranker = None
llm_client = None
evaluation_dataset = None
RERANK_IS_REMOTE = False
RERANK_SERVICE_ENDPOINT = ""


def load_config():
    """Load configuration from environment variables"""
    return {
        'index_dir': Path(os.getenv('INDEX_DIR', '../indexes')),
        'eval_dataset_path': Path(os.getenv('EVAL_DATASET_PATH', './sample/evaluation_dataset.json')),
        'results_dir': Path(os.getenv('RESULTS_DIR', '../eval_results')),

        'model_endpoint': os.getenv('MODEL_ENDPOINT', ''),
        'model_name': os.getenv('MODEL_NAME', ''),
        'api_key': os.getenv('API_KEY', 'dummy-key'),
        'temperature': float(os.getenv('TEMPERATURE', '0.2')),
        'max_tokens': int(os.getenv('MAX_TOKENS', '3000')),

        'final_k': int(os.getenv('FINAL_K', '6')),

        'use_rerank': os.getenv('USE_RERANK', 'true').lower() == 'true',
        'use_local_rerank': os.getenv('USE_LOCAL_RERANK', 'false').lower() == 'true',
        'rerank_model': os.getenv('RERANK_MODEL', ''),
        'rerank_endpoint': os.getenv('RERANK_SERVICE_ENDPOINT', '').rstrip('/'),

        'huggingface_token': os.getenv('HUGGINGFACE_TOKEN', ''),
        'rag_system_prompt': os.getenv(
            'RAG_SYSTEM_PROMPT',
            "당신은 Backend.AI 전문가입니다. 아래 컨텍스트를 바탕으로 정확하고 구체적인 답변을 제공하세요. "
            "근거가 부족하면 '근거 부족'이라고 말하세요."
        ),
        'non_rag_system_prompt': os.getenv(
            'NON_RAG_SYSTEM_PROMPT',
            "당신은 Backend.AI 전문 어시스턴트입니다. 간결하고 정확한 답변을 제공하세요."
        ),
    }


def load_evaluation_dataset(dataset_path: Path) -> List[Dict]:
    """Load evaluation dataset"""
    if not dataset_path.exists():
        logger.error(f"Dataset not found at {dataset_path}")
        return []

    with open(dataset_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    logger.info(f"Loaded {len(dataset)} evaluation questions")
    return dataset


def load_components(config):
    """Load RAG components (vectorstore, BM25, reranker, LLM)"""
    global vectorstore, bm25_retriever, reranker, llm_client, RERANK_IS_REMOTE, RERANK_SERVICE_ENDPOINT

    index_dir = config['index_dir']
    metadata_path = index_dir / "index_metadata.pkl"

    if not metadata_path.exists():
        raise FileNotFoundError(f"Index metadata not found at {metadata_path}")

    with open(metadata_path, 'rb') as f:
        index_metadata = pickle.load(f)

    embedding_config = index_metadata['embedding_config']

    # --- Embeddings for FAISS load (local HF or remote OpenAI-style) ---
    if 'embed_model' in embedding_config:
        # Local HuggingFace embeddings
        if config['huggingface_token']:
            os.environ['HF_TOKEN'] = config['huggingface_token']
        embeddings = HuggingFaceEmbeddings(model_name=embedding_config['embed_model'])
        logger.info(f"Using local HuggingFaceEmbeddings: {embedding_config['embed_model']}")
    else:
        # Remote embedding service
        embeddings = OpenAIEmbeddings(
            base_url=embedding_config['embed_endpoint'].rstrip("/"),
            api_key="dummy-key",
            model=embedding_config['embed_model_alias']
        )
        logger.info(
            f"Using remote embeddings: {embedding_config['embed_endpoint']} "
            f"model={embedding_config['embed_model_alias']}"
        )

    # --- Load indexes ---
    faiss_path = Path(index_metadata['faiss_path'])
    vectorstore = FAISS.load_local(str(faiss_path), embeddings, allow_dangerous_deserialization=True)

    bm25_path = Path(index_metadata['bm25_path'])
    with open(bm25_path, 'rb') as f:
        bm25_retriever = pickle.load(f)

    logger.info("FAISS & BM25 indexes loaded successfully")

    # --- Reranker (local or remote) ---
    reranker = None
    RERANK_IS_REMOTE = False
    RERANK_SERVICE_ENDPOINT = ""

    if config['use_rerank']:
        if config['use_local_rerank']:
            try:
                from FlagEmbedding import FlagReranker
                reranker = FlagReranker(config['rerank_model'], use_fp16=False, device="cpu")
                logger.info(f"Local reranker loaded: {config['rerank_model']}")
            except Exception as e:
                logger.warning(f"Failed to load local reranker: {e}")
                reranker = None
        else:
            # remote rerank API
            if not config['rerank_endpoint']:
                logger.warning("USE_RERANK=true but RERANK_SERVICE_ENDPOINT is empty. Rerank will be disabled.")
            else:
                RERANK_IS_REMOTE = True
                RERANK_SERVICE_ENDPOINT = config['rerank_endpoint']
                logger.info(f"Using remote rerank endpoint: {RERANK_SERVICE_ENDPOINT}")

    # --- LLM client (OpenAI-style endpoint) ---
    base_url = config['model_endpoint'].rstrip('/') + '/v1'
    llm_client = ChatOpenAI(
        model=config['model_name'],
        base_url=base_url,
        api_key=config['api_key'],
        temperature=config['temperature'],
        max_tokens=config['max_tokens'],
        timeout=120,
    )

    logger.info("All components loaded successfully")


def _remote_rerank_call(query: str, docs: List[Document], top_k: int) -> List[Document]:
    """Call remote rerank API: POST RERANK_SERVICE_ENDPOINT"""
    if not RERANK_SERVICE_ENDPOINT:
        return docs[:top_k]

    payload = {
        "query": query,
        "documents": [d.page_content for d in docs],
        "top_n": top_k,
    }

    try:
        r = requests.post(RERANK_SERVICE_ENDPOINT, json=payload, timeout=120)
        r.raise_for_status()
        data = r.json()
        ranked_indices = [item["index"] for item in data.get("data", [])]
        ranked_docs = [docs[i] for i in ranked_indices]
        return ranked_docs
    except Exception as e:
        logger.error(f"Remote rerank API call failed: {e}")
        return docs[:top_k]


def retrieve_and_generate(query: str, config) -> Tuple[str, str, List[str]]:
    """Generate both RAG and non-RAG responses"""
    global reranker, llm_client, vectorstore, bm25_retriever

    # Simplified retrieval
    bm25_docs = bm25_retriever.invoke(query)[:config['final_k']]
    vector_docs = vectorstore.as_retriever(
        search_kwargs={"k": config['final_k']}
    ).invoke(query)

    # Combine documents, dedupe by object id
    all_docs = list({id(doc): doc for doc in bm25_docs + vector_docs}.values())

    # Apply reranking if configured
    if config['use_rerank'] and all_docs:
        if RERANK_IS_REMOTE:
            final_docs = _remote_rerank_call(query, all_docs, config['final_k'])
        elif reranker is not None:
            try:
                pairs = [[query, doc.page_content] for doc in all_docs]
                scores = reranker.compute_score(pairs)
                reranked = sorted(zip(all_docs, scores), key=lambda x: x[1], reverse=True)
                final_docs = [doc for doc, _ in reranked[:config['final_k']]]
            except Exception as e:
                logger.warning(f"Local rerank failed: {e}")
                final_docs = all_docs[:config['final_k']]
        else:
            final_docs = all_docs[:config['final_k']]
    else:
        final_docs = all_docs[:config['final_k']]

    # Context (간단 버전: 앞부분만 잘라 사용)
    context = "\n\n".join(
        f"[{doc.metadata.get('source_file','?')}#p{doc.metadata.get('page','?')}]\n{doc.page_content[:500]}"
        for doc in final_docs
    )

    # RAG response
    rag_system = config['rag_system_prompt']
    rag_user = f"질문: {query}\n\n[컨텍스트]\n{context}\n\n답변:"

    rag_response = llm_client.invoke(
        [
            {"role": "system", "content": rag_system},
            {"role": "user", "content": rag_user},
        ]
    ).content

    # Non-RAG response
    non_rag_system = config['non_rag_system_prompt']
    non_rag_user = f"질문: {query}"

    non_rag_response = llm_client.invoke(
        [
            {"role": "system", "content": non_rag_system},
            {"role": "user", "content": non_rag_user},
        ]
    ).content

    sources = [
        f"{doc.metadata.get('source_file','?')}#p{doc.metadata.get('page','?')}"
        for doc in final_docs
    ]

    return rag_response, non_rag_response, sources


def calculate_metrics(predicted: str, reference: str) -> Dict[str, float]:
    """Calculate evaluation metrics with enhanced scoring"""
    metrics = {}

    if not METRICS_AVAILABLE:
        return {"error": "Metrics not available"}

    try:
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=False)
        rouge_scores = scorer.score(reference, predicted)

        for key, score in rouge_scores.items():
            metrics[f"{key}_precision"] = score.precision
            metrics[f"{key}_recall"] = score.recall
            metrics[f"{key}_fmeasure"] = score.fmeasure

        try:
            P, R, F1 = bert_score([predicted], [reference], lang='ko', verbose=False)
            metrics["bertscore_precision"] = P.mean().item()
            metrics["bertscore_recall"] = R.mean().item()
            metrics["bertscore_f1"] = F1.mean().item()
        except Exception as e:
            logger.warning(f"BERTScore calculation failed: {e}")
            metrics["bertscore_precision"] = 0.0
            metrics["bertscore_recall"] = 0.0
            metrics["bertscore_f1"] = 0.0

        pred_words = predicted.lower().split()
        ref_words = reference.lower().split()

        length_ratio = len(pred_words) / max(len(ref_words), 1)
        metrics["length_ratio"] = length_ratio
        metrics["length_score"] = (
            1.0 - abs(1.0 - length_ratio) if length_ratio <= 2.0 else max(0.3, 1.0 / length_ratio)
        )

        pred_set = set(pred_words)
        ref_set = set(ref_words)
        overlap = len(pred_set.intersection(ref_set))
        metrics["keyword_overlap"] = overlap / max(len(ref_set), 1)

        metrics["insufficient_evidence_penalty"] = -0.2 if "근거 부족" in predicted else 0.0
        metrics["citation_bonus"] = 0.1 if any(
            marker in predicted for marker in ["출처:", "파일명", ".pdf", "#p"]
        ) else 0.0

        base_score = (
            0.25 * metrics.get("rouge1_fmeasure", 0)
            + 0.20 * metrics.get("rouge2_fmeasure", 0)
            + 0.20 * metrics.get("rougeL_fmeasure", 0)
            + 0.25 * metrics.get("bertscore_f1", 0)
            + 0.05 * metrics.get("length_score", 0)
            + 0.05 * metrics.get("keyword_overlap", 0)
        )

        composite_score = base_score + metrics["citation_bonus"] + metrics["insufficient_evidence_penalty"]
        metrics["composite_score"] = max(0.0, min(1.0, composite_score))

        if metrics["composite_score"] >= 0.8:
            metrics["quality_tier"] = "Excellent"
        elif metrics["composite_score"] >= 0.6:
            metrics["quality_tier"] = "Good"
        elif metrics["composite_score"] >= 0.4:
            metrics["quality_tier"] = "Fair"
        else:
            metrics["quality_tier"] = "Poor"

    except Exception as e:
        logger.error(f"Metrics calculation failed: {e}")
        metrics["error"] = str(e)

    return metrics


def evaluate_single_question(item: Dict, config) -> Dict:
    """Evaluate a single question"""
    question = item['question']
    expected = item['expected_answer']

    try:
        rag_response, non_rag_response, sources = retrieve_and_generate(question, config)

        rag_metrics = calculate_metrics(rag_response, expected)
        non_rag_metrics = calculate_metrics(non_rag_response, expected)

        rag_has_sources = '[' in rag_response and '#p' in rag_response
        rag_length = len(rag_response.split())
        non_rag_length = len(non_rag_response.split())

        return {
            'id': item['id'],
            'question': question,
            'expected_answer': expected,
            'category': item.get('category', 'unknown'),
            'difficulty': item.get('difficulty', 'medium'),
            'requires_docs': item.get('requires_specific_docs', False),
            'rag_response': rag_response,
            'non_rag_response': non_rag_response,
            'sources': sources,
            'rag_metrics': rag_metrics,
            'non_rag_metrics': non_rag_metrics,
            'rag_has_sources': rag_has_sources,
            'rag_length': rag_length,
            'non_rag_length': non_rag_length,
            'retrieved_docs': len(sources),
        }

    except Exception as e:
        logger.error(f"Error evaluating question {item['id']}: {e}")
        return {
            'id': item['id'],
            'error': str(e),
            'question': question,
            'expected_answer': expected,
        }


def run_full_evaluation(config, progress_callback=None) -> Dict:
    """Run evaluation on full dataset"""
    dataset = load_evaluation_dataset(config['eval_dataset_path'])
    if not dataset:
        return {"error": "No dataset loaded"}

    results = []
    total = len(dataset)

    for i, item in enumerate(dataset):
        if progress_callback:
            progress_callback(i + 1, total, item["id"])
        result = evaluate_single_question(item, config)
        logger.info(f"answered result for question {item['id']}: {result}")
        results.append(result)

    summary = analyze_results(results)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = config['results_dir'] / f"evaluation_results_{timestamp}.json"
    config['results_dir'].mkdir(parents=True, exist_ok=True)

    evaluation_report = {
        'timestamp': timestamp,
        'config': {k: str(v) for k, v in config.items()},
        'summary': summary,
        'detailed_results': results,
    }

    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(evaluation_report, f, ensure_ascii=False, indent=2)

    logger.info(f"Evaluation results saved to {results_file}")
    return evaluation_report


def analyze_results(results: List[Dict]) -> Dict:
    """Analyze evaluation results and generate summary"""
    if not results or 'error' in results[0]:
        return {"error": "No valid results to analyze"}

    valid_results = [r for r in results if 'error' not in r]
    if not valid_results:
        return {"error": "No successful evaluations"}

    summary = {
        'total_questions': len(results),
        'successful_evaluations': len(valid_results),
        'failed_evaluations': len(results) - len(valid_results),
    }

    if METRICS_AVAILABLE and valid_results:
        rag_rouge1_f1 = [r['rag_metrics'].get('rouge1_fmeasure', 0) for r in valid_results]
        non_rag_rouge1_f1 = [r['non_rag_metrics'].get('rouge1_fmeasure', 0) for r in valid_results]

        if rag_rouge1_f1 and non_rag_rouge1_f1:
            summary.update(
                {
                    'rag_rouge1_mean': float(np.mean(rag_rouge1_f1)),
                    'non_rag_rouge1_mean': float(np.mean(non_rag_rouge1_f1)),
                    'rag_rouge1_std': float(np.std(rag_rouge1_f1)),
                    'non_rag_rouge1_std': float(np.std(non_rag_rouge1_f1)),
                    'rag_wins': sum(
                        1 for i in range(len(rag_rouge1_f1)) if rag_rouge1_f1[i] > non_rag_rouge1_f1[i]
                    ),
                    'non_rag_wins': sum(
                        1 for i in range(len(rag_rouge1_f1)) if non_rag_rouge1_f1[i] > rag_rouge1_f1[i]
                    ),
                    'ties': sum(
                        1 for i in range(len(rag_rouge1_f1)) if rag_rouge1_f1[i] == non_rag_rouge1_f1[i]
                    ),
                }
            )

    categories = {}
    for result in valid_results:
        cat = result.get('category', 'unknown')
        if cat not in categories:
            categories[cat] = {'count': 0, 'rag_better': 0, 'non_rag_better': 0}
        categories[cat]['count'] += 1

        if METRICS_AVAILABLE:
            rag_score = result['rag_metrics'].get('rouge1_fmeasure', 0)
            non_rag_score = result['non_rag_metrics'].get('rouge1_fmeasure', 0)
            if rag_score > non_rag_score:
                categories[cat]['rag_better'] += 1
            elif non_rag_score > rag_score:
                categories[cat]['non_rag_better'] += 1

    summary['category_analysis'] = categories
    return summary


def create_evaluation_interface():
    """Create Gradio interface for evaluation"""
    config = load_config()

    try:
        load_components(config)
        dataset = load_evaluation_dataset(config['eval_dataset_path'])
    except Exception as e:
        logger.error(f"Failed to initialize components for Gradio: {e}")
        dataset = []

    def evaluate_single_gradio(question, expected_answer=""):
        """Evaluate a single question with detailed scoring"""
        if not question.strip():
            return "질문을 입력해주세요.", "", "", "", "", 0.0, 0.0

        try:
            item = {
                'id': 'custom',
                'question': question,
                'expected_answer': expected_answer or '참고 답변 없음',
            }
            result = evaluate_single_question(item, config)

            if 'error' in result:
                return f"오류: {result['error']}", "", "", "", "", 0.0, 0.0

            rag_metrics = result.get('rag_metrics', {})
            non_rag_metrics = result.get('non_rag_metrics', {})

            rag_score_details = f"""
**📊 RAG 점수 상세:**
- ROUGE-1 F1: {rag_metrics.get('rouge1_fmeasure', 0):.3f}
- ROUGE-L F1: {rag_metrics.get('rougeL_fmeasure', 0):.3f}
- BERTScore F1: {rag_metrics.get('bertscore_f1', 0):.3f}
- 길이 점수: {rag_metrics.get('length_score', 0):.3f}
- 키워드 겹침: {rag_metrics.get('keyword_overlap', 0):.3f}
- 인용 보너스: {rag_metrics.get('citation_bonus', 0):.3f}
- **종합 점수**: {rag_metrics.get('composite_score', 0):.3f}
- **품질 등급**: {rag_metrics.get('quality_tier', 'N/A')}
"""

            non_rag_score_details = f"""
**📊 Non-RAG 점수 상세:**
- ROUGE-1 F1: {non_rag_metrics.get('rouge1_fmeasure', 0):.3f}
- ROUGE-L F1: {non_rag_metrics.get('rougeL_fmeasure', 0):.3f}
- BERTScore F1: {non_rag_metrics.get('bertscore_f1', 0):.3f}
- 길이 점수: {non_rag_metrics.get('length_score', 0):.3f}
- 키워드 겹침: {non_rag_metrics.get('keyword_overlap', 0):.3f}
- 근거 부족 페널티: {non_rag_metrics.get('insufficient_evidence_penalty', 0):.3f}
- **종합 점수**: {non_rag_metrics.get('composite_score', 0):.3f}
- **품질 등급**: {non_rag_metrics.get('quality_tier', 'N/A')}
"""

            sources_info = (
                f"사용된 문서: {len(result['sources'])} / 출처: {', '.join(result['sources'][:3])}"
                if result['sources']
                else "문서를 찾지 못함"
            )

            return (
                result['rag_response'],
                result['non_rag_response'],
                rag_score_details,
                non_rag_score_details,
                sources_info,
                rag_metrics.get('composite_score', 0.0),
                non_rag_metrics.get('composite_score', 0.0),
            )

        except Exception as e:
            return f"오류: {str(e)}", "", "", "", "", 0.0, 0.0

    with gr.Blocks(title="RAG Evaluation", theme=gr.themes.Soft()) as demo:
        gr.HTML("<h1 style='text-align: center;'>🧪 RAG vs Non-RAG Evaluation</h1>")

        with gr.Tab("🔍 Single Question Test"):
            gr.Markdown("Test individual questions to see RAG vs Non-RAG comparison with detailed scoring.")

            with gr.Row():
                question_input = gr.Textbox(
                    label="질문 입력",
                    placeholder="예: Backend.AI에서 모델 서비스를 생성하는 방법은?",
                    lines=2,
                )
                expected_input = gr.Textbox(
                    label="예상 답변 (선택사항)",
                    placeholder="점수 계산을 위한 참고 답변",
                    lines=2,
                )

            test_button = gr.Button("테스트", variant="secondary")

            with gr.Row():
                with gr.Column():
                    gr.HTML("<h4>🔍 RAG 응답</h4>")
                    rag_output = gr.Textbox(label="RAG 응답", lines=8, show_copy_button=True)
                    rag_score_display = gr.Markdown(label="RAG 점수 상세")
                with gr.Column():
                    gr.HTML("<h4>❌ Non-RAG 응답</h4>")
                    non_rag_output = gr.Textbox(label="Non-RAG 응답", lines=8, show_copy_button=True)
                    non_rag_score_display = gr.Markdown(label="Non-RAG 점수 상세")

            with gr.Row():
                sources_output = gr.Textbox(label="사용된 소스", lines=2)
                with gr.Column():
                    rag_score_num = gr.Number(label="RAG 종합점수", precision=3)
                    non_rag_score_num = gr.Number(label="Non-RAG 종합점수", precision=3)

            gr.Examples(
                examples=[
                    [
                        "Backend.AI에서 제한된 사용자만 모델 서비스에 접근하도록 설정하는 방법은?",
                        "모델 서비스 생성 시 'Open to Public' 비활성화, 토큰 발급을 통한 인증 설정, 엔드포인트 접속 시 토큰 사용",
                    ],
                    [
                        "Backend.AI에서 모델 서빙을 위한 토큰 발급 절차는?",
                        "모델 서비스가 비공개인 경우 필요, Backend.AI 웹 UI의 토큰 발급 섹션에서 생성",
                    ],
                    [
                        "Backend.AI에서 GPU 리소스를 효율적으로 활용하는 방법은?",
                        "적절한 GPU 타입 선택, 배치 크기 조정, 메모리 사용량 모니터링",
                    ],
                ],
                inputs=[question_input, expected_input],
            )

            test_button.click(
                fn=evaluate_single_gradio,
                inputs=[question_input, expected_input],
                outputs=[
                    rag_output,
                    non_rag_output,
                    rag_score_display,
                    non_rag_score_display,
                    sources_output,
                    rag_score_num,
                    non_rag_score_num,
                ],
            )

    return demo


def main():
    """Main function"""
    demo = create_evaluation_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        inbrowser=False,
        show_error=True,
    )


if __name__ == "__main__":
    main()
