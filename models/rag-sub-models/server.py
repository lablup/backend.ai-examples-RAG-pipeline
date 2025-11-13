import os
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from FlagEmbedding import FlagReranker
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

# === Read Model ID from environment variable ===
EMBED_MODEL_ID = os.getenv("EMBED_MODEL_ID", "nlpai-lab/KURE-v1")
TOKENIZER_MODEL_ID = os.getenv(
    "TOKENIZER_MODEL_ID",
    "MLP-KTLim/llama-3-Korean-Bllossom-8B",
)
RERANKER_MODEL_ID = os.getenv(
    "RERANKER_MODEL_ID",
    "BAAI/bge-reranker-v2-m3",
)

app = FastAPI(
    title="KURE Embedding + Bllossom Tokenizer + BGE Reranker",
    version="1.0.0",
    description=(
        "CPU-only RAG utility service running on Backend.AI.\n"
        "- /v1/embeddings: OpenAI-style embeddings API\n"
        "- /tokenize: tokenizer API\n"
        "- /v1/rerank: OpenAI-style rerank API"
    ),
)

# === Load embedding model (CPU only) ===
print(f"[EMBED] Loading embedding model on CPU: {EMBED_MODEL_ID}")
embed_model = SentenceTransformer(EMBED_MODEL_ID, device="cpu")

print(f"[TOKENIZER] Loading tokenizer: {TOKENIZER_MODEL_ID}")
tokenizer = AutoTokenizer.from_pretrained(
    TOKENIZER_MODEL_ID,
    use_fast=True,
)

print(f"[RERANK] Loading reranker on CPU: {RERANKER_MODEL_ID}")
reranker = FlagReranker(
    RERANKER_MODEL_ID,
    use_fp16=False,
    device="cpu",
)


# === Pydantic model definition ===

class EmbedRequest(BaseModel):
    texts: List[str]


class EmbedResponse(BaseModel):
    model: str
    dimension: int
    embeddings: List[List[float]]


class TokenizeRequest(BaseModel):
    texts: List[str]
    add_special_tokens: bool = True


class TokenizeResponse(BaseModel):
    model: str
    input_ids: List[List[int]]
    attention_mask: List[List[int]]
    tokens: List[List[str]]
    token_counts: List[int]


class RerankRequest(BaseModel):
    query: str
    documents: List[str]
    max_length: int = 1024
    top_k: Optional[int] = None  # Restrict to top_k results (None means all)


class RerankedDocument(BaseModel):
    index: int        # index from documents list
    score: float
    document: str


class RerankResponse(BaseModel):
    model: str
    results: List[RerankedDocument]


# === Internal Helpers ===

def _compute_rerank(
    query: str,
    documents: List[str],
    max_length: int = 1024,
) -> List[float]:
    """
    Compute rerank scores using FlagReranker.
    """
    pairs = [[query, doc] for doc in documents]
    scores = reranker.compute_score(
        pairs,
        max_length=max_length,
    )
    # FlagEmbedding may return float or tensor values, so convert to float
    return [float(s) for s in scores]


# === Endpoints ===

@app.get("/")
async def root():
    return {
        "service": "kure-bllossom-bge-rerank",
        "endpoints": [
            "/health",
            "/embed",
            "/v1/embeddings",
            "/tokenize",
            "/rerank",
            "/v1/rerank",
        ],
    }


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "embed_model": EMBED_MODEL_ID,
        "tokenizer_model": TOKENIZER_MODEL_ID,
        "reranker_model": RERANKER_MODEL_ID,
    }


# --- 1) /embed: Custom Embedding API (for maintaining previous format) ---


@app.post("/embed", response_model=EmbedResponse)
async def embed(req: EmbedRequest):
    if not req.texts:
        return EmbedResponse(
            model=EMBED_MODEL_ID,
            dimension=0,
            embeddings=[],
        )

    vectors = embed_model.encode(
        req.texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).tolist()

    dim = len(vectors[0]) if vectors else 0

    return EmbedResponse(
        model=EMBED_MODEL_ID,
        dimension=dim,
        embeddings=vectors,
    )


# --- 2) /v1/embeddings: OpenAI-style Embeddings API ---

@app.post("/v1/embeddings")
async def openai_embeddings(body: Dict[str, Any]):
    """
    OpenAI-style /v1/embeddings endpoint.

    Request example:
    {
      "model": "kure",
      "input": "single sentence"  or ["sentence1", "sentence2"]
    }
    """
    if "input" not in body:
        raise HTTPException(status_code=400, detail="'input' field is required")

    raw_input = body["input"]

    # Convert input to list of strings
    texts = [raw_input] if isinstance(raw_input, str) else [str(x) for x in raw_input]
    if not isinstance(raw_input, (str, list)):
        raise HTTPException(status_code=400, detail="'input' must be string or list of strings")

    if not texts:
        return {
            "object": "list",
            "data": [],
            "model": EMBED_MODEL_ID,
            "usage": {
                "prompt_tokens": 0,
                "total_tokens": 0,
            },
        }

    vectors = embed_model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).tolist()

    data = []
    for i, v in enumerate(vectors):
        data.append({
            "object": "embedding",
            "embedding": v,
            "index": i,
        })

    # Calculate token count approximately using Bllossom tokenizer (optional)
    encoded = tokenizer(
        texts,
        padding=False,
        truncation=False,
        add_special_tokens=True,
    )
    token_counts = [len(ids) for ids in encoded["input_ids"]]
    total_tokens = int(sum(token_counts))

    return {
        "object": "list",
        "data": data,
        "model": EMBED_MODEL_ID,
        "usage": {
            "prompt_tokens": total_tokens,
            "total_tokens": total_tokens,
        },
    }


# --- 3) /tokenize: tokenize API ---

@app.post("/tokenize", response_model=TokenizeResponse)
async def tokenize(req: TokenizeRequest):
    if not req.texts:
        return TokenizeResponse(
            model=TOKENIZER_MODEL_ID,
            input_ids=[],
            attention_mask=[],
            tokens=[],
            token_counts=[],
        )

    encoded = tokenizer(
        req.texts,
        padding=False,
        truncation=False,
        add_special_tokens=req.add_special_tokens,
    )

    tokens_batch = [
        tokenizer.convert_ids_to_tokens(ids)
        for ids in encoded["input_ids"]
    ]
    token_counts = [len(ids) for ids in encoded["input_ids"]]

    return TokenizeResponse(
        model=TOKENIZER_MODEL_ID,
        input_ids=encoded["input_ids"],
        attention_mask=encoded["attention_mask"],
        tokens=tokens_batch,
        token_counts=token_counts,
    )


# --- 4) /rerank: custom Rerank API ---

@app.post("/rerank", response_model=RerankResponse)
async def rerank_endpoint(req: RerankRequest):
    if not req.documents:
        return RerankResponse(model=RERANKER_MODEL_ID, results=[])

    scores = _compute_rerank(
        query=req.query,
        documents=req.documents,
        max_length=req.max_length,
    )

    indexed = [
        (i, s, doc)
        for i, (s, doc) in enumerate(zip(scores, req.documents))
    ]

    # Sort by score descending
    indexed.sort(key=lambda x: x[1], reverse=True)

    # If top_k is specified, truncate to top_k
    if req.top_k is not None:
        indexed = indexed[: req.top_k]

    results = [
        RerankedDocument(index=i, score=s, document=d)
        for i, s, d in indexed
    ]

    return RerankResponse(
        model=RERANKER_MODEL_ID,
        results=results,
    )


# --- 5) /v1/rerank: OpenAI-style Rerank API ---

@app.post("/v1/rerank")
async def openai_rerank(body: Dict[str, Any]):
    """
    OpenAI-style /v1/rerank endpoint (unofficial compatibility format).

    Request example:
    {
      "model": "bge-reranker",
      "query": "question",
      "documents": ["document1", "document2", ...],
      "top_n": 3        # optional (OpenAI uses the name top_n)
    }

    Response example:
    {
      "object": "list",
      "data": [
        {"index": 1, "relevance_score": 0.95},
        {"index": 0, "relevance_score": 0.74},
        ...
      ],
      "model": "BAAI/bge-reranker-v2-m3"
    }
    """
    if "query" not in body or "documents" not in body:
        raise HTTPException(
            status_code=400,
            detail="'query' and 'documents' fields are required",
        )

    query = str(body["query"])
    documents = body["documents"]
    if not isinstance(documents, list):
        raise HTTPException(
            status_code=400,
            detail="'documents' must be a list of strings",
        )
    documents = [str(d) for d in documents]

    top_n = body.get("top_n", None)
    max_length = int(body.get("max_length", 1024))

    if not documents:
        return {
            "object": "list",
            "data": [],
            "model": RERANKER_MODEL_ID,
        }

    scores = _compute_rerank(
        query=query,
        documents=documents,
        max_length=max_length,
    )

    indexed = [
        (i, s)
        for i, s in enumerate(scores)
    ]
    indexed.sort(key=lambda x: x[1], reverse=True)

    if top_n is not None:
        indexed = indexed[: int(top_n)]

    data = [
        {
            "index": i,
            "relevance_score": float(s),
        }
        for i, s in indexed
    ]

    return {
        "object": "list",
        "data": data,
        "model": RERANKER_MODEL_ID,
    }
