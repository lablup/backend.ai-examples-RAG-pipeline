#!/bin/bash
set -euo pipefail

cd /models

echo "[SERVICE] Starting model service with FastAPI"

# 0) .env.template → .env auto-create
if [ -f ".env.template" ] && [ ! -f ".env" ]; then
  echo "[SERVICE] .env not found. Creating from .env.template..."
  cp .env.template .env
fi

# 1) Load .env
if [ -f ".env" ]; then
  echo "[SERVICE] Loading environment variables from .env"
  # export all variables in .env
  set -a
  source .env
  set +a
else
  echo "[SERVICE] Warning: .env not found. Using default environment variables."
fi

# 2) Install dependencies
if [ -f "requirements.txt" ]; then
  echo "[SERVICE] Installing dependencies from requirements.txt..."
  pip install --no-cache-dir -r requirements.txt
else
  echo "[SERVICE] requirements.txt not found, installing minimal deps..."
  pip install --no-cache-dir \
    fastapi \
    "uvicorn[standard]" \
    sentence-transformers \
    transformers \
    FlagEmbedding \
    pydantic
fi

# 3) Setup Environment Variable (overrided if already in .env)

: "${TOKENIZER_MODEL_ID:=MLP-KTLim/llama-3-Korean-Bllossom-8B}"
: "${RERANKER_MODEL_ID:=BAAI/bge-reranker-v2-m3}"

export EMBED_MODEL_ID
export TOKENIZER_MODEL_ID
export RERANKER_MODEL_ID

# 4) Check server.py exists
if [ ! -f "server.py" ]; then
  echo "[SERVICE][ERROR] server.py not found in /models"
  exit 1
fi

echo "[SERVICE] Launching uvicorn..."
exec python3 -m uvicorn server:app --host 0.0.0.0 --port 8000
