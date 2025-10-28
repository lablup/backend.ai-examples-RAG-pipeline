# Task 6: Web Interface for RAG Comparison

A Gradio-based web interface for comparing RAG vs non-RAG responses side-by-side.

## Features

- **Interactive Query Input**: Enter questions in Korean or English
- **Side-by-Side Comparison**: RAG vs non-RAG responses simultaneously
- **Source Display**: View which documents were used for RAG response
- **Real-time Processing**: Instant responses from your LLM endpoint
- **Example Queries**: Pre-built Backend.AI questions

## Prerequisites

Tasks 1-3 must be completed first to generate the required indexes:
- `../indexes/faiss/` - Vector search index
- `../indexes/bm25.pkl` - BM25 keyword search index  
- `../indexes/index_metadata.pkl` - Index configuration

## Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables
export INDEX_DIR=../indexes
export MODEL_ENDPOINT=your-llm-endpoint
export MODEL_NAME=your-model-name

# Run the interface
python main.py
# Access at: http://localhost:7860
```

## Docker Deployment

### Option 1: Docker Compose (Recommended)

```bash
# Build and run with docker-compose
docker-compose up -d

# View logs
docker-compose logs -f

# Stop
docker-compose down
```

### Option 2: Docker Build & Run

```bash
# Build the image
docker build -t korean-rag-web .

# Run the container
docker run -d \
  --name korean-rag-web \
  -p 7860:7860 \
  -v $(pwd)/../indexes:/app/indexes:ro \
  -e MODEL_ENDPOINT=your-endpoint \
  -e MODEL_NAME=your-model \
  -e API_KEY=your-key \
  -e HUGGINGFACE_TOKEN=your-hf-token \
  korean-rag-web
```

## Environment Variables

### Required
- `INDEX_DIR`: Path to indexes directory (default: `/app/indexes`)
- `MODEL_ENDPOINT`: LLM API endpoint
- `MODEL_NAME`: LLM model name

### Optional
- `API_KEY`: API key for LLM service (default: `dummy-key`)
- `TEMPERATURE`: LLM temperature (default: `0.2`)
- `MAX_TOKENS`: Max response tokens (default: `3000`)
- `TOP_K`: Initial retrieval count (default: `32`)
- `FINAL_K`: Final document count (default: `6`)
- `USE_RERANK`: Enable reranking (default: `true`)
- `HUGGINGFACE_TOKEN`: HuggingFace API token

## Container Independence

The web interface container is completely independent and only requires:

1. **Index Files**: Mount the `indexes` directory as a volume
2. **LLM Access**: Network access to your LLM endpoint
3. **Environment Config**: Set model endpoint and parameters

### Volume Mounts

```bash
# Required volume
-v /path/to/indexes:/app/indexes:ro

# Optional volumes (if you need access to other data)
-v /path/to/processed:/app/processed:ro
-v /path/to/query_results:/app/query_results
```

### Network Requirements

- **Outbound HTTP/HTTPS**: To reach LLM endpoint
- **Inbound HTTP**: Port 7860 for web interface
- **Optional**: HuggingFace API access for models

## Production Deployment

### Docker Swarm
```yaml
version: '3.8'
services:
  rag-web:
    image: korean-rag-web:latest
    ports:
      - "7860:7860"
    volumes:
      - indexes:/app/indexes:ro
    environment:
      - MODEL_ENDPOINT=https://your-llm-endpoint/v1
    deploy:
      replicas: 2
      restart_policy:
        condition: on-failure
```

### Kubernetes
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: korean-rag-web
spec:
  replicas: 2
  selector:
    matchLabels:
      app: korean-rag-web
  template:
    metadata:
      labels:
        app: korean-rag-web
    spec:
      containers:
      - name: web
        image: korean-rag-web:latest
        ports:
        - containerPort: 7860
        volumeMounts:
        - name: indexes
          mountPath: /app/indexes
          readOnly: true
        env:
        - name: MODEL_ENDPOINT
          value: "https://your-llm-endpoint/v1"
      volumes:
      - name: indexes
        persistentVolumeClaim:
          claimName: rag-indexes-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: korean-rag-web-service
spec:
  selector:
    app: korean-rag-web
  ports:
  - port: 80
    targetPort: 7860
  type: LoadBalancer
```

## Troubleshooting

### Common Issues

1. **"Index metadata not found"**
   - Ensure tasks 1-3 have been completed
   - Check that `indexes` directory is properly mounted
   - Verify file permissions

2. **"Failed to load FAISS index"**
   - Check that FAISS index files exist in `indexes/faiss/`
   - Ensure proper read permissions on mounted volume

3. **"Connection refused to LLM endpoint"**
   - Verify MODEL_ENDPOINT is accessible from container
   - Check network connectivity and firewall rules
   - Validate API_KEY if required

### Health Check
```bash
# Check container status
docker ps | grep korean-rag-web

# View logs
docker logs korean-rag-web

# Test endpoint
curl http://localhost:7860
```

## Performance Considerations

- **Memory**: ~2-8GB depending on index size
- **CPU**: 2-4 cores recommended for reranking
- **Storage**: Read-only access to indexes (~100MB-2GB)
- **Network**: Bandwidth depends on LLM endpoint response time

The container is stateless and can be easily scaled horizontally.