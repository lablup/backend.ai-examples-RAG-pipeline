# Korean RAG Sequential Tasks

This project implements a Retrieval-Augmented Generation (RAG) system for Korean documents, broken down into 5 sequential tasks that can be run independently in separate environments (e.g., Docker containers).

## Overview

The RAG pipeline is divided into these sequential tasks:

1. **Data Ingestion** (`task1_data_ingestion/main.py`) - Extract and clean text from PDF files
2. **Document Processing** (`task2_document_processing/main.py`) - Create document chunks and generate embeddings
3. **Index Building** (`task3_index_building/main.py`) - Build FAISS and BM25 search indexes
4. **Query Processing** (`task4_query_processing/main.py`) - Perform hybrid retrieval and reranking
5. **Response Generation** (`task5_response_generation/main.py`) - Generate final response using LLM

Each task installs its own Python packages and can run in a completely isolated environment.

## Configuration

All configuration is handled through environment variables. See `.env.template` for all available options.

### Key Configuration Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `MODEL_ENDPOINT` | LLM API endpoint | Your LLM endpoint URL |
| `EMBED_MODEL` | HuggingFace embedding model | Your embedding model |
| `RERANK_MODEL` | Cross-encoder reranking model | Your reranker model |
| `CHUNK_SIZE` | Text chunk size | `180` |
| `CHUNK_OVERLAP` | Chunk overlap | `80` |
| `TOP_K` | Initial retrieval count | `32` |
| `FINAL_K` | Final reranked results | `6` |
| `RERANK_POOL` | Reranking pool size | `40` |
| `HUGGINGFACE_TOKEN` | HuggingFace API token | - |
| `RAG_SYSTEM_PROMPT` | System prompt for RAG responses | Backend.AI expert prompt |
| `NON_RAG_SYSTEM_PROMPT` | System prompt for non-RAG responses | Backend.AI assistant prompt |

## Quick Start

1. **Setup configuration**:
   ```bash
   cp .env.template .env
   # Edit .env with your configuration
   ```

2. **Add your PDF files** to the `data/` directory

3. **Run tasks individually** as needed (see below)

## Running Individual Tasks

Each task can be run independently with proper environment variables:

```bash
# Task 1: Data Ingestion
cd task1_data_ingestion
pip install -r requirements.txt
export DATA_DIR=../data CACHE_DIR=../cleaned
python main.py

# Task 2: Document Processing  
cd task2_document_processing
pip install -r requirements.txt
export CACHE_DIR=../cleaned PROCESSED_DIR=../processed CHUNK_SIZE=180 CHUNK_OVERLAP=80 EMBED_MODEL=nlpai-lab/KURE-v1
python main.py

# Task 3: Index Building
cd task3_index_building
pip install -r requirements.txt
export PROCESSED_DIR=../processed INDEX_DIR=../indexes
python main.py

# Task 4: Query Processing
cd task4_query_processing
pip install -r requirements.txt
export INDEX_DIR=../indexes QUERY_DIR=../query_results TOP_K=32 FINAL_K=6 RERANK_POOL=40 QUERY="Your question here"
python main.py

# Task 5: Response Generation (Command Line)
cd task5_response_generation
pip install -r requirements.txt
export QUERY_DIR=../query_results RESPONSE_DIR=../responses MODEL_ENDPOINT="your-endpoint" MODEL_NAME="your-model"
python main.py

```

## Optimization: Skipping Tasks for Different Scenarios

When running the pipeline multiple times under different conditions, you can skip certain tasks to save time:

### **Scenario 1: New Data Inputs (New PDFs)**
**Skip:** Nothing - need full pipeline
```bash
# Run all tasks (1→2→3→4→5)
./run_task.sh
```
**Reason:** New data requires complete reprocessing

### **Scenario 2: Changed LLM Models**
**Skip:** Tasks 1-4 (only regenerate response)
```bash
# Only run Task 5
cd task5_response_generation
pip install -r requirements.txt
python main.py
```
**Reason:** Same retrieved context, just different LLM for generation

### **Scenario 3: Changed Chunk Size**
**Skip:** Task 1 (PDFs already extracted)
```bash
# Run tasks 2→3→4→5
cd task2_document_processing && pip install -r requirements.txt && python main.py && cd ..
cd task3_index_building && pip install -r requirements.txt && python main.py && cd ..
cd task4_query_processing && pip install -r requirements.txt && python main.py && cd ..
cd task5_response_generation && pip install -r requirements.txt && python main.py && cd ..
```
**Reason:** Text extraction unchanged, but chunks need rebuilding

### **Scenario 4: Changed Tokenizer Model**
**Skip:** Tasks 1-4 (only affects response generation)
```bash
# Only run Task 5
cd task5_response_generation
pip install -r requirements.txt
python main.py
```
**Reason:** Tokenizer only affects context packing in response generation

### **Additional Scenarios:**

| Change | Skip Tasks | Run Tasks | Reason |
|--------|------------|-----------|---------|
| Embedding Model | 1 | 2→3→4→5 | Need new embeddings and indexes |
| Rerank Model | 1,2,3 | 4→5 | Only affects reranking step |
| Query Text | 1,2,3 | 4→5 | Same indexes, new retrieval |
| Retrieval Parameters (TOP_K, FINAL_K) | 1,2,3 | 4→5 | Same indexes, different retrieval |

### **Time-Saving Tips:**
- **Most expensive:** Task 2 (embedding generation) and Task 3 (index building)
- **Quickest:** Task 5 (response generation) - usually under 30 seconds
- **Medium:** Task 1 (PDF extraction) and Task 4 (retrieval)

## Resource Requirements

Each task has different computing resource needs. Minimum baseline: **1 CPU Core, 1 GiB RAM, 64 MB Shared Memory**

### **Task 1: Data Ingestion (PDF Extraction)**
- **CPU:** 1-2 cores (I/O bound, minimal CPU usage)
- **RAM:** 512 MB - 2 GiB (depends on PDF size)
- **Storage:** 10-50 MB per PDF for text cache
- **Time:** 1-5 minutes for small PDFs, 10+ minutes for large documents
- **Bottleneck:** Disk I/O and PDF complexity

### **Task 2: Document Processing (Embeddings)**
- **CPU:** 2-8 cores (embedding model inference)
- **RAM:** 4-16 GiB (model loading + document batches)
- **GPU:** Optional but recommended (10x speedup)
- **Storage:** 100 MB - 2 GiB for embeddings cache
- **Time:** 10-60 minutes (CPU), 2-10 minutes (GPU)
- **Bottleneck:** Model inference, most resource-intensive task

### **Task 3: Index Building**
- **CPU:** 2-4 cores (FAISS index construction)
- **RAM:** 2-8 GiB (holds all embeddings in memory)
- **Storage:** 200 MB - 1 GiB for index files
- **Time:** 2-15 minutes
- **Bottleneck:** Memory bandwidth and vector operations

### **Task 4: Query Processing (Retrieval)**
- **CPU:** 1-4 cores (hybrid search + reranking)
- **RAM:** 2-6 GiB (indexes loaded in memory)
- **GPU:** Optional for reranker (2x speedup)
- **Time:** 10-60 seconds per query
- **Bottleneck:** Reranker inference if enabled

### **Task 5: Response Generation**
- **CPU:** 1-2 cores (API calls, tokenization)
- **RAM:** 1-4 GiB (tokenizer + context processing)
- **Network:** Stable connection to LLM endpoint
- **Time:** 5-30 seconds per query
- **Bottleneck:** LLM API response time

### **Resource Scaling Recommendations:**

| Document Count | Task 2 RAM | Task 3 RAM | Total Time (CPU) |
|----------------|-------------|-------------|------------------|
| < 100 pages | 4 GiB | 2 GiB | 15-30 min |
| 100-1000 pages | 8 GiB | 4 GiB | 30-90 min |
| 1000+ pages | 16+ GiB | 8+ GiB | 90+ min |

### **Cost-Effective Options:**
- **CPU-only setup:** Minimum 4 GiB RAM, expect 2-4x longer processing
- **GPU acceleration:** Reduces Task 2 time by ~80%, Task 4 reranking by ~50%
- **Memory optimization:** Use smaller embedding models (e.g., `all-MiniLM-L6-v2` instead of `KURE-v1`)


## Environment Variables Reference

### Directory Paths
- `DATA_DIR`: PDF files directory
- `CACHE_DIR`: Cleaned text files directory  
- `PROCESSED_DIR`: Processed documents directory
- `INDEX_DIR`: Search indexes directory
- `QUERY_DIR`: Query results directory
- `RESPONSE_DIR`: Final responses directory
- `EVAL_DATASET_PATH`: Evaluation dataset JSON file path
- `RESULTS_DIR`: Evaluation results directory

### Model Configuration
- `MODEL_ENDPOINT`: LLM API endpoint
- `MODEL_NAME`: LLM model name
- `API_KEY`: API key for LLM service
- `EMBED_MODEL`: Embedding model name
- `HUGGINGFACE_TOKEN`: HuggingFace token
- `RAG_SYSTEM_PROMPT`: System prompt for RAG responses
- `NON_RAG_SYSTEM_PROMPT`: System prompt for non-RAG responses

### Processing Parameters
- `CHUNK_SIZE`: Document chunk size
- `CHUNK_OVERLAP`: Overlap between chunks
- `TOP_K`: Initial retrieval count
- `FINAL_K`: Final document count
- `RERANK_POOL`: Reranking pool size
- `USE_RERANK`: Enable reranking (true/false)
- `RERANK_MODEL`: Reranker model name

### Response Generation
- `TEMPERATURE`: LLM temperature
- `MAX_TOKENS`: Maximum response tokens
- `MODEL_CTX_LIMIT`: Model context limit
- `TOKENIZER_MODEL`: Tokenizer model name
- `PER_DOC_CAP`: Max tokens per document
- `SAFETY_MARGIN`: Token counting safety margin

## File Structure

```
backend.ai-examples-RAG-pipeline/
├── README.md                               # Main documentation
├── .env.template                           # Environment configuration template
├── run_task.sh                            # Task execution script
├── setup_dirs.sh                          # Directory setup script
├── data/                                   # PDF files (input)
│   └── sample/                            # Sample data directory
│       └── Backend.AI Web-UI User Guide (v25.05.250508KO).pdf
├── models/                                # Model configurations
│   ├── RAG-nonRAG-service/               # RAG evaluation service
│   │   ├── main.py                       # Evaluation system with web interface
│   │   ├── requirements.txt              # Python dependencies
│   │   ├── entrypoint.sh                 # Entry point script
│   │   └── model-definition.yml          # Backend.AI model definition
│   ├── llama-3-Korean-Bllossom-8B-awq/   # LLM model configuration
│   │   ├── model-definition.yaml         # Backend.AI model definition
│   │   └── run-model-with-vllm.sh        # vLLM startup script
│   └── rag-sub-models/                   # RAG sub-models service
│       ├── model-definition.yaml         # Backend.AI model definition
│       ├── requirements.txt              # Python dependencies
│       ├── run-model-with-fastapi.sh     # FastAPI startup script
│       └── server.py                     # FastAPI server implementation
├── pipeline/                              # Pipeline configuration
│   └── RAG-pipeline.yaml                 # YAML pipeline definition
├── tasks/                                 # Task implementations
│   ├── task1_data_ingestion/
│   │   ├── main.py                       # PDF extraction and cleaning
│   │   └── requirements.txt              # Python dependencies
│   ├── task2_document_processing/
│   │   ├── main.py                       # Chunking and embedding generation
│   │   └── requirements.txt              # Python dependencies
│   ├── task3_index_building/
│   │   ├── main.py                       # FAISS and BM25 index creation
│   │   └── requirements.txt              # Python dependencies
│   ├── task4_query_processing/
│   │   ├── main.py                       # Hybrid retrieval and reranking
│   │   └── requirements.txt              # Python dependencies
│   └── task5_response_generation/
│       ├── main.py                       # LLM response generation
│       └── requirements.txt              # Python dependencies
└── utils/                                 # Utility functions and helpers

# Generated directories (created during execution):
├── cleaned/                               # Cleaned text files (Task 1)
├── processed/                             # Processed documents and embeddings (Task 2)
├── indexes/                               # FAISS and BM25 indexes (Task 3)
├── query_results/                         # Query processing results (Task 4)
├── responses/                             # Final responses (Task 5)
```

## Data Flow

### Sequential Pipeline Flow
```
PDFs (data/) 
    ↓ Task 1: Data Ingestion
Cleaned Text (cleaned/) 
    ↓ Task 2: Document Processing  
Documents + Embeddings (processed/)
    ↓ Task 3: Index Building
FAISS + BM25 Indexes (indexes/)
    ↓ Task 4: Query Processing
Retrieved Context (query_results/)
    ↓ Task 5: Response Generation
Final Response (responses/)
```

### Key Data Artifacts
- **Task 1 Output**: Cleaned text files with source markers
- **Task 2 Output**: Document chunks + embeddings + embedding config
- **Task 3 Output**: FAISS vectorstore + BM25 index + metadata
- **Task 4 Output**: Retrieved documents + sources + query metadata  
- **Task 5 Output**: RAG vs Non-RAG responses + comparison metrics

Each task produces persistent output enabling complete pipeline isolation, debugging, and independent execution.

## Notes

- **Environment Isolation**: Each task installs its own Python packages for complete environment isolation
- **Debugging Support**: All intermediate results are saved to disk for debugging and inspection
- **Model Flexibility**: System supports any HuggingFace embeddings, LLM endpoints, and reranker models
- **Configurable System Prompts**: Both RAG and non-RAG prompts are configurable via environment variables
- **Optional Components**: Reranking is optional and can be disabled via `USE_RERANK=false`
- **Token Management**: Precise token counting ensures accurate context packing within model limits
- **Vendor Neutral**: No hardcoded model defaults - all models must be explicitly configured by users
