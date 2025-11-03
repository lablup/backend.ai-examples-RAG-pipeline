#!/bin/bash

# copy environment variables template and load them
cp .env.template .env
source .env

# add VLLM environment variable
export VLLM_ATTENTION_BACKEND=TORCH_SDPA
export VLLM_USE_FLASH_ATTENTION=0
export VLLM_USE_FLASHINFER=0
export CUDA_LAUNCH_BLOCKING=1

# download model files from huggingface if not already present
if [ ! -f "${MODEL_PATH}/config.json" ]; then
    pip install huggingface_hub
    huggingface-cli download ${HF_MODEL_REPO} --local-dir ${MODEL_PATH}
fi

python3 -m vllm.entrypoints.openai.api_server \
--model ${MODEL_PATH} \
--served-model-name ${MODEL_NAME} \
--tokenizer ${MODEL_PATH} \
--tokenizer-mode "slow" \
--quantization "awq" \
--tensor-parallel-size ${TENSOR_PARALLEL_SIZE} \
--host "0.0.0.0" \
--port ${MODEL_PORT} \
--max-model-len ${MAX_MODEL_LEN} \
--max-num-batched-tokens ${MAX_NUM_BATCHED_TOKENS} \
--dtype ${D_TYPE} \
--max-num-seq ${MAX_NUM_SEQ} \
--enforce-eager \
--gpu-memory-utilization ${GPU_MEMORY_UTILIZATION} \
