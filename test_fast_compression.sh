#!/bin/bash
# Test script for aggressive compression config
# Reduces base architecture but keeps ALL compression features (K-means, Hadamard, zstd)

cd /workspace/parameter-golf/parameter-golf

export CUDA_LAUNCH_BLOCKING=0
export TORCH_CUDNN_V8_API_ENABLED=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

# Aggressive reduction config - try to hit <300ms
export NUM_LAYERS=6
export MODEL_DIM=384
export TRAIN_SEQ_LEN=512
export MLP_MULT=1.5
export BIGRAM_DIM=32
export BIGRAM_VOCAB_SIZE=2048

# Keep ALL compression features
export INT8_HADAMARD=1
export INT8_VQ=1
export VAL_LOSS_EVERY=50
export TRAIN_LOG_EVERY=10

# Optional responsiveness mode:
# nice -n 10 taskset -c 0-7 torchrun --standalone --nproc_per_node=1 ...

RUN_ID=fast_compression_$(date +%Y%m%d_%H%M%S) \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
ITERATIONS=50 \
WARMUP_STEPS=5 \
TRAIN_BATCH_TOKENS=131072 \
torchrun --standalone --nproc_per_node=1 train_gpt_latest.py

echo "--- Fast Compression Test Complete ---"
