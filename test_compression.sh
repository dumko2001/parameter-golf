#!/bin/bash
# Test script for compression-iteration architecture
# Uses EXACT same config as warmup_v2.sh to compare against baseline

export CUDA_LAUNCH_BLOCKING=0
export TORCH_CUDNN_V8_API_ENABLED=1
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16

cd /workspace/parameter-golf/parameter-golf

RUN_ID=compression_test_$(date +%Y%m%d_%H%M%S) \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
ITERATIONS=200 \
WARMUP_STEPS=10 \
TRAIN_BATCH_TOKENS=262144 \
TRAIN_SEQ_LEN=1024 \
VAL_LOSS_EVERY=10 \
VAL_SUBSET_FRAC=0.05 \
TRAIN_LOG_EVERY=50 \
torchrun --standalone --nproc_per_node=1 train_gpt_compression_test.py

echo "--- Compression Test Complete ---"
