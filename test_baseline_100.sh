#!/bin/bash
# Baseline test script (warmup_v2 config) - for comparison
cd /workspace/parameter-golf/parameter-golf

export CUDA_LAUNCH_BLOCKING=0
export TORCH_CUDNN_V8_API_ENABLED=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

# Reduce batch for faster completion
export TRAIN_BATCH_TOKENS=65536  # Lower host/GPU pressure profile
export VAL_SUBSET_FRAC=0.1  # 10% validation

# Optional responsiveness mode:
# nice -n 10 taskset -c 0-7 torchrun --standalone --nproc_per_node=1 ...

RUN_ID=baseline_$(date +%Y%m%d_%H%M%S) \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
ITERATIONS=100 \
WARMUP_STEPS=10 \
TRAIN_SEQ_LEN=1024 \
VAL_LOSS_EVERY=20 \
TRAIN_LOG_EVERY=20 \
torchrun --standalone --nproc_per_node=1 train_gpt.py

echo "--- Baseline Test Complete ---"
