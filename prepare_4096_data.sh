#!/bin/bash
set -e

# -----------------------------
# 0. DEPENDENCY CHECK & INSTALL
# -----------------------------
echo "Checking dependencies..."
python3 -m pip install --quiet requests sentencepiece huggingface_hub tqdm numpy torch

# Ensure directories exist
mkdir -p data/tokenizers
mkdir -p data/datasets
mkdir -p logs

# Use all available CPU cores for fast tokenization
export MATCHED_FINEWEB_TOKENIZER_THREADS=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 8)
echo "Using $MATCHED_FINEWEB_TOKENIZER_THREADS CPU threads for tokenization."

# -----------------------------
# 1. DATA PREPARATION (STREAMING)
# -----------------------------
# This pulls ~750MB of raw text to train the 4096 tokenizer and export shards.
# No 48GB download required.
# SentencePiece training is CPU-bound but uses all threads.
echo "Starting streaming tokenization (Target: 4096 Vocab)..."
python3 data/download_hf_docs_and_tokenize.py \
  --output-root data/ \
  --tokenizer-train-docs 30000 \
  --limit-docs 200000 \
  --reuse-sp-model 1024=data/tokenizers/fineweb_1024_bpe.model \
  --skip-byte

echo ""
echo "========================================================="
echo "DATA PREPARATION COMPLETE"
echo "========================================================="
echo "Shards created at:    ./data/datasets/fineweb10B_sp4096"
echo "Tokenizer created at: ./data/tokenizers/fineweb_4096_bpe.model"
echo "VOCAB_SIZE is:        4096"
echo ""
echo "To run your training script (e.g. baseline_3090.py), use:"
echo "export DATA_PATH='./data/datasets/fineweb10B_sp4096'"
echo "export TOKENIZER_PATH='./data/tokenizers/fineweb_4096_bpe.model'"
echo "export VOCAB_SIZE=4096"
echo "python3 baseline_3090.py"
echo "========================================================="
