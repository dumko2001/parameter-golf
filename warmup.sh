#!/bash/bin
# Warmup script for Parameter Golf training

echo "--- Starting Warmup Training ---"

RUN_ID=warmup_$(date +%Y%m%d_%H%M%S) \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
VAL_LOSS_EVERY=0 \
MAX_STEPS=10 \
torchrun --standalone --nproc_per_node=1 train_gpt.py

echo "--- Warmup Complete ---"
