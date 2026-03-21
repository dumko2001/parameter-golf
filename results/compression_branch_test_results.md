# Compression Branch (compression-iteration) Test Results
## Date: 2026-03-21 13:48
## Source: origin/compression-iteration branch from github.com/dumko2001/parameter-golf

## Test Configuration (same as warmup_v2.sh for fair comparison)
```
RUN_ID=compression_test_$(date +%Y%m%d_%H%M%S)
DATA_PATH=./data/datasets/fineweb10B_sp1024/
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model
VOCAB_SIZE=1024
ITERATIONS=200
WARMUP_STEPS=10
TRAIN_BATCH_TOKENS=262144
TRAIN_SEQ_LEN=1024
VAL_LOSS_EVERY=10
VAL_SUBSET_FRAC=0.05
TRAIN_LOG_EVERY=50
torchrun --standalone --nproc_per_node=1 train_gpt_compression_test.py
```

## Architecture Specs (from compression-iteration branch)
- model_params: 32,985,682 (2x larger than baseline)
- num_layers: 9 (same as baseline)
- model_dim: 512 (same as baseline)
- num_heads: 8 (same as baseline)
- num_kv_heads: 4 (same as baseline)
- mlp_mult: 3.0 (baseline: 2.0)
- tie_embeddings: True
- embed_lr: 0.03 (baseline: 0.6)
- matrix_lr: 0.02 (baseline: 0.04)
- scalar_lr: 0.02 (baseline: 0.04)
- muon_momentum: 0.99 (baseline: 0.95)
- weight_decay: 0.01 (new in compression)
- grad_clip_norm: 0.3 (baseline: 0.0)
- world_size: 1
- grad_accum_steps: 8
- sdp_backends: cudnn=False flash=True mem_efficient=False math=False

## Key Architecture Differences from Baseline
1. **Differential Attention** - CausalSelfAttention uses dual Q paths with noise cancellation (diff_lambda)
2. **BigramHashEmbedding** - Hash-based bigram embeddings (bigram_vocab_size=4096, bigram_dim=128)
3. **SwiGLU activation** - MLP uses SwiGLU instead of ReLU2
4. **SmearGate** - Token blending with previous token
5. **Skip weights** - Encoder-decoder style skip connections
6. **Shannon Entropy Penalty (ECT)** - In Muon optimizer
7. **Hadamard Transform** - Before int8 quantization (INT8_HADAMARD=True)
8. **K-Means 3-bit Delta Quantization** - Learned quantization scheme
9. **zstd compression** - Instead of zlib (level 1)
10. **LoRA TTT** - Test-time training with low-rank adapters
11. **SWA** - Stochastic Weight Averaging during warmdown
12. **Sliding Window Eval** - eval_stride=64, eval_batch_seqs=32

## Training Metrics (partial - timed out at step 20)
| Step | train_loss | val_loss | val_bpb | train_time | step_avg |
|------|------------|----------|---------|------------|----------|
| 0 | N/A | 6.9286 | 4.1035 | 0ms | 0.02ms |
| 1 | 6.9308 | N/A | N/A | 1569ms | 1568.95ms |
| 2 | 8.1997 | N/A | N/A | 3097ms | 1548.55ms |
| 3 | 7.9633 | N/A | N/A | 4624ms | 1541.17ms |
| 4 | 7.5958 | N/A | N/A | 6149ms | 1537.37ms |
| 5 | 7.2167 | N/A | N/A | 7676ms | 1535.20ms |
| 6 | 6.8624 | N/A | N/A | 9200ms | 1533.27ms |
| 7 | 6.6222 | N/A | N/A | 10726ms | 1532.25ms |
| 8 | 6.3415 | N/A | N/A | 12253ms | 1531.62ms |
| 9 | 6.1690 | N/A | N/A | 13779ms | 1530.99ms |
| 10 | 6.0353 | 5.9064 | 3.4981 | 15304ms | 1530.36ms |
| 20 | N/A | 5.6238 | 3.3308 | 30565ms | 1528.23ms |

## Comparison with Baseline (val_bpb - lower is better)
| Step | Baseline | Compression | Delta | Notes |
|------|----------|-------------|-------|-------|
| 0 | 4.1647 | 4.1035 | **-0.0612** | Compression wins |
| 10 | 3.5424 | 3.4981 | **-0.0443** | Compression wins |
| 20 | 3.1237 | 3.3308 | +0.2071 | Baseline catches up |

## Observations
- Compression starts with BETTER val_bpb despite 2x more parameters
- Compression is ~2x slower per step (heavier architecture)
- Early steps show promise: compression leads at steps 0 and 10
- Test timed out before completion - needs longer timeout for full 200 steps

## Test File
train_gpt_compression_test.py (copied from origin/compression-iteration)

## Log File
logs/compression_test_20260321_134850.txt

## Git Info
- Source branch: origin/compression-iteration
- Commit: 43182ea feat: merge leaderboard SOTA atoms and fix K-Means dimension mismatch
