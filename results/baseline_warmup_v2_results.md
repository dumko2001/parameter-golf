# Baseline Warmup v2 Results
## Date: 2026-03-21 08:00

## Configuration (warmup_v2.sh)
```
RUN_ID=warmup_v2_$(date +%Y%m%d_%H%M%S)
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
torchrun --standalone --nproc_per_node=1 train_gpt.py
```

## Architecture Specs
- model_params: 17,059,912
- num_layers: 9
- model_dim: 512
- num_heads: 8
- num_kv_heads: 4
- mlp_mult: 2
- tie_embeddings: True
- embed_lr: 0.6
- matrix_lr: 0.04
- scalar_lr: 0.04
- muon_momentum: 0.95
- world_size: 1
- grad_accum_steps: 8
- sdp_backends: cudnn=False flash=True mem_efficient=False math=False

## Training Metrics
| Step | train_loss | val_loss | val_bpb | train_time | step_avg |
|------|------------|----------|---------|------------|----------|
| 0 | N/A | 6.9345 | 4.1647 | 0ms | 0.02ms |
| 1 | 6.9369 | N/A | N/A | 813ms | 812.70ms |
| 2 | 16.8267 | N/A | N/A | 1634ms | 817.20ms |
| 3 | 10.8511 | N/A | N/A | 2456ms | 818.65ms |
| 4 | 7.1223 | N/A | N/A | 3279ms | 819.66ms |
| 5 | 6.2808 | N/A | N/A | 4099ms | 819.75ms |
| 6 | 6.2470 | N/A | N/A | 4919ms | 819.88ms |
| 7 | 6.2057 | N/A | N/A | 5739ms | 819.83ms |
| 8 | 6.0032 | N/A | N/A | 6558ms | 819.77ms |
| 9 | 5.9478 | N/A | N/A | 7379ms | 819.94ms |
| 10 | 5.9470 | 5.8984 | 3.5424 | 8199ms | 819.95ms |
| 20 | N/A | 5.2011 | 3.1237 | 16404ms | 820.21ms |
| 30 | N/A | 4.6002 | 2.7628 | 24618ms | 820.61ms |
| 40 | N/A | 4.3352 | 2.6036 | 32828ms | 820.71ms |
| 50 | 4.2009 | 4.1457 | 2.4898 | 41046ms | 820.92ms |
| 60 | N/A | 3.9032 | 2.3441 | 49263ms | 821.05ms |
| 70 | N/A | 3.7216 | 2.2351 | 57488ms | 821.26ms |
| 80 | N/A | 3.5813 | 2.1509 | 65711ms | 821.38ms |
| 90 | N/A | 3.4708 | 2.0845 | 73994ms | 822.15ms |
| 100 | 3.4070 | 3.3822 | 2.0313 | 82216ms | 822.16ms |
| 110 | N/A | 3.2934 | 1.9779 | 90445ms | 822.23ms |
| 120 | N/A | 3.2376 | 1.9444 | 98674ms | 822.28ms |
| 130 | N/A | 3.1707 | 1.9043 | 106901ms | 822.31ms |
| 140 | N/A | 3.0927 | 1.8574 | 115133ms | 822.38ms |
| 150 | 3.1182 | 3.0360 | 1.8233 | 123537ms | 823.58ms |
| 160 | N/A | 2.9999 | 1.8017 | 134772ms | 842.32ms |
| 170 | N/A | 2.9547 | 1.7745 | 146066ms | 859.21ms |
| 180 | N/A | 2.9230 | 1.7555 | 156750ms | 870.84ms |
| 190 | N/A | 2.8935 | 1.7378 | 168011ms | 884.27ms |
| 200 | 2.8933 | 2.8523 | 1.7130 | 179280ms | 896.40ms |

## Final Results
- peak memory allocated: 5253 MiB
- peak memory reserved: 5634 MiB
- Serialized model: 67,224,983 bytes (~67MB)
- Code size: 61,492 bytes
- Total submission size: 67,286,475 bytes
- Serialized model int8+zlib: 10,375,218 bytes (payload:17,178,912 raw_torch:17,224,025 payload_ratio:3.91x)
- Total submission size int8+zlib: 10,436,710 bytes
- final_int8_zlib_roundtrip val_loss: 2.8577
- final_int8_zlib_roundtrip val_bpb: 1.7162
- eval_time: 4004ms
- final_int8_zlib_roundtrip_exact val_loss: 2.85766127
- final_int8_zlib_roundtrip_exact val_bpb: 1.71623603

## Log File
logs/warmup_v2_20260321_080046.txt
