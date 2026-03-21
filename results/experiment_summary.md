# Parameter Golf Experiment Summary
## Date: 2026-03-21

## Current Status
- **Current branch**: master
- **Working directory**: /workspace/parameter-golf
- **Compression test file**: train_gpt_compression_test.py (from origin/compression-iteration)

## Results Saved
- `results/baseline_warmup_v2_results.md` - Baseline warmup_v2 full results
- `results/compression_branch_test_results.md` - Compression branch test results (partial)
- `logs/warmup_v2_20260321_080046.txt` - Baseline training log
- `logs/compression_test_20260321_134850.txt` - Compression test log

## Quick Comparison (val_bpb - lower is better)

| Step | Baseline | Compression | Winner |
|------|----------|-------------|--------|
| 0 | 4.1647 | 4.1035 | Compression (-0.061) |
| 10 | 3.5424 | 3.4981 | Compression (-0.044) |
| 20 | 3.1237 | 3.3308 | Baseline (+0.207) |

## Architecture Summary
| Metric | Baseline | Compression |
|--------|----------|-------------|
| model_params | 17.0M | 32.9M (2x) |
| step_avg | ~820ms | ~1530ms (2x slower) |
| mlp_mult | 2 | 3 |
| embed_lr | 0.6 | 0.03 |
| matrix_lr | 0.04 | 0.02 |

## Key Findings
1. Compression branch starts with better val_bpb despite 2x more parameters
2. Baseline catches up by step 20 due to faster training speed
3. Compression has many advanced features (Differential Attention, K-Means Quantization, SWA, etc.)
4. Test timed out - needs longer runtime for full 200 iterations

## Next Steps
- Create new branch for runpod experiments
- Run full 200 iterations on compression branch with longer timeout
- Compare final compressed sizes (int8+zstd vs int8+zlib)
- Test various hyperparameter combinations
