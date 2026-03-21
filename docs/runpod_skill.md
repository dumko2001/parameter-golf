# RunPod Skill: High-Performance Iteration

This document summarizes the best practices for running Parameter Golf experiments on RunPod at the lowest possible cost with maximum throughput.

## 1. Hardware & Tiering
*   **Target GPU**: **RTX 4090** (Community Cloud).
*   **Market Price**: **$0.20 - $0.25 / hr**.
*   **Why**: The 4090 offers the highest "Value per TFLOP". It processes ~600k tokens per second for a 17M parameter model.
*   **Bid Strategy**: Use **Spot/Interruptible** with a bid of ~$0.20.

## 2. Storage & Persistence (The "Stop" Rule)
*   **Container Disk**: Use **50 GB** local scratch disk. 
    *   *Rationale*: The image (10GB) + Data (10GB) + Libraries (2GB) requires ~22GB total. 50GB provides safety.
*   **Network Volume**: Set to **0 GB**. Persistent network volumes are 5x more expensive than local disk.
*   **Retention**: Never **Terminate** a pod until the project is finished. Use **STOP**.
    *   *Stopping* costs ~$0.12 / day but keeps the 10GB Docker layers and your FineWeb shards **cached** on the host. Starting a stopped pod is **instant**.

## 3. Training Environment
To see logs in real-time and avoid common crashes, always export these variables:

```bash
export PYTHONUNBUFFERED=1    # Disables log buffering (essential for live tailing)
export MASTER_ADDR=localhost # Required for distributed initialization
export MASTER_PORT=49152     # Use a high, unique port to avoid "Address in Use" errors
export RANK=0
export WORLD_SIZE=1          # Running on a single GPU
export NCCL_P2P_DISABLE=1    # Bypasses P2P locks on some consumer cards
```

## 4. Execution Patterns
*   **Triton Compilation**: The first run on a 4090 will spend 2-5 minutes at **110% CPU** (Compiling Triton kernels). Do not panic; the log will be silent until "Step 0".
*   **Throughput Calibration**: Use **2,000 steps** for a definitive baseline.
*   **Early Signal**: BPB scores stabilize significantly by **Step 500**. You can usually see if an architecture change is working by then (~6 minutes of training).
*   **Data Strategy**: Use a subset of shards for fast iteration:
    ```bash
    python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 10
    ```

## 5. Helpful Commands
*   **Live Metrics**: `tail -f final_baseline.log`
*   **Force Cleanup**: `pkill -9 python3` (Warning: May disconnect SSH session).
*   **Check Resource Usage**: `top -b -n 1`
