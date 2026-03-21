import os
import io
import math
import time
import uuid
import glob
import copy
import random
import subprocess
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch import Tensor
from pathlib import Path

# Try to import zstandard for SOTA compression, fallback to zlib
try:
    import zstandard as zstd
    _COMPRESSOR = "zstd"
except ImportError:
    import zlib
    _COMPRESSOR = "zlib"

import sentencepiece as spm

# -----------------------------
# HYPERPARAMETERS (17M BASELINE)
# -----------------------------

class Hyperparameters:
    data_path = os.environ.get("DATA_PATH", "./data/datasets/fineweb10B_sp1024")
    train_files = os.path.join(data_path, "fineweb_train_*.bin")
    val_files = os.path.join(data_path, "fineweb_val_*.bin")
    tokenizer_path = os.environ.get("TOKENIZER_PATH", "./data/tokenizers/fineweb_1024_bpe.model")
    run_id = os.environ.get("RUN_ID", str(uuid.uuid4()))
    seed = 1337

    # Training Config
    iterations = 200
    train_batch_tokens = 262144
    train_seq_len = 1024
    warmup_steps = 10
    max_wallclock_seconds = 600.0
    
    # Model Config (Yields ~17,059,912 params)
    vocab_size = 1024
    num_layers = 9
    model_dim = 512
    num_heads = 8
    num_kv_heads = 4
    mlp_mult = 2
    tie_embeddings = True
    logit_softcap = 30.0

    # Optimizer (Muon + Adam)
    matrix_lr = 0.04
    scalar_lr = 0.04
    muon_momentum = 0.95
    beta1, beta2 = 0.9, 0.95
    adam_eps = 1e-8
    ect_lambda = 1e-4  # Entropy-Constrained Training penalty

# -----------------------------
# SOTA COMPRESSION UTILITIES
# -----------------------------

def fwht_inplace(x: Tensor):
    """Fast Walsh-Hadamard Transform to spread outliers."""
    d = x.shape[-1]
    if (d & (d - 1)) != 0: return
    h = 1
    while h < d:
        for i in range(0, d, h * 2):
            for j in range(i, i + h):
                x_j = x[..., j].clone()
                x[..., j] += x[..., j + h]
                x[..., j + h] = x_j - x[..., j + h]
        h *= 2
    x /= math.sqrt(d)

def pack_3bit(indices: Tensor) -> np.ndarray:
    """Packs 8x3-bit indices into 3x8-bit bytes."""
    idx = indices.flatten().cpu().numpy().astype(np.uint8)
    pad = (8 - (len(idx) % 8)) % 8
    if pad: idx = np.concatenate([idx, np.zeros(pad, dtype=np.uint8)])
    reshaped = idx.reshape(-1, 8)
    b0 = (reshaped[:, 0] << 5) | (reshaped[:, 1] << 2) | (reshaped[:, 2] >> 1)
    b1 = (reshaped[:, 2] << 7) | (reshaped[:, 3] << 4) | (reshaped[:, 4] << 1) | (reshaped[:, 5] >> 2)
    b2 = (reshaped[:, 5] << 6) | (reshaped[:, 6] << 3) | (reshaped[:, 7])
    return np.stack([b0, b1, b2], axis=1).flatten()

def unpack_3bit(packed: np.ndarray, length: int) -> Tensor:
    reshaped = packed.reshape(-1, 3)
    b0, b1, b2 = reshaped[:, 0], reshaped[:, 1], reshaped[:, 2]
    idx = np.zeros((len(reshaped), 8), dtype=np.uint8)
    idx[:, 0] = (b0 >> 5) & 0x7
    idx[:, 1] = (b0 >> 2) & 0x7
    idx[:, 2] = ((b0 << 1) & 0x6) | ((b1 >> 7) & 0x1)
    idx[:, 3] = (b1 >> 4) & 0x7
    idx[:, 4] = (b1 >> 1) & 0x7
    idx[:, 5] = ((b1 << 2) & 0x4) | ((b2 >> 6) & 0x3)
    idx[:, 6] = (b2 >> 3) & 0x7
    idx[:, 7] = b2 & 0x7
    return torch.from_numpy(idx.flatten()[:length])

# -----------------------------
# OPTIMIZER (ENTROPY-CONSTRAINED MUON)
# -----------------------------

def zeropower_via_newtonschulz5(G: Tensor, steps: int = 5, eps: float = 1e-7) -> Tensor:
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    X /= X.norm() + eps
    if G.size(0) > G.size(1): X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    return X.T if G.size(0) > G.size(1) else X

class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr, momentum, ect_lambda=1e-4):
        super().__init__(params, dict(lr=lr, momentum=momentum, ect_lambda=ect_lambda))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr, mom, ect = group["lr"], group["momentum"], group["ect_lambda"]
            for p in group["params"]:
                if p.grad is None: continue
                g = p.grad
                # ECT: Shannon Entropy Penalty (Pre-quantization prep)
                g = g + ect * torch.sign(g) * torch.log1p(g.abs())
                state = self.state[p]
                if "mom" not in state: state["mom"] = torch.zeros_like(g)
                buf = state["mom"]
                buf.mul_(mom).add_(g)
                g = g.add(buf, alpha=mom)
                g = zeropower_via_newtonschulz5(g)
                g *= max(1, g.size(0) / g.size(1)) ** 0.5
                p.add_(g, alpha=-lr)

# -----------------------------
# MODEL (BASELINE ARCHITECTURE)
# -----------------------------

class RMSNorm(nn.Module):
    def forward(self, x): return F.rms_norm(x, (x.size(-1),), eps=1e-5)

class CastedLinear(nn.Linear):
    def forward(self, x): return F.linear(x, self.weight.to(x.dtype), self.bias.to(x.dtype) if self.bias is not None else None)

class Block(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.attn_norm = RMSNorm()
        self.mlp_norm = RMSNorm()
        self.c_q = CastedLinear(args.model_dim, args.model_dim, bias=False)
        self.c_k = CastedLinear(args.model_dim, (args.num_kv_heads * args.model_dim // args.num_heads), bias=False)
        self.c_v = CastedLinear(args.model_dim, (args.num_kv_heads * args.model_dim // args.num_heads), bias=False)
        self.proj = CastedLinear(args.model_dim, args.model_dim, bias=False)
        self.fc = CastedLinear(args.model_dim, args.model_dim * args.mlp_mult, bias=False)
        self.mlp_proj = CastedLinear(args.model_dim * args.mlp_mult, args.model_dim, bias=False)
        self.num_heads = args.num_heads
        self.num_kv_heads = args.num_kv_heads
        self.head_dim = args.model_dim // args.num_heads

    def forward(self, x):
        bsz, seqlen, _ = x.shape
        # Attention
        r = x
        x = self.attn_norm(x)
        q = self.c_q(x).view(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.c_k(x).view(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.c_v(x).view(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).reshape(bsz, seqlen, -1)
        x = r + self.proj(y)
        # MLP
        r = x
        x = self.mlp_norm(x)
        x = r + self.mlp_proj(F.relu(self.fc(x)))
        return x

class GPT(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.tok_emb = nn.Embedding(args.vocab_size, args.model_dim)
        self.blocks = nn.ModuleList([Block(args) for _ in range(args.num_layers)])
        self.final_norm = RMSNorm()

    def forward(self, x, y):
        x = self.tok_emb(x)
        for b in self.blocks: x = b(x)
        logits = F.linear(self.final_norm(x), self.tok_emb.weight)
        logits = 30.0 * torch.tanh(logits / 30.0)
        return F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))

# -----------------------------
# DATA LOADING
# -----------------------------

def load_data_shard(file: Path) -> Tensor:
    header = np.fromfile(file, dtype="<i4", count=256)
    num_tokens = int(header[2])
    tokens_np = np.fromfile(file, dtype="<u2", count=num_tokens, offset=256*4)
    return torch.from_numpy(tokens_np.astype(np.uint16, copy=False))

class TokenStream:
    def __init__(self, pattern: str):
        self.files = [Path(p) for p in sorted(glob.glob(pattern))]
        self.file_idx = 0
        self.tokens = load_data_shard(self.files[0])
        self.pos = 0

    def take(self, n: int) -> Tensor:
        chunks = []
        while n > 0:
            avail = self.tokens.numel() - self.pos
            if avail <= 0:
                self.file_idx = (self.file_idx + 1) % len(self.files)
                self.tokens = load_data_shard(self.files[self.file_idx])
                self.pos = 0
                continue
            k = min(n, avail)
            chunks.append(self.tokens[self.pos : self.pos + k])
            self.pos += k
            n -= k
        return torch.cat(chunks) if len(chunks) > 1 else chunks[0]

# -----------------------------
# MAIN
# -----------------------------

def main():
    args = Hyperparameters()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = GPT(args).to(device).bfloat16()
    for m in model.modules():
        if isinstance(m, CastedLinear): m.float()

    # Optimizer
    matrix_params = [p for p in model.parameters() if p.ndim == 2]
    scalar_params = [p for p in model.parameters() if p.ndim < 2]
    optimizer_muon = Muon(matrix_params, args.matrix_lr, args.muon_momentum, args.ect_lambda)
    optimizer_adam = torch.optim.Adam(scalar_params, lr=args.scalar_lr, betas=(args.beta1, args.beta2), eps=args.adam_eps)

    # Simple training loop for 200 iterations
    train_loader = TokenStream(args.train_files)
    t0 = time.perf_counter()
    for step in range(args.iterations):
        chunk = train_loader.take(args.train_batch_tokens + 1).to(device, dtype=torch.long)
        x, y = chunk[:-1].view(-1, args.train_seq_len), chunk[1:].view(-1, args.train_seq_len)
        
        with torch.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu", dtype=torch.bfloat16):
            loss = model(x, y)
        
        optimizer_muon.zero_grad()
        optimizer_adam.zero_grad()
        loss.backward()
        optimizer_muon.step()
        optimizer_adam.step()
        
        if step % 10 == 0:
            print(f"step {step} loss {loss.item():.4f} time {time.perf_counter()-t0:.2f}s")

    # -----------------------------
    # SOTA EXPORT
    # -----------------------------
    print("Starting SOTA Compression...")
    sd = model.state_dict()
    processed = {}
    for name, t in sd.items():
        if t.ndim != 2 or t.numel() < 4096:
            processed[name] = {"type": "raw", "data": t.half().cpu()}
            continue
        w = t.float().clone()
        fwht_inplace(w)
        w_flat = w.flatten()
        centroids = w_flat[torch.randperm(w_flat.size(0))[:8]].clone()
        for _ in range(5):
            dists = (w_flat.unsqueeze(1) - centroids.unsqueeze(0)).abs()
            idx = dists.argmin(dim=1)
            for i in range(8):
                if (idx == i).any(): centroids[i] = w_flat[idx == i].mean()
        delta = torch.zeros_like(idx, dtype=torch.uint8)
        delta[0], delta[1:] = idx[0], (idx[1:] - idx[:-1]) % 8
        processed[name] = {"type": "3bit_delta", "shape": t.shape, "centroids": centroids.half().cpu(), "data": pack_3bit(delta)}

    buffer = io.BytesIO()
    torch.save(processed, buffer)
    if _COMPRESSOR == "zstd":
        compressed = zstd.ZstdCompressor(level=1).compress(buffer.getvalue())
    else:
        compressed = zlib.compress(buffer.getvalue(), 9)
    with open("final_model.int8.ptz", "wb") as f: f.write(compressed)
    print(f"Export Complete: {len(compressed)/1024/1024:.2f} MB")

if __name__ == "__main__":
    main()
