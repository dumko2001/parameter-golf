"""
exp_1_compression.py: The "Parallel Powerhouse" configuration.
Architecture: 10L, 4x MLP, SeqLen 512, BigramHash(24k), SmearGate, Full Attention Residuals.
Compression: 3-bit K-Means Delta + Bit-Packing + Zstd Level 1.
"""

from __future__ import annotations

import copy
import glob
import io
import math
import os
import random
import subprocess
import sys
import time
import uuid
from pathlib import Path

import numpy as np
import sentencepiece as spm
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP
import zstandard as zstd

# -----------------------------
# HYPERPARAMETERS
# -----------------------------

class Hyperparameters:
    data_path = os.environ.get("DATA_PATH", "./data/datasets/fineweb10B_sp1024")
    train_files = os.path.join(data_path, "fineweb_train_*.bin")
    val_files = os.path.join(data_path, "fineweb_val_*.bin")
    tokenizer_path = os.environ.get("TOKENIZER_PATH", "./data/tokenizers/fineweb_1024_bpe.model")
    run_id = os.environ.get("RUN_ID", str(uuid.uuid4()))
    seed = int(os.environ.get("SEED", 1337))

    val_batch_size = int(os.environ.get("VAL_BATCH_SIZE", 524_288))
    val_loss_every = int(os.environ.get("VAL_LOSS_EVERY", 100))
    train_log_every = int(os.environ.get("TRAIN_LOG_EVERY", 20))

    iterations = int(os.environ.get("ITERATIONS", 20000))
    warmdown_iters = int(os.environ.get("WARMDOWN_ITERS", 3000))
    warmup_steps = int(os.environ.get("WARMUP_STEPS", 20))
    train_batch_tokens = int(os.environ.get("TRAIN_BATCH_TOKENS", 524_288))
    train_seq_len = int(os.environ.get("TRAIN_SEQ_LEN", 512)) # Speed Booster
    max_wallclock_seconds = float(os.environ.get("MAX_WALLCLOCK_SECONDS", 600.0))
    qk_gain_init = float(os.environ.get("QK_GAIN_INIT", 1.5))

    # Model shape (~48M parameters)
    vocab_size = int(os.environ.get("VOCAB_SIZE", 1024))
    num_layers = 10 # Corrected from 18 to reduce Sequential Tax
    num_kv_heads = 4
    model_dim = 512
    num_heads = 8
    mlp_mult = 4 # Corrected to 4 for Parallel IQ
    tie_embeddings = True
    rope_base = 10000.0
    logit_softcap = 30.0

    # Components
    bigram_vocab_size = 24576 # Golden Ratio for 3-bit space
    bigram_dim = 128

    # Optimizer
    matrix_lr = 0.04
    scalar_lr = 0.04
    muon_momentum = 0.95
    ect_lambda = 1e-4  # Entropy-Constrained Training penalty
    weight_decay = 0.04 # High WD for better quantization range
    beta1, beta2 = 0.9, 0.95
    adam_eps = 1e-8

# -----------------------------
# SOTA COMPRESSION UTILITIES
# -----------------------------

def fwht_inplace(x: Tensor):
    """In-place Fast Walsh-Hadamard Transform."""
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
    """Packs eight 3-bit indices into three 8-bit bytes."""
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
    b0, b1, b2 = reshaped[:, 0].astype(np.uint8), reshaped[:, 1].astype(np.uint8), reshaped[:, 2].astype(np.uint8)
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
# MUON OPTIMIZER
# -----------------------------

def zeropower_via_newtonschulz5(G: Tensor, steps: int = 10, eps: float = 1e-7) -> Tensor:
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    X /= X.norm() + eps
    transposed = G.size(0) > G.size(1)
    if transposed: X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    return X.T if transposed else X

class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr, momentum, ect_lambda=1e-4, weight_decay=0.04):
        super().__init__(params, dict(lr=lr, momentum=momentum, ect_lambda=ect_lambda, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr, mom, ect, wd = group["lr"], group["momentum"], group["ect_lambda"], group["weight_decay"]
            for p in group["params"]:
                if p.grad is None: continue
                g = p.grad
                # ECT: Shannon Entropy Penalty
                g = g + ect * torch.sign(g) * torch.log1p(g.abs())
                state = self.state[p]
                if "momentum_buffer" not in state: state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(mom).add_(g)
                g = g.add(buf, alpha=mom)
                g = zeropower_via_newtonschulz5(g)
                g *= max(1, g.size(0) / g.size(1)) ** 0.5
                if wd > 0: p.data.mul_(1.0 - lr * wd)
                p.add_(g, alpha=-lr)

# -----------------------------
# TRANSFORMER MODULES
# -----------------------------

class RMSNorm(nn.Module):
    def forward(self, x): return F.rms_norm(x, (x.size(-1),), eps=1e-5)

class CastedLinear(nn.Linear):
    def forward(self, x): return F.linear(x, self.weight.to(x.dtype), self.bias.to(x.dtype) if self.bias is not None else None)

class SmearGate(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gate = nn.Parameter(torch.zeros(dim))
    def forward(self, x):
        g = torch.sigmoid(self.gate.to(dtype=x.dtype))[None, None, :]
        x_prev = torch.cat([torch.zeros_like(x[:, :1]), x[:, :-1]], dim=1)
        return (1 - g) * x + g * x_prev

class BigramHashEmbedding(nn.Module):
    def __init__(self, vocab_size, dim, model_dim):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed = nn.Embedding(vocab_size, dim)
        self.proj = CastedLinear(dim, model_dim, bias=False) if dim != model_dim else None
        self.scale = nn.Parameter(torch.tensor(0.05))
    def forward(self, x):
        t = x.to(torch.int32)
        mod = self.vocab_size - 1
        h = torch.empty_like(t)
        h[..., 0] = mod
        h[..., 1:] = torch.bitwise_xor(36313 * t[..., 1:], 27191 * t[..., :-1]) % mod
        e = self.embed(h.long())
        if self.proj: e = self.proj(e)
        return e * self.scale.to(dtype=e.dtype)

class AttentionResidual(nn.Module):
    """Implementation of Full Attention Residuals (arXiv:2603.15031)"""
    def __init__(self, dim):
        super().__init__()
        self.query = nn.Parameter(torch.zeros(dim)) # Init to 0 for uniform start
        self.norm = RMSNorm()
    def forward(self, history: list[Tensor]) -> Tensor:
        V = torch.stack(history) # [num_sources, B, T, D]
        K = self.norm(V)
        # Compute softmax weights across depth
        logits = torch.einsum('d, n b t d -> n b t', self.query, K)
        weights = logits.softmax(dim=0)
        return torch.einsum('n b t, n b t d -> b t d', weights, V)

class Block(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.attn_norm, self.mlp_norm = RMSNorm(), RMSNorm()
        self.c_q = CastedLinear(args.model_dim, args.model_dim, bias=False)
        self.c_k = CastedLinear(args.model_dim, (args.num_kv_heads * args.model_dim // args.num_heads), bias=False)
        self.c_v = CastedLinear(args.model_dim, (args.num_kv_heads * args.model_dim // args.num_heads), bias=False)
        self.proj = CastedLinear(args.model_dim, args.model_dim, bias=False)
        self.fc = CastedLinear(args.model_dim, args.model_dim * args.mlp_mult, bias=False)
        self.mlp_proj = CastedLinear(args.model_dim * args.mlp_mult, args.model_dim, bias=False)
        
        # Attention Residual Controllers
        self.attn_res = AttentionResidual(args.model_dim)
        self.mlp_res = AttentionResidual(args.model_dim)
        
        self.num_heads, self.num_kv_heads = args.num_heads, args.num_kv_heads
        self.head_dim = args.model_dim // args.num_heads

    def forward(self, history: list[Tensor]):
        # 1. Attention Step
        h = self.attn_res(history)
        x = self.attn_norm(h)
        q = self.c_q(x).view(x.size(0), x.size(1), self.num_heads, self.head_dim).transpose(1, 2)
        k = self.c_k(x).view(x.size(0), x.size(1), self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.c_v(x).view(x.size(0), x.size(1), self.num_kv_heads, self.head_dim).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).reshape(x.size(0), x.size(1), -1)
        out_attn = self.proj(y)
        history.append(out_attn)
        
        # 2. MLP Step
        h = self.mlp_res(history)
        x = self.mlp_norm(h)
        out_mlp = self.mlp_proj(F.relu(self.fc(x)).square())
        history.append(out_mlp)
        return history

class GPT(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.tok_emb = nn.Embedding(args.vocab_size, args.model_dim)
        self.bigram = BigramHashEmbedding(args.bigram_vocab_size, args.bigram_dim, args.model_dim)
        self.smear = SmearGate(args.model_dim)
        self.blocks = nn.ModuleList([Block(args) for _ in range(args.num_layers)])
        self.final_norm = RMSNorm()
        self._init_weights()

    def _init_weights(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                if ".proj" in name or name.endswith("proj"):
                    with torch.no_grad(): m.weight.mul_(1.0 / math.sqrt(2 * len(self.blocks)))

    def forward(self, x, y):
        e = self.tok_emb(x) + self.bigram(x)
        x = self.smear(F.rms_norm(e, (e.size(-1),)))
        
        history = [x]
        for b in self.blocks:
            history = b(history)
            
        # Final aggregation (Full AttnRes style)
        x = self.final_norm(history[-1])
        logits = F.linear(x, self.tok_emb.weight)
        logits = 30.0 * torch.tanh(logits / 30.0)
        return F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))

# -----------------------------
# DATA & MAIN PIPELINE
# -----------------------------

def load_data_shard(file):
    header = np.fromfile(file, dtype="<i4", count=256)
    num_tokens = int(header[2])
    tokens_np = np.fromfile(file, dtype="<u2", count=num_tokens, offset=256*4)
    return torch.from_numpy(tokens_np.astype(np.uint16, copy=False))

class TokenStream:
    def __init__(self, pattern):
        self.files = [Path(p) for p in sorted(glob.glob(pattern))]
        self.file_idx, self.pos = 0, 0
        self.tokens = load_data_shard(self.files[0])
    def take(self, n):
        chunks = []
        while n > 0:
            avail = self.tokens.numel() - self.pos
            if avail <= 0:
                self.file_idx = (self.file_idx + 1) % len(self.files); self.tokens = load_data_shard(self.files[self.file_idx]); self.pos = 0
                continue
            k = min(n, avail); chunks.append(self.tokens[self.pos : self.pos + k]); self.pos += k; n -= k
        return torch.cat(chunks) if len(chunks) > 1 else chunks[0]

def main():
    args = Hyperparameters()
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    device = torch.device("cuda", rank % torch.cuda.device_count()) if torch.cuda.is_available() else "cpu"
    if world_size > 1: dist.init_process_group(backend="nccl")
    
    random.seed(args.seed + rank); np.random.seed(args.seed + rank); torch.manual_seed(args.seed + rank)
    
    model = GPT(args).to(device).bfloat16()
    for m in model.modules():
        if isinstance(m, CastedLinear): m.float()
    
    matrix_params = [p for n, p in model.named_parameters() if p.ndim == 2 and "embed" not in n]
    scalar_params = [p for n, p in model.named_parameters() if p.ndim < 2 or "embed" in n]
    optimizer_muon = Muon(matrix_params, args.matrix_lr, args.muon_momentum, args.ect_lambda, args.weight_decay)
    optimizer_adam = torch.optim.AdamW(scalar_params, lr=args.scalar_lr, betas=(args.beta1, args.beta2), eps=args.adam_eps)

    train_loader = TokenStream(args.train_files)
    t0 = time.perf_counter()
    for step in range(args.iterations):
        chunk = train_loader.take(args.train_batch_tokens + 1).to(device, dtype=torch.long)
        x, y = chunk[:-1].view(-1, args.train_seq_len), chunk[1:].view(-1, args.train_seq_len)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16): loss = model(x, y)
        optimizer_muon.zero_grad(); optimizer_adam.zero_grad(); loss.backward(); optimizer_muon.step(); optimizer_adam.step()
        if rank == 0 and step % 20 == 0:
            elapsed = time.perf_counter() - t0
            print(f"step {step} loss {loss.item():.4f} time {elapsed:.1f}s")
            if elapsed > args.max_wallclock_seconds: break

    if rank == 0:
        print("Starting SOTA Export...")
        sd = model.state_dict(); processed = {}
        for name, t in sd.items():
            if t.ndim != 2 or t.numel() < 4096:
                processed[name] = {"type": "raw", "data": t.half().cpu()}
                continue
            w = t.float().clone().cpu(); fwht_inplace(w); w_flat = w.flatten()
            centroids = w_flat[torch.randperm(w_flat.size(0))[:8]].clone()
            for _ in range(10):
                dists = (w_flat.unsqueeze(1) - centroids.unsqueeze(0)).abs(); idx = dists.argmin(dim=1)
                for i in range(8):
                    if (idx == i).any(): centroids[i] = w_flat[idx == i].mean()
            delta = torch.zeros_like(idx, dtype=torch.uint8); delta[0], delta[1:] = idx[0], (idx[1:] - idx[:-1]) % 8
            processed[name] = {"type": "3bit_delta", "shape": t.shape, "centroids": centroids.half().cpu(), "data": pack_3bit(delta)}
        
        buffer = io.BytesIO(); torch.save(processed, buffer)
        compressed = zstd.ZstdCompressor(level=1).compress(buffer.getvalue())
        with open("final_model.int8.ptz", "wb") as f: f.write(compressed)
        print(f"Final Artifact: {len(compressed)/1024/1024:.2f} MB")

    if world_size > 1: dist.destroy_process_group()

if __name__ == "__main__":
    main()
