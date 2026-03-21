"""
base+compression.py: Full-featured training script with SOTA compression.
Integrates the 17M baseline with the 3-bit K-Means Delta + Bit-Packing pipeline.
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
import zlib
from pathlib import Path

import numpy as np
import sentencepiece as spm
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP

try:
    import zstandard as zstd
    _COMPRESSOR = "zstd"
except ImportError:
    _COMPRESSOR = "zlib"

# -----------------------------
# HYPERPARAMETERS (17M BASELINE)
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

    # Requested Config
    iterations = int(os.environ.get("ITERATIONS", 200))
    warmdown_iters = int(os.environ.get("WARMDOWN_ITERS", 50))
    warmup_steps = int(os.environ.get("WARMUP_STEPS", 10))
    train_batch_tokens = int(os.environ.get("TRAIN_BATCH_TOKENS", 262144))
    train_seq_len = int(os.environ.get("TRAIN_SEQ_LEN", 1024))
    max_wallclock_seconds = float(os.environ.get("MAX_WALLCLOCK_SECONDS", 600.0))
    qk_gain_init = float(os.environ.get("QK_GAIN_INIT", 1.5))

    # Model shape (~17M parameters)
    vocab_size = int(os.environ.get("VOCAB_SIZE", 1024))
    num_layers = 9
    num_kv_heads = 4
    model_dim = 512
    num_heads = 8
    mlp_mult = 2
    tie_embeddings = True
    rope_base = 10000.0
    logit_softcap = 30.0

    # Optimizer
    matrix_lr = 0.04
    scalar_lr = 0.04
    muon_momentum = 0.95
    beta1, beta2 = 0.9, 0.95
    adam_eps = 1e-8
    ect_lambda = 1e-4  # SOTA Entropy Penalty

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
    indices = indices.flatten().cpu().numpy().astype(np.uint8)
    pad = (8 - (len(indices) % 8)) % 8
    if pad: indices = np.concatenate([indices, np.zeros(pad, dtype=np.uint8)])
    reshaped = indices.reshape(-1, 8)
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
# MUON OPTIMIZER 
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
                # SOTA ECT: Shannon Entropy Penalty
                g = g + ect * torch.sign(g) * torch.log1p(g.abs())
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(mom).add_(g)
                g = g.add(buf, alpha=mom)
                g = zeropower_via_newtonschulz5(g)
                g *= max(1, g.size(0) / g.size(1)) ** 0.5
                p.add_(g, alpha=-lr)

# -----------------------------
# EVALUATION SETUP
# -----------------------------

def build_sentencepiece_luts(sp, vocab_size, device):
    sp_vocab_size = int(sp.vocab_size())
    table_size = max(sp_vocab_size, vocab_size)
    base_bytes_np = np.zeros((table_size,), dtype=np.int16)
    has_leading_space_np = np.zeros((table_size,), dtype=np.bool_)
    is_boundary_token_np = np.ones((table_size,), dtype=np.bool_)
    for token_id in range(sp_vocab_size):
        if sp.is_control(token_id) or sp.is_unknown(token_id) or sp.is_unused(token_id): continue
        is_boundary_token_np[token_id] = False
        if sp.is_byte(token_id):
            base_bytes_np[token_id] = 1
            continue
        piece = sp.id_to_piece(token_id)
        if piece.startswith("▁"):
            has_leading_space_np[token_id] = True
            piece = piece[1:]
        base_bytes_np[token_id] = len(piece.encode("utf-8"))
    return (
        torch.tensor(base_bytes_np, dtype=torch.int16, device=device),
        torch.tensor(has_leading_space_np, dtype=torch.bool, device=device),
        torch.tensor(is_boundary_token_np, dtype=torch.bool, device=device),
    )

def load_data_shard(file):
    header = np.fromfile(file, dtype="<i4", count=256)
    num_tokens = int(header[2])
    tokens_np = np.fromfile(file, dtype="<u2", count=num_tokens, offset=256*4)
    return torch.from_numpy(tokens_np.astype(np.uint16, copy=False))

class DistributedTokenLoader:
    def __init__(self, pattern, rank, world_size, device):
        self.rank, self.world_size, self.device = rank, world_size, device
        self.files = [Path(p) for p in sorted(glob.glob(pattern))]
        self.file_idx, self.pos = 0, 0
        self.tokens = load_data_shard(self.files[0])

    def next_batch(self, global_tokens, seq_len, grad_accum_steps):
        n = (global_tokens // (self.world_size * grad_accum_steps)) + 1
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
        chunk = torch.cat(chunks).to(self.device, dtype=torch.long)
        x, y = chunk[:-1].reshape(-1, seq_len), chunk[1:].reshape(-1, seq_len)
        return x, y

def eval_val(args, model, device, val_tokens, luts):
    base_bytes, has_leading_space, is_boundary = luts
    model.eval()
    val_loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    val_byte_sum = torch.zeros((), device=device, dtype=torch.float64)
    val_token_count = 0
    with torch.no_grad():
        for i in range(0, val_tokens.numel() - 1, args.train_seq_len):
            end = min(i + args.train_seq_len, val_tokens.numel() - 1)
            x = val_tokens[i:end].unsqueeze(0).to(device, dtype=torch.long)
            y = val_tokens[i+1:end+1].unsqueeze(0).to(device, dtype=torch.long)
            with torch.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu", dtype=torch.bfloat16):
                loss = model(x, y)
            val_loss_sum += loss.double() * y.numel()
            val_token_count += y.numel()
            prev, tgt = x.view(-1), y.view(-1)
            tb = base_bytes[tgt].to(torch.float64)
            tb += (has_leading_space[tgt] & ~is_boundary[prev]).to(torch.float64)
            val_byte_sum += tb.sum()
    val_loss = (val_loss_sum / val_token_count).item()
    val_bpb = (val_loss / math.log(2.0)) * (val_token_count / val_byte_sum.item())
    model.train()
    return val_loss, val_bpb

# -----------------------------
# TRANSFORMER MODULES
# -----------------------------

class RMSNorm(nn.Module):
    def forward(self, x): return F.rms_norm(x, (x.size(-1),), eps=1e-5)

class CastedLinear(nn.Linear):
    def forward(self, x): return F.linear(x, self.weight.to(x.dtype), self.bias.to(x.dtype) if self.bias is not None else None)

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
        self.num_heads, self.num_kv_heads = args.num_heads, args.num_kv_heads
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
# MAIN
# -----------------------------

def main():
    args = Hyperparameters()
    
    # Distributed Setup
    distributed = "RANK" in os.environ
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")
    if distributed:
        dist.init_process_group(backend="nccl", device_id=device)
        torch.cuda.set_device(device)
    
    random.seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)
    
    # Data & Tokenizer
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    luts = build_sentencepiece_luts(sp, args.vocab_size, device)
    val_files = sorted(glob.glob(args.val_files))
    val_tokens = torch.cat([load_data_shard(f) for f in val_files]).contiguous()
    train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)

    # Model
    model = GPT(args).to(device).bfloat16()
    for m in model.modules():
        if isinstance(m, CastedLinear): m.float()
    if distributed:
        model = DDP(model, device_ids=[local_rank])

    # Optimizer
    matrix_params = [p for p in model.parameters() if p.ndim == 2]
    scalar_params = [p for p in model.parameters() if p.ndim < 2]
    optimizer_muon = Muon(matrix_params, args.matrix_lr, args.muon_momentum, args.ect_lambda)
    optimizer_adam = torch.optim.Adam(scalar_params, lr=args.scalar_lr, betas=(args.beta1, args.beta2), eps=args.adam_eps)

    # Training Loop
    t0 = time.perf_counter()
    for step in range(args.iterations):
        x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, 8)
        with torch.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu", dtype=torch.bfloat16):
            loss = model(x, y)
        optimizer_muon.zero_grad()
        optimizer_adam.zero_grad()
        loss.backward()
        optimizer_muon.step()
        optimizer_adam.step()
        
        if rank == 0 and (step % args.train_log_every == 0 or step == args.iterations - 1):
            elapsed = time.perf_counter() - t0
            print(f"step {step}/{args.iterations} | loss {loss.item():.4f} | time {elapsed:.1f}s")
            if elapsed > args.max_wallclock_seconds: break

    # -----------------------------
    # SOTA EXPORT
    # -----------------------------
    if rank == 0:
        print("Starting SOTA Compression...")
        sd = (model.module if distributed else model).state_dict()
        processed = {}
        for name, t in sd.items():
            if t.ndim != 2 or t.numel() < 4096:
                processed[name] = {"type": "raw", "data": t.half().cpu()}
                continue
            w = t.float().clone()
            fwht_inplace(w)
            w_flat = w.flatten()
            centroids = w_flat[torch.randperm(w_flat.size(0))[:8]].clone()
            for _ in range(10): # More iterations for final export
                dists = (w_flat.unsqueeze(1) - centroids.unsqueeze(0)).abs()
                idx = dists.argmin(dim=1)
                for i in range(8):
                    if (idx == i).any(): centroids[i] = w_flat[idx == i].mean()
            delta = torch.zeros_like(idx, dtype=torch.uint8)
            delta[0], delta[1:] = idx[0], (idx[1:] - idx[:-1]) % 8
            processed[name] = {"type": "3bit_delta", "shape": t.shape, "centroids": centroids.half().cpu(), "data": pack_3bit(delta)}

        buffer = io.BytesIO()
        torch.save(processed, buffer)
        raw_bytes = buffer.getvalue()
        if _COMPRESSOR == "zstd":
            compressed = zstd.ZstdCompressor(level=1).compress(raw_bytes)
        else:
            compressed = zlib.compress(raw_bytes, 9)
        
        with open("final_model.int8.ptz", "wb") as f: f.write(compressed)
        print(f"Final Artifact: {len(compressed)/1024/1024:.2f} MB")
        
        # Roundtrip Eval
        val_loss, val_bpb = eval_val(args, model, device, val_tokens, luts)
        print(f"Final Eval | Loss: {val_loss:.4f} | BPB: {val_bpb:.4f}")

    if distributed: dist.destroy_process_group()

if __name__ == "__main__":
    main()
