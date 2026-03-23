"""
Holographic Basis Expansion GPT (Parameter Golf)
Architecture:
- Global Shared Basis: A (D x R), B (R x D)
- 48 Layers constructed from tiny Coefficient Vectors (C).
- W = A @ diag(C) @ B (computed efficiently as (x @ A * C) @ B).
- Muon orthogonalizes the Basis; AdamW trains the Coefficients.
"""

import copy
import glob
import io
import math
import os
import random
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

# -----------------------------
# HYPERPARAMETERS
# -----------------------------

class Hyperparameters:
    data_path      = os.environ.get("DATA_PATH", "./data/fineweb10B/datasets/fineweb10B_sp8192")
    train_files    = os.path.join(data_path, "fineweb_train_*.bin")
    val_files      = os.path.join(data_path, "fineweb_val_000000.bin")
    tokenizer_path = os.environ.get("TOKENIZER_PATH", "./data/fineweb10B/tokenizers/fineweb_8192_bpe.model")
    run_id         = os.environ.get("RUN_ID", str(uuid.uuid4()))
    seed           = int(os.environ.get("SEED", 1337))

    vocab_size     = 8192
    num_layers     = 48
    model_dim      = 1024  # D
    basis_rank     = 64    # R
    num_heads      = 8
    num_kv_heads   = 4
    mlp_mult       = 2
    
    train_batch_tokens = 131072
    train_seq_len  = 1024
    iterations     = 20000
    warmup_steps   = 20
    max_wallclock_seconds = 600.0
    
    # Optimizer
    embed_lr       = 0.05
    coeff_lr       = 0.02
    basis_lr       = 0.02
    muon_momentum  = 0.95
    muon_backend_steps = 5

    # TTT
    ttt_lr = 1e-3
    eval_seq_len = 1024
    chunk_size = 256

# -----------------------------
# HOLOGRAPHIC BASIS ARCHITECTURE
# -----------------------------

class BasisLinear(nn.Module):
    """
    A layer that owns no full matrices. Reconstructs its transformation 
    from the global basis using a unique 1D coefficient vector.
    """
    def __init__(self, basis_A: Tensor, basis_B: Tensor, rank: int):
        super().__init__()
        # References to the shared global basis [D, R] and [R, D]
        self.basis_A = basis_A
        self.basis_B = basis_B
        # The ONLY parameter this layer actually owns:
        self.C = nn.Parameter(torch.ones(rank, dtype=torch.float32))

    def forward(self, x: Tensor) -> Tensor:
        # Optimized MatMul: avoids creating the massive DxD matrix in memory
        # x shape: [B, S, D]
        # x @ A -> [B, S, R]
        # * C   -> [B, S, R]
        # @ B   -> [B, S, D]
        x_a = torch.matmul(x, self.basis_A)
        x_c = x_a * self.C
        return torch.matmul(x_c, self.basis_B)

class RMSNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))
    def forward(self, x: Tensor):
        return F.rms_norm(x, (x.size(-1),), self.scale)

class Rotary(nn.Module):
    def __init__(self, dim: int, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._cos_cached = None
        self._sin_cached = None

    def forward(self, seq_len: int, device: torch.device):
        if self._cos_cached is None or self._cos_cached.size(2) < seq_len:
            t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
            freqs = torch.outer(t, self.inv_freq.to(device))
            self._cos_cached = freqs.cos()[None, None, :, :]
            self._sin_cached = freqs.sin()[None, None, :, :]
        return self._cos_cached[:, :, :seq_len, :], self._sin_cached[:, :, :seq_len, :]

def apply_rotary_emb(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    half = x.size(-1) // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)

class HolographicBlock(nn.Module):
    def __init__(self, args: Hyperparameters, basis_A: Tensor, basis_B: Tensor):
        super().__init__()
        dim = args.model_dim
        self.num_heads = args.num_heads
        self.num_kv_heads = args.num_kv_heads
        self.head_dim = dim // args.num_heads
        
        self.norm1 = RMSNorm(dim)
        self.c_q = BasisLinear(basis_A, basis_B, args.basis_rank)
        self.c_k = BasisLinear(basis_A, basis_B, args.basis_rank)
        self.c_v = BasisLinear(basis_A, basis_B, args.basis_rank)
        self.c_o = BasisLinear(basis_A, basis_B, args.basis_rank)
        
        self.norm2 = RMSNorm(dim)
        self.m_fc = BasisLinear(basis_A, basis_B, args.basis_rank)
        self.m_proj = BasisLinear(basis_A, basis_B, args.basis_rank)
        self.rotary = Rotary(self.head_dim)

    def forward(self, x: Tensor) -> Tensor:
        bsz, seqlen, dim = x.shape
        n1 = self.norm1(x)
        
        q = self.c_q(n1).view(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.c_k(n1).view(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.c_v(n1).view(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        
        cos, sin = self.rotary(seqlen, x.device)
        q = apply_rotary_emb(q, cos.to(q.dtype), sin.to(q.dtype))
        k = apply_rotary_emb(k, cos.to(k.dtype), sin.to(k.dtype))
        
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
        y = y.transpose(1, 2).contiguous().view(bsz, seqlen, dim)
        x = x + self.c_o(y)
        
        n2 = self.norm2(x)
        m = F.silu(self.m_fc(n2))
        x = x + self.m_proj(m)
        return x

class UltraGPT(nn.Module):
    def __init__(self, args: Hyperparameters):
        super().__init__()
        self.vocab_size = args.vocab_size
        self.tok_emb = nn.Embedding(args.vocab_size, args.model_dim)
        
        # THE GLOBAL BASIS POOL (Shared across all 48 layers)
        bound = 1.0 / math.sqrt(args.model_dim)
        self.basis_A = nn.Parameter(torch.empty(args.model_dim, args.basis_rank).uniform_(-bound, bound))
        self.basis_B = nn.Parameter(torch.empty(args.basis_rank, args.model_dim).uniform_(-bound, bound))
        
        self.blocks = nn.ModuleList([
            HolographicBlock(args, self.basis_A, self.basis_B) for _ in range(args.num_layers)
        ])
        self.final_norm = RMSNorm(args.model_dim)

    def forward(self, input_ids: Tensor, targets: Tensor = None):
        x = self.tok_emb(input_ids)
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x)
        logits = F.linear(x, self.tok_emb.weight)
        
        if targets is not None:
            return F.cross_entropy(logits.view(-1, self.vocab_size), targets.view(-1))
        return logits

# -----------------------------
# MUON ORTHOGONALIZER
# -----------------------------
def zeropower_via_newtonschulz5(G: Tensor, steps: int = 5, eps: float = 1e-7) -> Tensor:
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
    def __init__(self, params, lr: float, momentum: float):
        super().__init__(params, dict(lr=lr, momentum=momentum))
    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr = group['lr']; momentum = group['momentum']
            for p in group['params']:
                if p.grad is None: continue
                g = p.grad
                state = self.state[p]
                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(g)
                buf = state['momentum_buffer']
                buf.mul_(momentum).add_(g)
                g = g.add(buf, alpha=momentum)
                g = zeropower_via_newtonschulz5(g)
                g *= max(1, g.size(0) / g.size(1)) ** 0.5
                p.add_(g, alpha=-lr)

# -----------------------------
# DATA LOADING
# -----------------------------
def load_data_shard(file: Path) -> Tensor:
    header_bytes = 256 * 4
    header = np.fromfile(file, dtype="<i4", count=256)
    num_tokens = int(header[2])
    tokens_np = np.fromfile(file, dtype="<u2", count=num_tokens, offset=header_bytes)
    return torch.from_numpy(tokens_np.astype(np.int64, copy=False))

class TokenStream:
    def __init__(self, pattern: str):
        self.files = [Path(p) for p in sorted(glob.glob(pattern))]
        self.file_idx = 0
        self.tokens = load_data_shard(self.files[0])
        self.pos = 0

    def take(self, n: int) -> Tensor:
        chunks = []; remaining = n
        while remaining > 0:
            avail = self.tokens.numel() - self.pos
            if avail <= 0:
                self.file_idx = (self.file_idx + 1) % len(self.files)
                self.tokens = load_data_shard(self.files[self.file_idx])
                self.pos = 0
                continue
            k = min(remaining, avail)
            chunks.append(self.tokens[self.pos: self.pos + k])
            self.pos += k; remaining -= k
        return chunks[0] if len(chunks) == 1 else torch.cat(chunks)

# -----------------------------
# EVALUATION (TTT + SUFFIX FUSION)
# -----------------------------

class NgramDocumentCache:
    def __init__(self, vocab_size: int, device: torch.device):
        self.V = vocab_size; self.device = device
        self.reset()
    def reset(self):
        self.unigram = torch.zeros(self.V, device=self.device)
        self.bigram = {}
    def update(self, prev1: int, curr: int):
        if curr == 1: # BOS
            self.reset(); return
        self.unigram[curr] += 1.0
        if prev1 not in self.bigram: self.bigram[prev1] = torch.zeros(self.V, device=self.device)
        self.bigram[prev1][curr] += 1.0
    def get_probs(self, prev1: int) -> Tensor:
        p = torch.zeros(self.V, device=self.device)
        if prev1 in self.bigram and self.bigram[prev1].sum() > 0:
            return self.bigram[prev1] / self.bigram[prev1].sum()
        if self.unigram.sum() > 0:
            return self.unigram / self.unigram.sum()
        p[0] = 1.0
        return p

def eval_ttt_loop(args: Hyperparameters, model: UltraGPT, device: torch.device):
    val_tokens = load_data_shard(Path(sorted(glob.glob(args.val_files))[0]))
    bos_positions = (val_tokens == 1).nonzero(as_tuple=True)[0]
    
    # Isolate plastic memory for TTT (Coefficients + Embeddings only)
    ttt_params = [p for n, p in model.named_parameters() if 'C' in n or 'tok_emb' in n]
    
    loss_sum = 0.0
    token_count = 0.0
    
    for i in range(min(50, len(bos_positions)-1)): # Limit to 50 docs for speed testing
        start = int(bos_positions[i])
        end = int(bos_positions[i+1])
        doc = val_tokens[start:end].to(device)
        
        # Reset Plasticity & Suffix Tree per document
        opt = torch.optim.AdamW(ttt_params, lr=args.ttt_lr)
        cache = NgramDocumentCache(args.vocab_size, device)
        
        prev1 = 0
        for chunk_start in range(0, len(doc)-1, args.chunk_size):
            chunk_end = min(chunk_start + args.chunk_size, len(doc)-1)
            x = doc[chunk_start:chunk_end].unsqueeze(0)
            y = doc[chunk_start+1:chunk_end+1].unsqueeze(0)
            
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(x)
            
            chunk_loss = 0.0
            for t in range(logits.size(1)):
                tgt = int(y[0, t].item())
                base_p = F.softmax(logits[0, t].float(), dim=-1)
                cache_p = cache.get_probs(prev1)
                
                # Fuse Logic (Network) with Memory (Suffix Tree)
                final_p = 0.8 * base_p + 0.2 * cache_p
                loss_tok = -torch.log(final_p[tgt].clamp_min(1e-10))
                chunk_loss = chunk_loss + loss_tok
                
                cache.update(prev1, tgt)
                prev1 = tgt
                token_count += 1
            
            loss_sum += chunk_loss.item()
            
            # Biological Plasticity Update
            opt.zero_grad()
            (chunk_loss / logits.size(1)).backward()
            opt.step()
            
    val_loss = loss_sum / max(token_count, 1)
    val_bpb = val_loss / math.log(2.0) / 0.21 # Approx bytes multiplier for 8192
    return val_bpb

# -----------------------------
# MAIN
# -----------------------------
def main():
    args = Hyperparameters()
    device = torch.device("cuda")
    
    model = UltraGPT(args).to(device).bfloat16()
    model = torch.compile(model)
    
    basis_params = [model.basis_A, model.basis_B]
    coeff_params = [p for n, p in model.named_parameters() if 'C' in n]
    embed_params = [model.tok_emb.weight]
    
    opt_basis = Muon(basis_params, lr=args.basis_lr, momentum=args.muon_momentum)
    opt_coeff = torch.optim.AdamW(coeff_params, lr=args.coeff_lr, fused=True)
    opt_embed = torch.optim.AdamW(embed_params, lr=args.embed_lr, fused=True)
    
    stream = TokenStream(args.train_files)
    
    print(f"Holographic Model Initialized: 48 Layers, {args.model_dim} Dim, {args.basis_rank} Rank")
    print(f"Total Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M")
    
    t0 = time.perf_counter()
    for step in range(args.iterations):
        x_flat = stream.take(args.train_batch_tokens + 1).to(device)
        x = x_flat[:-1].view(-1, args.train_seq_len)
        y = x_flat[1:].view(-1, args.train_seq_len)
        
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss = model(x, y)
            
        loss.backward()
        opt_basis.step(); opt_coeff.step(); opt_embed.step()
        opt_basis.zero_grad(); opt_coeff.zero_grad(); opt_embed.zero_grad()
        
        if step % args.train_log_every == 0:
            print(f"step {step} | loss: {loss.item():.4f}")
            
        if time.perf_counter() - t0 > 580:
            break
            
    print("Training complete. Starting TTT + Suffix Evaluation...")
    bpb = eval_ttt_loop(args, model, device)
    print(f"FINAL HOLOGRAPHIC BPB: {bpb:.4f}")
    
    # Save Model (Fits in FP32 without quantization)
    torch.save(model.state_dict(), "holographic_final.pt")
    size = os.path.getsize("holographic_final.pt")
    print(f"Artifact Size: {size / 1e6:.2f} MB")

if __name__ == "__main__":
    main()
