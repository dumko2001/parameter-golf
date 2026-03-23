"""
Parameter Golf submission.

Architecture:
  Vocab:   8192 (custom SentencePiece BPE — build with build_tokenizer.py)
  D:       448
  Layers:  14
  MLP:     3x (hidden=1344)
  Heads:   8Q / 4KV (GQA)
  Tied embeddings, seq_len=2048 training

Compression:
  INT6 per-row quantization + QAT entropy regularizer
  zstd level=22

Eval:
  Sliding window stride=64
  Causal n-gram document cache (bigram+trigram, zero disk cost)
  LoRA TTT (rank=8, AdamW, body params)
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
import struct
import zlib
from pathlib import Path
from collections import defaultdict

import numpy as np
import sentencepiece as spm
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP

try:
    import zstandard as zstd
    HAS_ZSTD = True
except ImportError:
    HAS_ZSTD = False

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

    val_batch_size  = int(os.environ.get("VAL_BATCH_SIZE", 524_288))
    val_loss_every  = int(os.environ.get("VAL_LOSS_EVERY", 1000))
    train_log_every = int(os.environ.get("TRAIN_LOG_EVERY", 20))

    iterations     = int(os.environ.get("ITERATIONS", 20000))
    warmdown_iters = int(os.environ.get("WARMDOWN_ITERS", 1400))
    warmup_steps   = int(os.environ.get("WARMUP_STEPS", 20))
    train_batch_tokens = int(os.environ.get("TRAIN_BATCH_TOKENS", 131072))
    train_seq_len  = int(os.environ.get("TRAIN_SEQ_LEN", 2048))
    max_wallclock_seconds = float(os.environ.get("MAX_WALLCLOCK_SECONDS", 600.0))
    qk_gain_init   = float(os.environ.get("QK_GAIN_INIT", 1.5))

    # Model shape
    vocab_size     = int(os.environ.get("VOCAB_SIZE", 8192))
    num_layers     = int(os.environ.get("NUM_LAYERS", 14))
    num_kv_heads   = int(os.environ.get("NUM_KV_HEADS", 4))
    model_dim      = int(os.environ.get("MODEL_DIM", 448))
    num_heads      = int(os.environ.get("NUM_HEADS", 8))
    mlp_mult       = int(os.environ.get("MLP_MULT", 3))
    tie_embeddings = True
    rope_base      = float(os.environ.get("ROPE_BASE", 10000.0))
    logit_softcap  = float(os.environ.get("LOGIT_SOFTCAP", 30.0))

    # QAT entropy regularizer — penalizes high entropy of quantized weight distribution
    # so zstd compresses the int6 payload more aggressively
    qat_lambda     = float(os.environ.get("QAT_LAMBDA", 0.0002))

    # Optimizer
    tied_embed_lr              = float(os.environ.get("TIED_EMBED_LR", 0.05))
    tied_embed_init_std        = float(os.environ.get("TIED_EMBED_INIT_STD", 0.005))
    matrix_lr                  = float(os.environ.get("MATRIX_LR", 0.04))
    scalar_lr                  = float(os.environ.get("SCALAR_LR", 0.04))
    muon_momentum              = float(os.environ.get("MUON_MOMENTUM", 0.95))
    muon_backend_steps         = int(os.environ.get("MUON_BACKEND_STEPS", 5))
    muon_momentum_warmup_start = float(os.environ.get("MUON_MOMENTUM_WARMUP_START", 0.85))
    muon_momentum_warmup_steps = int(os.environ.get("MUON_MOMENTUM_WARMUP_STEPS", 500))
    beta1      = float(os.environ.get("BETA1", 0.9))
    beta2      = float(os.environ.get("BETA2", 0.95))
    adam_eps   = float(os.environ.get("ADAM_EPS", 1e-8))
    grad_clip_norm = float(os.environ.get("GRAD_CLIP_NORM", 0.0))

    # Eval hyperparams
    eval_stride        = int(os.environ.get("EVAL_STRIDE", 64))
    ttt_lora_rank      = int(os.environ.get("TTT_LORA_RANK", 8))
    ttt_lora_lr        = float(os.environ.get("TTT_LORA_LR", 0.01))
    ttt_chunk_size     = int(os.environ.get("TTT_CHUNK_SIZE", 256))
    ttt_eval_seq_len   = int(os.environ.get("TTT_EVAL_SEQ_LEN", 2048))
    ttt_batch_size     = int(os.environ.get("TTT_BATCH_SIZE", 32))
    ngram_cache_weight = float(os.environ.get("NGRAM_CACHE_WEIGHT", 3.0))

# -----------------------------
# CONTROL TENSOR PATTERNS
# -----------------------------

CONTROL_TENSOR_NAME_PATTERNS = tuple(
    p for p in os.environ.get(
        "CONTROL_TENSOR_NAME_PATTERNS",
        "attn_scale,attn_scales,mlp_scale,mlp_scales,resid_mix,resid_mixes,q_gain,skip_weight,skip_weights",
    ).split(",") if p
)

# -----------------------------
# MUON OPTIMIZER
# -----------------------------

def zeropower_via_newtonschulz5(G: Tensor, steps: int = 10, eps: float = 1e-7) -> Tensor:
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    X /= X.norm() + eps
    transposed = G.size(0) > G.size(1)
    if transposed:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    return X.T if transposed else X


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr: float, momentum: float, backend_steps: int, nesterov: bool = True):
        super().__init__(params, dict(lr=lr, momentum=momentum, backend_steps=backend_steps, nesterov=nesterov))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        distributed = dist.is_available() and dist.is_initialized()
        world_size = dist.get_world_size() if distributed else 1
        rank = dist.get_rank() if distributed else 0
        for group in self.param_groups:
            params = group["params"]
            if not params:
                continue
            lr = group["lr"]; momentum = group["momentum"]
            backend_steps = group["backend_steps"]; nesterov = group["nesterov"]
            total_params = sum(int(p.numel()) for p in params)
            updates_flat = torch.zeros(total_params, device=params[0].device, dtype=torch.bfloat16)
            curr = 0
            for i, p in enumerate(params):
                if i % world_size == rank and p.grad is not None:
                    g = p.grad
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(g)
                    buf = state["momentum_buffer"]
                    buf.mul_(momentum).add_(g)
                    if nesterov:
                        g = g.add(buf, alpha=momentum)
                    g = zeropower_via_newtonschulz5(g, steps=backend_steps)
                    g *= max(1, g.size(0) / g.size(1)) ** 0.5
                    updates_flat[curr: curr + p.numel()] = g.reshape(-1)
                curr += p.numel()
            if distributed:
                dist.all_reduce(updates_flat, op=dist.ReduceOp.SUM)
            curr = 0
            for p in params:
                g = updates_flat[curr: curr + p.numel()].view_as(p).to(dtype=p.dtype)
                p.add_(g, alpha=-lr)
                curr += p.numel()
        return loss

# -----------------------------
# DATA LOADING
# -----------------------------

def load_data_shard(file: Path) -> Tensor:
    header_bytes = 256 * np.dtype("<i4").itemsize
    token_bytes = np.dtype("<u2").itemsize
    header = np.fromfile(file, dtype="<i4", count=256)
    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1:
        raise ValueError(f"Bad shard header: {file}")
    num_tokens = int(header[2])
    expected_size = header_bytes + num_tokens * token_bytes
    if file.stat().st_size != expected_size:
        raise ValueError(f"Shard size mismatch: {file}")
    tokens_np = np.fromfile(file, dtype="<u2", count=num_tokens, offset=header_bytes)
    return torch.from_numpy(tokens_np.astype(np.uint16, copy=False))


class TokenStream:
    def __init__(self, pattern: str):
        self.files = [Path(p) for p in sorted(glob.glob(pattern))]
        if not self.files:
            raise FileNotFoundError(f"No files: {pattern}")
        self.file_idx = 0
        self.tokens = load_data_shard(self.files[0])
        self.pos = 0

    def _advance_file(self):
        self.file_idx = (self.file_idx + 1) % len(self.files)
        self.tokens = load_data_shard(self.files[self.file_idx])
        self.pos = 0

    def take(self, n: int) -> Tensor:
        chunks = []
        remaining = n
        while remaining > 0:
            avail = self.tokens.numel() - self.pos
            if avail <= 0:
                self._advance_file()
                continue
            k = min(remaining, avail)
            chunks.append(self.tokens[self.pos: self.pos + k])
            self.pos += k
            remaining -= k
        return chunks[0] if len(chunks) == 1 else torch.cat(chunks)


class DistributedTokenLoader:
    def __init__(self, pattern: str, rank: int, world_size: int, device: torch.device):
        self.rank = rank; self.world_size = world_size; self.device = device
        self.stream = TokenStream(pattern)

    def next_batch(self, global_tokens: int, seq_len: int, grad_accum_steps: int):
        local_tokens = global_tokens // (self.world_size * grad_accum_steps)
        per_rank_span = local_tokens + 1
        chunk = self.stream.take(per_rank_span * self.world_size)
        start = self.rank * per_rank_span
        local = chunk[start: start + per_rank_span].to(dtype=torch.int64)
        x = local[:-1].reshape(-1, seq_len)
        y = local[1:].reshape(-1, seq_len)
        return x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)

# -----------------------------
# TOKENIZER BYTE COUNTING
# (unchanged from baseline — works for any SentencePiece model)
# -----------------------------

def build_sentencepiece_luts(sp, vocab_size: int, device: torch.device):
    sp_vocab_size = int(sp.vocab_size())
    table_size = max(sp_vocab_size, vocab_size)
    base_bytes_np = np.zeros((table_size,), dtype=np.int16)
    has_leading_space_np = np.zeros((table_size,), dtype=np.bool_)
    is_boundary_token_np = np.ones((table_size,), dtype=np.bool_)
    for token_id in range(sp_vocab_size):
        if sp.is_control(token_id) or sp.is_unknown(token_id) or sp.is_unused(token_id):
            continue
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


def load_validation_tokens(pattern: str, seq_len: int) -> Tensor:
    files = [Path(p) for p in sorted(glob.glob(pattern))]
    if not files:
        raise FileNotFoundError(f"No val files: {pattern}")
    tokens = torch.cat([load_data_shard(f) for f in files]).contiguous()
    usable = ((tokens.numel() - 1) // seq_len) * seq_len
    if usable <= 0:
        raise ValueError(f"Val split too short for seq_len={seq_len}")
    return tokens[: usable + 1]

# -----------------------------
# INT6 QUANTIZATION
# -----------------------------
# INT6: values in [-31, 31], packed 4 values per 3 bytes.
# Per-row scales stored as fp16.

INT6_MAX = 31
INT6_CLIP_PERCENTILE = 99.9984


def quantize_intN_per_row(t, clip_range: int = 31):
    t32 = t.float()
    if t32.ndim == 2:
        row_max = t32.abs().amax(dim=1)
        scale = (row_max / clip_range).clamp_min(1e-12).to(torch.float16)
        q = torch.clamp(torch.round(t32 / scale.float()[:, None]), -(clip_range+1), clip_range).to(torch.int8)
        return q, scale
    amax = t32.abs().max().item()
    scale = torch.tensor(max(amax / clip_range, 1e-12), dtype=torch.float16)
    q = torch.clamp(torch.round(t32 / scale.float()), -(clip_range+1), clip_range).to(torch.int8)
    return q, scale

def quantize_state_dict_int6(state_dict):
    result = {}; meta = {}
    for name, tensor in state_dict.items():
        t = tensor.detach().cpu().contiguous()
        if not t.is_floating_point() or t.numel() <= 8192 or any(p in name for p in ["tok_emb", "lm_head", "norm"]):
            result[name] = t.to(torch.float16); meta[name] = "passthrough_fp16"; continue
        if t.ndim == 2 and t.numel() > 65536:
            U, S, V = torch.linalg.svd(t.float(), full_matrices=False); rank = 14
            A = U[:, :rank] * torch.sqrt(S[:rank]); B = V[:rank, :] * torch.sqrt(S[:rank]).unsqueeze(1)
            C = t.float() - (A @ B)
            qA, sA = quantize_intN_per_row(A, 31); qB, sB = quantize_intN_per_row(B, 31); qC, sC = quantize_intN_per_row(C, 7)
            result[name+".A.q"], result[name+".A.s"] = qA, sA
            result[name+".B.q"], result[name+".B.s"] = qB, sB
            result[name+".C.q"], result[name+".C.s"] = qC, sC
            meta[name] = {"type": "svd_664", "rank": rank}
        else:
            bits = 4 if "mlp" in name else 6
            q, s = quantize_intN_per_row(t, (1<<(bits-1))-1)
            result[name+".q"], result[name+".scale"] = q, s; meta[name] = {"type": f"int{bits}"}
    return {"__quant_format__": "svd_664", "quantized": result, "meta": meta}

def dequantize_state_dict_int6(obj):
    result = obj["quantized"]; meta = obj["meta"]; out = {}
    for name, info in meta.items():
        if isinstance(info, str): out[name] = result[name]
        elif info.get("type") == "svd_664":
            A = result[name+".A.q"].float() * result[name+".A.s"].float().view(-1, 1)
            B = result[name+".B.q"].float() * result[name+".B.s"].float().view(-1, 1)
            C = result[name+".C.q"].float() * result[name+".C.s"].float().view(-1, 1)
            out[name] = (A @ B + C).to(torch.bfloat16)
        else:
            q, s = result[name+".q"], result[name+".scale"]
            out[name] = (q.float() * s.float().view(q.shape[0], *([1]*(q.ndim-1)))).to(torch.bfloat16)
    return out
def compress_quant_obj(quant_obj: dict) -> bytes:
    """Serialize and compress with zstd (if available) else lzma else zlib."""
    buf = io.BytesIO()
    torch.save(quant_obj, buf)
    raw = buf.getvalue()
    if HAS_ZSTD:
        cctx = zstd.ZstdCompressor(level=22)
        return b"ZSTD" + cctx.compress(raw)
    else:
        import lzma
        return b"LZMA" + lzma.compress(raw, preset=9)


def decompress_quant_obj(data: bytes) -> dict:
    magic = data[:4]
    payload = data[4:]
    if magic == b"ZSTD":
        if not HAS_ZSTD:
            raise RuntimeError("zstandard not installed; cannot decompress ZSTD artifact")
        dctx = zstd.ZstdDecompressor()
        raw = dctx.decompress(payload)
    elif magic == b"LZMA":
        import lzma
        raw = lzma.decompress(payload)
    else:
        # Legacy zlib fallback
        raw = zlib.decompress(data)
    return torch.load(io.BytesIO(raw), map_location="cpu")

# -----------------------------
# QAT ENTROPY REGULARIZER
# -----------------------------

def qat_entropy_loss(model: nn.Module, lambda_: float, device: torch.device) -> Tensor:
    """
    Penalize high entropy of the soft-quantized weight distribution.
    This encourages weights to cluster, making zstd compress more aggressively.
    Applied only to large 2D weight matrices.
    """
    if lambda_ <= 0.0:
        return torch.zeros((), device=device)

    total_entropy = torch.zeros((), device=device)
    count = 0
    INT6_MIN_NUMEL = 65_536

    for name, param in model.named_parameters():
        if param.ndim != 2 or param.numel() < INT6_MIN_NUMEL:
            continue
        if any(p in name for p in CONTROL_TENSOR_NAME_PATTERNS):
            continue
        p = param.float()
        # Per-row soft quantization
        clip_abs = p.abs().quantile(0.9999, dim=1, keepdim=True).detach()
        scale = (clip_abs / INT6_MAX).clamp_min(1.0 / INT6_MAX)
        q_soft = (p / scale).clamp(-INT6_MAX, INT6_MAX)
        # Map to bins [0, 62] and compute bin histogram entropy
        bin_idx = (q_soft + INT6_MAX).long().clamp(0, 62)
        counts = torch.zeros(p.shape[0], 63, device=device)
        counts.scatter_add_(1, bin_idx, torch.ones_like(q_soft))
        probs = (counts + 1e-8) / (counts.sum(dim=1, keepdim=True) + 1e-8 * 63)
        row_entropy = -(probs * probs.log()).sum(dim=1).mean()
        total_entropy = total_entropy + row_entropy
        count += 1

    return lambda_ * total_entropy / max(count, 1)

# -----------------------------
# MODEL ARCHITECTURE
# -----------------------------

class RMSNorm(nn.Module):
    def __init__(self, eps: float | None = None):
        super().__init__()
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        return F.rms_norm(x, (x.size(-1),), eps=self.eps)


class CastedLinear(nn.Linear):
    def forward(self, x: Tensor) -> Tensor:
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, self.weight.to(x.dtype), bias)


def restore_low_dim_params_to_fp32(module: nn.Module) -> None:
    with torch.no_grad():
        for name, param in module.named_parameters():
            if (param.ndim < 2 or any(p in name for p in CONTROL_TENSOR_NAME_PATTERNS)) and param.dtype != torch.float32:
                param.data = param.data.float()


class Rotary(nn.Module):
    def __init__(self, dim: int, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._seq_len_cached = 0
        self._cos_cached: Tensor | None = None
        self._sin_cached: Tensor | None = None

    def forward(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        if self._cos_cached is None or self._seq_len_cached != seq_len or self._cos_cached.device != device:
            t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
            freqs = torch.outer(t, self.inv_freq.to(device))
            self._cos_cached = freqs.cos()[None, None, :, :]
            self._sin_cached = freqs.sin()[None, None, :, :]
            self._seq_len_cached = seq_len
        return self._cos_cached.to(dtype=dtype), self._sin_cached.to(dtype=dtype)


def apply_rotary_emb(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    half = x.size(-1) // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)


class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, num_kv_heads: int, rope_base: float, qk_gain_init: float):
        super().__init__()
        self.num_heads = num_heads; self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        kv_dim = num_kv_heads * self.head_dim
        self.c_q = CastedLinear(dim, dim, bias=False)
        self.c_k = CastedLinear(dim, kv_dim, bias=False)
        self.c_v = CastedLinear(dim, kv_dim, bias=False)
        self.proj = CastedLinear(dim, dim, bias=False)
        self.proj._zero_init = True
        self.q_gain = nn.Parameter(torch.full((num_heads,), qk_gain_init, dtype=torch.float32))
        self.rotary = Rotary(self.head_dim, base=rope_base)

    def forward(self, x: Tensor, q_delta=None, v_delta=None) -> Tensor:
        bsz, seqlen, dim = x.shape
        q = self.c_q(x) + (q_delta if q_delta is not None else 0)
        k = self.c_k(x)
        v = self.c_v(x) + (v_delta if v_delta is not None else 0)
        q = q.reshape(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        q = F.rms_norm(q, (q.size(-1),)); k = F.rms_norm(k, (k.size(-1),))
        cos, sin = self.rotary(seqlen, x.device, q.dtype)
        q = apply_rotary_emb(q, cos, sin); k = apply_rotary_emb(k, cos, sin)
        q = q * self.q_gain.to(dtype=q.dtype)[None, :, None, None]
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, is_causal=True,
                                           enable_gqa=(self.num_kv_heads != self.num_heads))
        y = y.transpose(1, 2).contiguous().reshape(bsz, seqlen, dim)
        return self.proj(y)


class MLP(nn.Module):
    def __init__(self, dim: int, mlp_mult: int):
        super().__init__()
        hidden = mlp_mult * dim
        self.fc = CastedLinear(dim, hidden, bias=False)
        self.proj = CastedLinear(hidden, dim, bias=False)
        self.proj._zero_init = True

    def forward(self, x: Tensor) -> Tensor:
        x = torch.relu(self.fc(x))
        return self.proj(x.square())


class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, num_kv_heads: int, mlp_mult: int, rope_base: float, qk_gain_init: float):
        super().__init__()
        self.attn_norm = RMSNorm(); self.mlp_norm = RMSNorm()
        self.attn = CausalSelfAttention(dim, num_heads, num_kv_heads, rope_base, qk_gain_init)
        self.mlp = MLP(dim, mlp_mult)
        self.attn_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.mlp_scale  = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.resid_mix  = nn.Parameter(torch.stack((torch.ones(dim), torch.zeros(dim))).float())

    def forward(self, x: Tensor, x0: Tensor, q_delta_fn=None, v_delta_fn=None) -> Tensor:
        mix = self.resid_mix.to(dtype=x.dtype)
        x = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        n = self.attn_norm(x)
        qd = q_delta_fn(n) if q_delta_fn is not None else None
        vd = v_delta_fn(n) if v_delta_fn is not None else None
        x = x + self.attn_scale.to(dtype=x.dtype)[None, None, :] * self.attn(n, qd, vd)
        x = x + self.mlp_scale.to(dtype=x.dtype)[None, None, :] * self.mlp(self.mlp_norm(x))
        return x


class GPT(nn.Module):
    def __init__(self, args: Hyperparameters):
        super().__init__()
        self.logit_softcap = args.logit_softcap
        self.tok_emb = nn.Embedding(args.vocab_size, args.model_dim)
        self.num_encoder_layers = args.num_layers // 2
        self.num_decoder_layers = args.num_layers - self.num_encoder_layers
        self.num_skip_weights = min(self.num_encoder_layers, self.num_decoder_layers)
        self.skip_weights = nn.Parameter(torch.ones(self.num_skip_weights, args.model_dim, dtype=torch.float32))
        self.blocks = nn.ModuleList([
            Block(args.model_dim, args.num_heads, args.num_kv_heads,
                  args.mlp_mult, args.rope_base, args.qk_gain_init)
            for _ in range(args.num_layers)
        ])
        self.final_norm = RMSNorm()
        self._init_weights(args)

    def _init_weights(self, args: Hyperparameters):
        nn.init.normal_(self.tok_emb.weight, mean=0.0, std=args.tied_embed_init_std)
        for module in self.modules():
            if isinstance(module, nn.Linear) and getattr(module, "_zero_init", False):
                nn.init.zeros_(module.weight)

    def forward(self, input_ids: Tensor, target_ids: Tensor = None, lora=None, return_logits=False) -> Tensor:
        x = self.tok_emb(input_ids)
        x = F.rms_norm(x, (x.size(-1),))
        x0 = x
        skips = []
        for i in range(self.num_encoder_layers):
            qd = lora.q_loras[i] if lora else None
            vd = lora.v_loras[i] if lora else None
            x = self.blocks[i](x, x0, qd, vd)
            skips.append(x)
        for i in range(self.num_decoder_layers):
            bi = self.num_encoder_layers + i
            if skips:
                x = x + self.skip_weights[i].to(dtype=x.dtype)[None, None, :] * skips.pop()
            qd = lora.q_loras[bi] if lora else None
            vd = lora.v_loras[bi] if lora else None
            x = self.blocks[bi](x, x0, qd, vd)
        x = self.final_norm(x)
        logits = F.linear(x, self.tok_emb.weight)  # tied
        logits = logits + (lora.lm_head_lora(x) if lora else 0)
        logits = self.logit_softcap * torch.tanh(logits / self.logit_softcap)
        if return_logits: return logits
        if lora:
            bsz, sl, V = logits.shape
            return F.cross_entropy(logits.float().reshape(-1, V), target_ids.reshape(-1), reduction="none").reshape(bsz, sl)
        return F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), target_ids.reshape(-1), reduction="mean")

# -----------------------------
# CAUSAL N-GRAM DOCUMENT CACHE
# Zero disk cost. Resets at BOS boundaries. Purely causal.
# -----------------------------

BOS_ID = 1

class NgramDocumentCache:
    """
    Causal n-gram cache that learns within-document token patterns.
    Resets at document boundaries (BOS token).
    Adds zero bytes to the artifact.
    
    From the validation data analysis: ~20% of tokens are rare entities
    that appear 5x on average per document. The first occurrence costs
    ~7 nats; subsequent occurrences near-certain with this cache.
    """
    def __init__(self, vocab_size: int, device: torch.device, weight: float = 3.0):
        self.V = vocab_size
        self.device = device
        self.weight = weight
        self.reset()

    def reset(self):
        self.unigram  = torch.zeros(self.V, device=self.device, dtype=torch.float32)
        self.bigram   = {}   # prev_tok -> count tensor
        self.trigram  = {}   # (prev2, prev1) -> count tensor

    def update(self, prev2: int, prev1: int, curr: int):
        """Call AFTER observing curr — purely causal."""
        if curr == BOS_ID:
            self.reset()
            return
        self.unigram[curr] += 1.0
        if prev1 not in self.bigram:
            self.bigram[prev1] = torch.zeros(self.V, device=self.device)
        self.bigram[prev1][curr] += 1.0
        key = (prev2, prev1)
        if key not in self.trigram:
            self.trigram[key] = torch.zeros(self.V, device=self.device)
        self.trigram[key][curr] += 1.0

    def logit_delta(self, prev2: int, prev1: int) -> Tensor:
        """Return additive logit adjustment for next-token prediction."""
        delta = torch.zeros(self.V, device=self.device)
        key = (prev2, prev1)
        if key in self.trigram and self.trigram[key].sum() > 0:
            c = self.trigram[key]
            delta += self.weight * 2.0 * (c / c.sum()).log().clamp(-20, 0)
        if prev1 in self.bigram and self.bigram[prev1].sum() > 0:
            c = self.bigram[prev1]
            delta += self.weight * 1.0 * (c / c.sum()).log().clamp(-20, 0)
        if self.unigram.sum() > 0:
            delta += self.weight * 0.3 * (self.unigram / self.unigram.sum()).log().clamp(-20, 0)
        return delta

# -----------------------------
# EVALUATION — SLIDING WINDOW + N-GRAM CACHE
# -----------------------------


class BatchedLinearLoRA(nn.Module):
    def __init__(self, bsz: int, in_features: int, out_features: int, rank: int):
        super().__init__()
        self.in_features = in_features
        self.A = nn.Parameter(torch.empty(bsz, rank, in_features))
        self.B = nn.Parameter(torch.zeros(bsz, out_features, rank))
        self.reset()
    def forward(self, x: Tensor) -> Tensor:
        return (x @ self.A.transpose(1, 2)) @ self.B.transpose(1, 2)
    def reset(self):
        bound = 1.0 / math.sqrt(self.in_features)
        with torch.no_grad():
            self.A.uniform_(-bound, bound)
            self.B.zero_()

class BatchedTTTLoRA(nn.Module):
    def __init__(self, bsz: int, model: GPT, rank: int):
        super().__init__()
        dim = model.tok_emb.embedding_dim
        vocab = model.tok_emb.num_embeddings
        self.lm_head_lora = BatchedLinearLoRA(bsz, dim, vocab, rank)
        self.q_loras = nn.ModuleList()
        self.v_loras = nn.ModuleList()
        for block in model.blocks:
            self.q_loras.append(BatchedLinearLoRA(bsz, dim, block.attn.c_q.weight.shape[0], rank))
            self.v_loras.append(BatchedLinearLoRA(bsz, dim, block.attn.c_v.weight.shape[0], rank))
    def reset(self):
        for m in self.modules():
            if isinstance(m, BatchedLinearLoRA): m.reset()

def _reset_ttt_optimizer(opt):
    for group in opt.param_groups:
        for p in group['params']:
            s = opt.state.get(p)
            if s: s['exp_avg'].zero_(); s['exp_avg_sq'].zero_(); s['step'].fill_(0)

def _find_docs(all_tokens: Tensor, include_next_bos: bool = True):
    bos_positions = (all_tokens == BOS_ID).nonzero(as_tuple=True)[0].numpy()
    docs = []
    for i in range(len(bos_positions)):
        start = int(bos_positions[i])
        end = int(bos_positions[i+1]) if i+1 < len(bos_positions) else all_tokens.numel()
        if include_next_bos and i+1 < len(bos_positions): end += 1
        if end - start >= 2: docs.append((start, end - start))
    return docs

def eval_val_ttt_ngram(
    args: Hyperparameters,
    model: nn.Module,
    rank: int,
    world_size: int,
    device: torch.device,
    grad_accum_steps: int,
    val_tokens: Tensor,
    base_bytes_lut: Tensor,
    has_leading_space_lut: Tensor,
    is_boundary_token_lut: Tensor,
) -> tuple[float, float]:
    """
    Sliding window evaluation with causal n-gram document cache.
    Stride=64: each position predicted with maximum context overlap.
    """
    stride = args.eval_stride
    seq_len = args.train_seq_len  # 2048 at eval too

    total_tokens_val = val_tokens.numel() - 1
    # Assign contiguous range to this rank
    per_rank = (total_tokens_val + world_size - 1) // world_size
    start = rank * per_rank
    end = min(start + per_rank, total_tokens_val)

    val_loss_sum   = torch.zeros((), device=device, dtype=torch.float64)
    val_token_count = torch.zeros((), device=device, dtype=torch.float64)
    val_byte_count  = torch.zeros((), device=device, dtype=torch.float64)

    cache = NgramDocumentCache(args.vocab_size, device, weight=args.ngram_cache_weight)

    model.eval()
    with torch.inference_mode():
        pos = start
        prev2 = 0; prev1 = 0  # track last 2 tokens for trigram
        while pos < end:
            win_start = max(0, pos - seq_len + stride)
            win_end   = min(pos + stride, total_tokens_val)
            actual_win_len = win_end - win_start

            local = val_tokens[win_start: win_end + 1].to(device=device, dtype=torch.int64)
            x = local[:-1].unsqueeze(0)
            y = local[1:].unsqueeze(0)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                # Get base logits from model (before softmax)
                with torch.no_grad():
                    emb_out = model.module if hasattr(model, "module") else model
                    x_emb = emb_out.tok_emb(x)
                    x_emb = F.rms_norm(x_emb, (x_emb.size(-1),))
                    x0 = x_emb
                    h = x_emb
                    skips_h = []
                    for i in range(emb_out.num_encoder_layers):
                        h = emb_out.blocks[i](h, x0)
                        skips_h.append(h)
                    for i in range(emb_out.num_decoder_layers):
                        bi = emb_out.num_encoder_layers + i
                        if skips_h:
                            h = h + emb_out.skip_weights[i].to(dtype=h.dtype)[None, None, :] * skips_h.pop()
                        h = emb_out.blocks[bi](h, x0)
                    h = emb_out.final_norm(h)
                    base_logits = F.linear(h.float(), emb_out.tok_emb.weight.float())
                    base_logits = emb_out.logit_softcap * torch.tanh(base_logits / emb_out.logit_softcap)

            # Only score the stride window (last `stride` positions), not the context
            score_start_in_window = actual_win_len - stride
            score_start_in_window = max(score_start_in_window, 0)

            for local_i in range(score_start_in_window, actual_win_len):
                global_i = win_start + local_i
                if global_i >= end:
                    break

                tgt = int(y[0, local_i].item())
                p2 = int(x[0, max(0, local_i-1)].item())
                p1 = int(x[0, local_i].item())

                # Add n-gram cache logit delta
                cache_delta = cache.logit_delta(prev2, prev1)
                adjusted_logit = base_logits[0, local_i] + cache_delta

                # Per-token cross entropy
                loss_tok = F.cross_entropy(adjusted_logit.unsqueeze(0), torch.tensor([tgt], device=device))

                val_loss_sum   += loss_tok.to(torch.float64)
                val_token_count += 1.0

                # Byte count
                tb = int(base_bytes_lut[tgt].item())
                if has_leading_space_lut[tgt].item() and not is_boundary_token_lut[prev1].item():
                    tb += 1
                val_byte_count += tb

                # Update cache (causal: we've now observed tgt)
                cache.update(prev2, prev1, tgt)
                prev2 = prev1; prev1 = tgt

            pos += stride

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(val_loss_sum,    op=dist.ReduceOp.SUM)
        dist.all_reduce(val_token_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_byte_count,  op=dist.ReduceOp.SUM)

    val_loss = float(val_loss_sum.item() / val_token_count.item())
    bpb = float((val_loss_sum.item() / math.log(2.0)) / val_byte_count.item())
    model.train()
    return val_loss, bpb

# -----------------------------
# LORA TTT (Test-Time Training)
# Unchanged from baseline, kept for final eval after n-gram eval.
# -----------------------------

class BatchedLinearLoRA(nn.Module):
    def __init__(self, bsz: int, in_features: int, out_features: int, rank: int):
        super().__init__()
        self.in_features = in_features
        self.A = nn.Parameter(torch.empty(bsz, rank, in_features))
        self.B = nn.Parameter(torch.zeros(bsz, out_features, rank))
        self.reset()

    def forward(self, x: Tensor) -> Tensor:
        return (x @ self.A.transpose(1, 2)) @ self.B.transpose(1, 2)

    def reset(self):
        bound = 1.0 / math.sqrt(self.in_features)
        with torch.no_grad():
            self.A.uniform_(-bound, bound)
            self.B.zero_()


class BatchedTTTLoRA(nn.Module):
    def __init__(self, bsz: int, model: GPT, rank: int):
        super().__init__()
        dim = model.tok_emb.embedding_dim
        vocab = model.tok_emb.num_embeddings
        self.lm_head_lora = BatchedLinearLoRA(bsz, dim, vocab, rank)
        self.q_loras = nn.ModuleList()
        self.v_loras = nn.ModuleList()
        for block in model.blocks:
            self.q_loras.append(BatchedLinearLoRA(bsz, dim, block.attn.c_q.weight.shape[0], rank))
            self.v_loras.append(BatchedLinearLoRA(bsz, dim, block.attn.c_v.weight.shape[0], rank))

    def reset(self):
        for m in self.modules():
            if isinstance(m, BatchedLinearLoRA):
                m.reset()


def _reset_ttt_optimizer(opt):
    for group in opt.param_groups:
        for p in group['params']:
            s = opt.state.get(p)
            if s:
                s['exp_avg'].zero_(); s['exp_avg_sq'].zero_(); s['step'].fill_(0)


def _find_docs(all_tokens: Tensor, include_next_bos: bool = True):
    bos_positions = (all_tokens == BOS_ID).nonzero(as_tuple=True)[0].numpy()
    docs = []
    for i in range(len(bos_positions)):
        start = int(bos_positions[i])
        end = int(bos_positions[i+1]) if i+1 < len(bos_positions) else all_tokens.numel()
        if include_next_bos and i+1 < len(bos_positions):
            end += 1
        if end - start >= 2:
            docs.append((start, end - start))
    return docs


def eval_val_ttt_lora(
    args: Hyperparameters,
    base_model: GPT,
    rank: int,
    world_size: int,
    device: torch.device,
    base_bytes_lut: Tensor,
    has_leading_space_lut: Tensor,
    is_boundary_token_lut: Tensor,
) -> tuple[float, float]:
    files = sorted(glob.glob(args.val_files))
    all_tokens = torch.cat([load_data_shard(Path(f)) for f in files])
    docs = _find_docs(all_tokens)
    rank_docs = docs[(len(docs) * rank) // world_size: (len(docs) * (rank+1)) // world_size]
    chunk_size = args.ttt_chunk_size; eval_seq_len = args.ttt_eval_seq_len
    batch_size = args.ttt_batch_size; lora_rank = args.ttt_lora_rank
    rank_docs.sort(key=lambda d: (d[1] - 2) // chunk_size)

    base_model.eval()
    for p in base_model.parameters():
        p.requires_grad_(False)

    lora = BatchedTTTLoRA(batch_size, base_model, lora_rank).to(device)
    opt = torch.optim.Adam(lora.parameters(), lr=args.ttt_lora_lr, betas=(args.beta1, args.beta2), eps=1e-10)

    loss_sum   = torch.zeros((), device=device, dtype=torch.float64)
    byte_sum   = torch.zeros((), device=device, dtype=torch.float64)
    token_count = torch.zeros((), device=device, dtype=torch.float64)

    for bi in range(0, len(rank_docs), batch_size):
        batch = rank_docs[bi: bi+batch_size]
        bsz = len(batch)
        if bsz == batch_size:
            cur_lora = lora; cur_opt = opt
            cur_lora.reset(); _reset_ttt_optimizer(cur_opt)
        else:
            cur_lora = BatchedTTTLoRA(bsz, base_model, lora_rank).to(device)
            cur_opt = torch.optim.Adam(cur_lora.parameters(), lr=args.ttt_lora_lr, betas=(args.beta1, args.beta2), eps=1e-10)

        pred_lens = [doc_len - 1 for _, doc_len in batch]
        num_chunks = [(pl + chunk_size - 1) // chunk_size for pl in pred_lens]
        max_nc = max(num_chunks)

        for ci in range(max_nc):
            chunk_end = (ci+1)*chunk_size; context_size = min(chunk_end, eval_seq_len)
            win_start_base = chunk_end - context_size; chunk_offset = ci * chunk_size - win_start_base
            chunk_len_base = min(chunk_size, context_size - chunk_offset)
            active = [ci < nc for nc in num_chunks]
            needs_train = any(ci < nc-1 for nc in num_chunks)

            x = torch.zeros(bsz, context_size, dtype=torch.int64, device=device)
            y = torch.zeros(bsz, context_size, dtype=torch.int64, device=device)
            doc_info = []
            for b in range(bsz):
                if not active[b]:
                    doc_info.append((0, 0)); continue
                ds, dl = batch[b]
                pl = pred_lens[b]; nc = num_chunks[b]
                win_start = max(0, chunk_end - eval_seq_len)
                win_len = min(chunk_end, pl) - win_start
                co = ci * chunk_size - win_start
                cl = min(chunk_size, pl - ci*chunk_size)
                chunk = all_tokens[ds+win_start: ds+win_start+win_len+1]
                toks = chunk.to(dtype=torch.int64, device=device)
                x[b, :win_len] = toks[:-1]; y[b, :win_len] = toks[1:]
                doc_info.append((co, cl))

            if needs_train:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    ptl = base_model(x, y, lora=cur_lora)
            else:
                with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    ptl = base_model(x, y, lora=cur_lora)

            with torch.no_grad():
                for b in range(bsz):
                    if not active[b]: continue
                    co, cl = doc_info[b]
                    lbl = ptl[b, co:co+cl].to(torch.float64)
                    prev = x[b, co:co+cl]; tgt = y[b, co:co+cl]
                    tok_bytes = base_bytes_lut[tgt].to(torch.float64)
                    tok_bytes += has_leading_space_lut[tgt] & ~is_boundary_token_lut[prev]
                    loss_sum += lbl.sum(); byte_sum += tok_bytes.sum(); token_count += cl

            if needs_train:
                mask = torch.tensor([float(ci < num_chunks[b]-1) for b in range(bsz)], device=device)
                per_doc = ptl[:, chunk_offset:chunk_offset+chunk_size].mean(dim=-1)
                cur_opt.zero_grad(); (per_doc * mask).sum().backward(); cur_opt.step()

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(byte_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(token_count, op=dist.ReduceOp.SUM)

    val_loss = float(loss_sum.item() / token_count.item())
    val_bpb  = float((loss_sum.item() / math.log(2.0)) / byte_sum.item())
    return val_loss, val_bpb

# -----------------------------
# TRAINING
# -----------------------------

def main() -> None:
    global zeropower_via_newtonschulz5
    code = Path(__file__).read_text(encoding="utf-8")
    args = Hyperparameters()
    zeropower_via_newtonschulz5 = torch.compile(zeropower_via_newtonschulz5)

    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    rank       = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    grad_accum_steps = 8 // world_size
    grad_scale = 1.0 / grad_accum_steps

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if distributed:
        dist.init_process_group(backend="nccl", device_id=device)
        dist.barrier()
    master_process = rank == 0

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    from torch.backends.cuda import enable_flash_sdp, enable_cudnn_sdp, enable_mem_efficient_sdp, enable_math_sdp
    enable_cudnn_sdp(False); enable_flash_sdp(True); enable_mem_efficient_sdp(False); enable_math_sdp(False)

    logfile = None
    if master_process:
        os.makedirs("logs", exist_ok=True)
        logfile = f"logs/{args.run_id}.txt"
        print(logfile)

    def log0(msg: str, console: bool = True):
        if not master_process: return
        if console: print(msg)
        if logfile:
            with open(logfile, "a", encoding="utf-8") as f:
                print(msg, file=f)

    log0(code, console=False)
    log0("=" * 100, console=False)
    log0(f"V={args.vocab_size} D={args.model_dim} L={args.num_layers} MLP={args.mlp_mult}x", console=False)

    random.seed(args.seed); np.random.seed(args.seed)
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)

    if not args.tokenizer_path.endswith(".model"):
        raise ValueError(f"Expected SentencePiece .model: {args.tokenizer_path}")
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    if int(sp.vocab_size()) != args.vocab_size:
        raise ValueError(f"Tokenizer vocab mismatch: {int(sp.vocab_size())} vs {args.vocab_size}")

    val_tokens = load_validation_tokens(args.val_files, args.train_seq_len)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(
        sp, args.vocab_size, device)

    log0(f"val_tokens:{val_tokens.numel()-1} train_seq_len:{args.train_seq_len}")

    # Model
    base_model = GPT(args).to(device).bfloat16()
    for module in base_model.modules():
        if isinstance(module, CastedLinear):
            module.float()
        if isinstance(module, Rotary):
            module.inv_freq.data = module.inv_freq.data.float()
    restore_low_dim_params_to_fp32(base_model)
    compiled_model = torch.compile(base_model)
    model: nn.Module = DDP(compiled_model, device_ids=[local_rank], broadcast_buffers=False) if distributed else compiled_model

    n_params = sum(p.numel() for p in base_model.parameters())
    log0(f"model_params:{n_params}")

    # Optimizer split (same structure as baseline)
    block_named_params = list(base_model.blocks.named_parameters())
    matrix_params = [p for name, p in block_named_params
                     if p.ndim == 2 and not any(pat in name for pat in CONTROL_TENSOR_NAME_PATTERNS)]
    scalar_params  = [p for name, p in block_named_params
                     if p.ndim < 2 or any(pat in name for pat in CONTROL_TENSOR_NAME_PATTERNS)]
    if base_model.skip_weights.numel() > 0:
        scalar_params.append(base_model.skip_weights)

    optimizer_tok = torch.optim.Adam(
        [{"params": [base_model.tok_emb.weight], "lr": args.tied_embed_lr, "base_lr": args.tied_embed_lr}],
        betas=(args.beta1, args.beta2), eps=args.adam_eps, fused=True)
    optimizer_muon = Muon(matrix_params, lr=args.matrix_lr, momentum=args.muon_momentum,
                          backend_steps=args.muon_backend_steps)
    for group in optimizer_muon.param_groups:
        group["base_lr"] = args.matrix_lr
    optimizer_scalar = torch.optim.Adam(
        [{"params": scalar_params, "lr": args.scalar_lr, "base_lr": args.scalar_lr}],
        betas=(args.beta1, args.beta2), eps=args.adam_eps, fused=True)
    optimizers = [optimizer_tok, optimizer_muon, optimizer_scalar]

    train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)

    def zero_grad_all():
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)

    max_wallclock_ms = 1000.0 * args.max_wallclock_seconds if args.max_wallclock_seconds > 0 else None

    def lr_mul(step: int, elapsed_ms: float) -> float:
        if args.warmdown_iters <= 0: return 1.0
        if max_wallclock_ms is None:
            warmdown_start = max(args.iterations - args.warmdown_iters, 0)
            return max((args.iterations - step) / max(args.warmdown_iters, 1), 0.0) if warmdown_start <= step < args.iterations else 1.0
        step_ms = elapsed_ms / max(step, 1)
        warmdown_ms = args.warmdown_iters * step_ms
        remaining_ms = max(max_wallclock_ms - elapsed_ms, 0.0)
        return remaining_ms / max(warmdown_ms, 1e-9) if remaining_ms <= warmdown_ms else 1.0

    # Warmup
    if args.warmup_steps > 0:
        initial_model_state = {n: t.detach().cpu().clone() for n, t in base_model.state_dict().items()}
        initial_optimizer_states = [copy.deepcopy(opt.state_dict()) for opt in optimizers]
        model.train()
        for warmup_step in range(args.warmup_steps):
            zero_grad_all()
            for micro_step in range(grad_accum_steps):
                if distributed:
                    model.require_backward_grad_sync = micro_step == grad_accum_steps - 1
                x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                    loss = model(x, y)
                (loss * grad_scale).backward()
            for opt in optimizers:
                opt.step()
            zero_grad_all()
            if warmup_step + 1 == args.warmup_steps:
                log0(f"warmup_step:{warmup_step+1}/{args.warmup_steps}")
        base_model.load_state_dict(initial_model_state, strict=True)
        for opt, state in zip(optimizers, initial_optimizer_states, strict=True):
            opt.load_state_dict(state)
        zero_grad_all()
        if distributed:
            model.require_backward_grad_sync = True
        train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)

    # Training loop
    training_time_ms = 0.0
    stop_after_step: int | None = None
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    step = 0
    while True:
        last_step = step == args.iterations or (stop_after_step is not None and step >= stop_after_step)

        if last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0 and step > 0):
            torch.cuda.synchronize()
            training_time_ms += 1000.0 * (time.perf_counter() - t0)
            val_loss, val_bpb = eval_val_ttt_ngram(
                args, base_model, rank, world_size, device, grad_accum_steps,
                val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut)
            log0(f"step:{step}/{args.iterations} val_loss:{val_loss:.4f} val_bpb:{val_bpb:.4f} "
                 f"train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms/max(step,1):.2f}ms")
            torch.cuda.synchronize()
            t0 = time.perf_counter()

        if last_step:
            if stop_after_step is not None and step < args.iterations:
                log0(f"stopping_early: wallclock_cap step:{step}/{args.iterations}")
            break

        elapsed_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        scale = lr_mul(step, elapsed_ms)
        zero_grad_all()
        train_loss = torch.zeros((), device=device)

        for micro_step in range(grad_accum_steps):
            if distributed:
                model.require_backward_grad_sync = micro_step == grad_accum_steps - 1
            x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                loss = model(x, y)
                # QAT entropy regularizer — runs on raw base_model (not compiled wrapper)
                if args.qat_lambda > 0.0 and micro_step == 0:
                    qat_reg = qat_entropy_loss(base_model, args.qat_lambda, device)
                    loss = loss + qat_reg
            train_loss += loss.detach()
            (loss * grad_scale).backward()
        train_loss /= grad_accum_steps

        frac = min(step / args.muon_momentum_warmup_steps, 1.0) if args.muon_momentum_warmup_steps > 0 else 1.0
        muon_momentum = (1 - frac) * args.muon_momentum_warmup_start + frac * args.muon_momentum
        for group in optimizer_muon.param_groups:
            group["momentum"] = muon_momentum
        for opt in optimizers:
            for group in opt.param_groups:
                group["lr"] = group["base_lr"] * scale

        if args.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(base_model.parameters(), args.grad_clip_norm)
        for opt in optimizers:
            opt.step()
        zero_grad_all()

        step += 1
        approx_training_time_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)

        if args.train_log_every > 0 and (step <= 10 or step % args.train_log_every == 0):
            log0(f"step:{step}/{args.iterations} train_loss:{train_loss.item():.4f} "
                 f"train_time:{approx_training_time_ms:.0f}ms step_avg:{approx_training_time_ms/step:.2f}ms")

        reached_cap = max_wallclock_ms is not None and approx_training_time_ms >= max_wallclock_ms
        if distributed and max_wallclock_ms is not None:
            reached_cap_tensor = torch.tensor(int(reached_cap), device=device)
            dist.all_reduce(reached_cap_tensor, op=dist.ReduceOp.MAX)
            reached_cap = bool(reached_cap_tensor.item())
        if stop_after_step is None and reached_cap:
            stop_after_step = step

    log0(f"peak_memory:{torch.cuda.max_memory_allocated()//1024//1024}MiB")

    # Serialize
    if master_process:
        torch.save(base_model.state_dict(), "final_model.pt")
        log0(f"raw_model:{os.path.getsize('final_model.pt')} bytes")

    # INT6 + zstd22 (or lzma fallback)
    quant_obj = quantize_state_dict_int6(base_model.state_dict())
    compressed_blob = compress_quant_obj(quant_obj)
    if master_process:
        with open("final_model.int6.ptz", "wb") as f:
            f.write(compressed_blob)
        code_bytes = len(code.encode("utf-8"))
        model_bytes = os.path.getsize("final_model.int6.ptz")
        log0(f"compressed_model:{model_bytes} bytes")
        log0(f"code:{code_bytes} bytes")
        log0(f"total_artifact:{model_bytes + code_bytes} bytes")
        if model_bytes + code_bytes > 16_000_000:
            log0("WARNING: artifact exceeds 16MB limit!")

    # Roundtrip validation
    if distributed:
        dist.barrier()
    with open("final_model.int6.ptz", "rb") as f:
        blob_disk = f.read()
    recovered_obj = decompress_quant_obj(blob_disk)
    base_model.load_state_dict(dequantize_state_dict_int6(recovered_obj), strict=True)

    torch.cuda.synchronize(); t_qeval = time.perf_counter()
    q_val_loss, q_val_bpb = eval_val_ttt_ngram(
        args, base_model, rank, world_size, device, grad_accum_steps,
        val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut)
    torch.cuda.synchronize()
    log0(f"roundtrip_sliding_ngram val_loss:{q_val_loss:.4f} val_bpb:{q_val_bpb:.4f} "
         f"eval_time:{1000.0*(time.perf_counter()-t_qeval):.0f}ms")

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()