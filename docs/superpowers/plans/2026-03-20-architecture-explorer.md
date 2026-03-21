# Architecture Explorer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a fully autonomous research system that uses simulated annealing + an LLM-as-researcher to discover completely different architectural families for parameter-golf. The LLM decides what to try next based on: (1) its own knowledge of ML research and papers, (2) a central exploration log showing everything that's been tried and what scored what. No predefined menu. The LLM is the researcher.

**Architecture:** A single `explore/` directory. Two agent-modifiable files (`architecture.py`, `train_config.py`). An immutable `harness.py`. An `orchestrator.py` that loops forever: read exploration log → call LLM → LLM proposes a novel idea and writes code → train → measure BPB → SA accept/reject → update exploration log → repeat.

**The central insight:** The LLM is not picking from a list. It reads the full history of what's been tried (scores, notes, what worked, what didn't), reasons about what unexplored territory exists in the ML literature, and proposes the next experiment. It is doing autonomous research.

**Tech Stack:** PyTorch, torchrun (multi-GPU), litellm (LLM API), sentencepiece, zlib/int8 quantization. Uses existing parameter-golf data pipeline unchanged.

---

## File Map

| File | Who touches it | Responsibility |
|------|---------------|----------------|
| `explore/architecture.py` | LLM agent (70% of turns) | Model class. LLM rewrites this with novel architectures it proposes. |
| `explore/train_config.py` | LLM agent (30% of turns) | `Config` dataclass. LLM tunes hyperparameters to match the current architecture. |
| `explore/harness.py` | IMMUTABLE | Training loop, BPB eval, 16MB size check, output format. Never touched by the LLM. |
| `explore/orchestrator.py` | Human sets up, then autonomous forever | SA loop, LLM calls, backup/restore, log updates, file enforcement. Runs until manually stopped. |
| `explore/exploration_log.md` | Orchestrator writes only | Compressed research memory. Two sections: Family Summaries (updated in-place) + Recent Experiments (rolling window of 10). Never grows beyond ~2 pages. |
| `explore/CONSTRAINTS.md` | Human writes once | Hard technical constraints. LLM reads this before every proposal. |
| `explore/history.json` | Orchestrator writes only | SA state. Machine-readable. Not shown to LLM. |
| `explore/results.tsv` | Orchestrator writes only | TSV log of all runs. |
| `explore/run.log` | Harness writes (overwritten each run) | Stdout/stderr from last harness run. Used for crash diagnosis. |
| `explore/architecture.py.bak` | Orchestrator backup | Snapshot before each LLM write. Restored on reject/crash. |
| `explore/train_config.py.bak` | Orchestrator backup | Snapshot before each LLM write. Restored on reject/crash. |

**Everything else in `explore/` is deleted automatically after each step (file enforcement).**

### Strict File Whitelist

After every LLM write and after every harness run, the orchestrator scans `explore/` and deletes
any file not on this list:

```python
ALLOWED_FILES = {
    "architecture.py", "architecture.py.bak",
    "train_config.py", "train_config.py.bak",
    "harness.py", "orchestrator.py",
    "CONSTRAINTS.md", "exploration_log.md",
    "history.json", "results.tsv", "run.log",
}
```

If the LLM writes extra files (e.g., `utils.py`, `mamba.py`, `__pycache__`), they are deleted
and the violation is noted. The LLM is told in its system prompt: **one file, no imports of files
you create, no helper modules**.

### Key Interface Contract (harness ↔ agent files)

```python
# harness.py imports these — contract must never change:
from architecture import build_model            # required always
from architecture import get_optimizer          # optional hook
from train_config import Config                 # required always
```

**`build_model(cfg: Config) -> nn.Module`**
- `forward(x: Tensor, y: Tensor) -> Tensor` returns scalar cross-entropy loss

**`get_optimizer(model, cfg) -> list[torch.optim.Optimizer]`** (optional)
- If absent: harness defaults to `p.ndim == 2` → Muon, else → AdamW
- If present: returns list of optimizers, all get same LR schedule
- Example: `return [torch.optim.AdamW(model.parameters(), lr=cfg.matrix_lr)]`

---

## Prerequisites (H100 machine)

```bash
pip install litellm
echo "litellm" >> requirements.txt
```

---

## Task 1: Create CONSTRAINTS.md and exploration_log.md (initial state)

**Files:**
- Create: `explore/CONSTRAINTS.md`
- Create: `explore/exploration_log.md`

These two files are the LLM's entire world context before it proposes an experiment.

- [ ] **Step 1: Create CONSTRAINTS.md**

```markdown
# Parameter Golf: Hard Constraints

You are an autonomous ML researcher trying to win the OpenAI Parameter Golf challenge.
The goal: train the best language model that fits in 16MB, in under 10 minutes on 8xH100.
Scored by bits-per-byte (BPB) on FineWeb validation — lower is better.

## Current Leaderboard (as of 2026-03-19)
- SOTA: 1.1748 BPB (sliding window eval + FP16 embeddings + 10L + Muon WD + spectral init)
- Baseline transformer: 1.2244 BPB

## What you are doing
Each turn, you either:
(A) Write a completely new model architecture in architecture.py — propose something novel
    that the leaderboard hasn't explored yet. Draw on ML research papers you know.
(B) Tune train_config.py hyperparameters to better fit the current architecture.

## Hard Technical Constraints

### 16MB artifact limit
The 16MB budget covers: compressed int8 model bytes + architecture.py + train_config.py + harness.py.
If you exceed 16MB, BPB is penalized. Smaller models = more budget for depth/cleverness.

### Interface contract (NEVER break these)
architecture.py MUST contain:
  def build_model(cfg) -> nn.Module:
      # module.forward(x: Tensor, y: Tensor) -> scalar loss (cross-entropy)

architecture.py MAY contain:
  def get_optimizer(model, cfg) -> list[torch.optim.Optimizer]:
      # if absent, harness uses: ndim==2 → Muon, else → AdamW

train_config.py MUST contain:
  @dataclass
  class Config:
      # These fields are REQUIRED (harness reads them):
      data_path, tokenizer_path, vocab_size
      max_wallclock_seconds, train_batch_tokens, train_seq_len, val_batch_size
      val_loss_every, train_log_every, warmup_steps, warmdown_iters, seed, grad_clip_norm
      model_dim, qk_gain_init, tied_embed_init_std, tie_embeddings, logit_softcap
      embed_lr, head_lr, tied_embed_lr, matrix_lr, scalar_lr
      muon_momentum, muon_backend_steps, muon_momentum_warmup_start, muon_momentum_warmup_steps
      beta1, beta2, adam_eps
      # You may ADD new fields freely for architecture-specific settings

### Packages available
torch, numpy, sentencepiece, math, zlib, os, sys, copy, random — standard library + PyTorch only.
No new pip installs. No internet access during training.

### Output only code
When writing architecture.py or train_config.py, output ONLY valid Python. No markdown, no prose.
```

- [ ] **Step 2: Create exploration_log.md (initial state)**

```markdown
# Exploration Log

This is the central memory of all experiments run. Every entry records:
- What architectural idea was tried
- The key hypothesis ("why might this work?")
- The result (val_bpb, penalized_bpb, artifact size)
- Whether it was kept (SA accepted) or discarded
- Key insight from the result

Read this before proposing your next experiment. Do not repeat ideas that scored poorly
unless you have a specific reason to believe a different implementation would help.
Look for patterns: what families are unexplored? What near-misses could be pushed further?

---

## Run History

*No experiments yet. You are starting fresh.*

The parameter-golf leaderboard has explored:
- Standard transformer (9-10 layers, 512 dim, GQA, RoPE, tied embeddings): 1.2244 BPB
- FP16 embeddings + Muon weight decay: 1.2197 BPB
- Longer sequence lengths (2k, 4k context): 1.2014 BPB
- Mixed int6/int8 quantization: 1.2147 BPB
- LoRA test-time training at eval: 1.1928 BPB
- Sliding window evaluation: 1.1925 BPB
- 10 layers + sliding window + spectral init: 1.1748 BPB (current SOTA)

**Architectures NOT yet explored on the leaderboard (as of 2026-03-19):**
- State space models (Mamba, S4, S6)
- Linear attention / RWKV-style recurrence
- Depth-recurrent transformers (weight sharing across layers)
- Binary/ternary weight networks (BitNet)
- Mixture of Experts
- Hyena / long convolution operators
- Any non-transformer sequence model

Your job: explore these and anything else you can think of from ML research.
```

- [ ] **Step 3: Verify files exist**

```bash
ls explore/CONSTRAINTS.md explore/exploration_log.md
```
Expected: both listed.

---

## Task 2: Create train_config.py

**Files:**
- Create: `explore/train_config.py`

- [ ] **Step 1: Write train_config.py**

```python
"""
train_config.py — LLM-modifiable hyperparameters.

INTERFACE: Config must be a dataclass, instantiated as cfg = Config().
Fields marked REQUIRED must remain — harness.py reads them.
LLM may add new fields for architecture-specific settings.
"""
from dataclasses import dataclass
import os


@dataclass
class Config:
    # --- Data (REQUIRED) ---
    data_path: str = os.environ.get("DATA_PATH", "./data/datasets/fineweb10B_sp1024")
    tokenizer_path: str = os.environ.get("TOKENIZER_PATH", "./data/tokenizers/fineweb_1024_bpe.model")
    vocab_size: int = 1024  # REQUIRED

    # --- Training budget (REQUIRED) ---
    max_wallclock_seconds: float = float(os.environ.get("MAX_WALLCLOCK_SECONDS", 600.0))
    train_batch_tokens: int = 524_288
    train_seq_len: int = 1024       # REQUIRED
    val_batch_size: int = 524_288
    val_loss_every: int = 1000
    train_log_every: int = 200
    warmup_steps: int = 20
    warmdown_iters: int = 1200
    seed: int = int(os.environ.get("SEED", 1337))
    grad_clip_norm: float = 0.0     # REQUIRED (0.0 = disabled)

    # --- Model shape (REQUIRED fields, LLM tunes values) ---
    model_dim: int = 512
    num_layers: int = 9
    num_heads: int = 8
    num_kv_heads: int = 4
    mlp_mult: int = 2
    tie_embeddings: bool = True     # REQUIRED
    rope_base: float = 10000.0
    logit_softcap: float = 30.0     # REQUIRED
    qk_gain_init: float = 1.5       # REQUIRED

    # --- Optimizer (REQUIRED fields, LLM tunes values) ---
    embed_lr: float = 0.6
    head_lr: float = 0.008
    tied_embed_lr: float = 0.05
    tied_embed_init_std: float = 0.005  # REQUIRED
    matrix_lr: float = 0.04
    scalar_lr: float = 0.04
    muon_momentum: float = 0.95
    muon_backend_steps: int = 5
    muon_momentum_warmup_start: float = 0.85
    muon_momentum_warmup_steps: int = 500
    beta1: float = 0.9
    beta2: float = 0.95
    adam_eps: float = 1e-8

    # --- Architecture-specific extras (LLM adds fields here freely) ---
    # Examples:
    # recurrence_steps: int = 4       # depth-recurrent
    # ssm_state_dim: int = 16         # SSM/Mamba
    # bitnet_threshold: float = 1.0   # BitNet
```

- [ ] **Step 2: Verify locally (no GPU needed)**

```bash
python -c "
import sys; sys.path.insert(0, 'explore')
from train_config import Config
c = Config()
print(c.model_dim, c.qk_gain_init, c.tied_embed_init_std)
"
```
Expected: `512 1.5 0.005`

---

## Task 3: Create architecture.py (baseline transformer as starting point)

**Files:**
- Create: `explore/architecture.py`

Extract the transformer model classes from `train_gpt.py`. The LLM will replace this entire file on its first architecture turn.

**Classes to extract** — run this first to get exact line numbers:
```bash
grep -n "^class " /Users/sidharth/Documents/parameter-golf/train_gpt.py
```

Then read lines from first class to end of `GPT` class and copy into `architecture.py`.
Classes needed: `RMSNorm`, `CastedLinear`, `Rotary`, `CausalSelfAttention`, `MLP`, `Block`, `GPT`.

Do NOT copy quantization functions or data loading — those stay in harness.py.

- [ ] **Step 1: Create explore/architecture.py with model classes + build_model**

End the file with:

```python
def build_model(cfg) -> nn.Module:
    """
    INTERFACE — harness.py calls only this function from this file.
    LLM: replace EVERYTHING above and this function with your architecture.

    Optional: add get_optimizer(model, cfg) -> list[Optimizer] for custom optimizer.
    """
    return GPT(
        vocab_size=cfg.vocab_size,
        num_layers=cfg.num_layers,
        num_heads=cfg.num_heads,
        num_kv_heads=cfg.num_kv_heads,
        model_dim=cfg.model_dim,
        mlp_mult=cfg.mlp_mult,
        tie_embeddings=cfg.tie_embeddings,
        tied_embed_init_std=cfg.tied_embed_init_std,
        rope_base=cfg.rope_base,
        logit_softcap=cfg.logit_softcap,
        qk_gain_init=cfg.qk_gain_init,
    )
```

- [ ] **Step 2: Verify interface locally (no GPU needed)**

```bash
python -c "
import torch, sys; sys.path.insert(0, 'explore')
from train_config import Config
from architecture import build_model
cfg = Config()
m = build_model(cfg)
print('params M:', sum(p.numel() for p in m.parameters()) / 1e6)
x = torch.randint(0, 1024, (1, 16))
loss = m(x, x)
print('loss:', loss.item())
"
```
Expected: `params M: ~16.x` and `loss: ~6.9`

---

## Task 4: Create harness.py

**Files:**
- Create: `explore/harness.py`

**⚠️ REQUIRES CUDA. Smoke test on H100 only.**

Extract from `train_gpt.py`. Verify exact line numbers first:
```bash
grep -n "^def \|^class \|^# ---" /Users/sidharth/Documents/parameter-golf/train_gpt.py | head -60
```

| What to extract | Approx lines |
|----------------|-------------|
| Muon optimizer (`zeropower_via_newtonschulz5`, `Muon` class) | ~100–175 |
| Eval setup (`build_sentencepiece_luts`, `load_validation_tokens`, `eval_val`) | ~187–285 |
| Quantization + size functions | ~295–430 |
| Data loading (`load_data_shard`, `DataLoader`) | ~436–505 |
| Training loop (main block) | ~961–1370 |

**Key modifications from train_gpt.py:**

**1. Top of file — dynamic import of agent files:**
```python
"""harness.py — IMMUTABLE. The LLM must never modify this file."""
import sys, os, importlib.util
sys.path.insert(0, os.path.dirname(__file__))

def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_here = os.path.dirname(os.path.abspath(__file__))
_arch = _load_module("architecture", os.path.join(_here, "architecture.py"))
_conf = _load_module("train_config",  os.path.join(_here, "train_config.py"))

build_model    = _arch.build_model
get_optimizer_fn = getattr(_arch, "get_optimizer", None)
Config         = _conf.Config
```

**2. Default optimizer — use parameter shape, not module path:**
```python
def _build_optimizers(model, cfg, optimizers_override_fn):
    if optimizers_override_fn is not None:
        return optimizers_override_fn(model, cfg)
    # Default: 2D parameters → Muon, everything else → AdamW
    matrix_params = [p for p in model.parameters() if p.ndim == 2 and p.requires_grad]
    scalar_params  = [p for p in model.parameters() if p.ndim != 2 and p.requires_grad]
    return [
        Muon(matrix_params, lr=cfg.matrix_lr, momentum=cfg.muon_momentum,
             backend_steps=cfg.muon_backend_steps),
        torch.optim.AdamW(scalar_params, lr=cfg.scalar_lr,
                          betas=(cfg.beta1, cfg.beta2), eps=cfg.adam_eps),
    ]
```

**3. Size measurement — include all source files:**
```python
SIZE_LIMIT_BYTES = 16_000_000

def _measure_artifact(model, explore_dir) -> tuple[float, bool]:
    # Quantize + zlib compress model weights
    model_bytes = _quantize_and_compress(model)   # use quantize logic from train_gpt.py
    # Add source code bytes
    code_bytes = sum(
        os.path.getsize(os.path.join(explore_dir, f))
        for f in ("architecture.py", "train_config.py", "harness.py")
        if os.path.exists(os.path.join(explore_dir, f))
    )
    total = model_bytes + code_bytes
    return total / 1e6, total <= SIZE_LIMIT_BYTES
```

**4. Output block at end of main:**
```python
size_penalty = max(0.0, (total_bytes - SIZE_LIMIT_BYTES) / 1e6 * 0.1)
print(f"val_bpb:       {val_bpb:.4f}")
print(f"val_loss:      {val_loss:.4f}")
print(f"artifact_mb:   {artifact_mb:.2f}")
print(f"size_ok:       {'true' if size_ok else 'false'}")
print(f"size_penalty:  {size_penalty:.4f}")
print(f"penalized_bpb: {val_bpb + size_penalty:.4f}")
print(f"num_params_M:  {num_params/1e6:.1f}")
print(f"num_steps:     {step}")
```

- [ ] **Step 1: Write harness.py (~500 lines)**

- [ ] **Step 2: Smoke test on H100 (2-minute budget)**

```bash
cd /workspace/parameter-golf
MAX_WALLCLOCK_SECONDS=120 \
DATA_PATH=./data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
torchrun --standalone --nproc_per_node=1 explore/harness.py 2>&1 | tail -15
```
Expected: prints `val_bpb: X.XXXX` block.

- [ ] **Step 3: Confirm parse**

```bash
MAX_WALLCLOCK_SECONDS=120 torchrun --standalone --nproc_per_node=1 \
  explore/harness.py > explore/run.log 2>&1
grep "^penalized_bpb:" explore/run.log
```
Expected: `penalized_bpb: X.XXXX`

---

## Task 5: Create orchestrator.py

**Files:**
- Create: `explore/orchestrator.py`

The orchestrator handles four concerns:
1. **SA loop** — propose → run → accept/reject → cool
2. **Crash retry** — on crash, pass error back to LLM, fix and retry (max 3 attempts)
3. **File enforcement** — after every write, delete any file not on the whitelist
4. **Log compression** — when Recent Experiments hits 10, compress into Family Summaries

**Crash retry logic:**
```
propose code → write file → enforce whitelist → run harness
  if crash:
    for attempt in 1..3:
      call LLM: "your code crashed with this error, fix it"
      write fixed code → enforce whitelist → run harness
      if not crash: break
    if still crash after 3: restore backup, log as crash, move on
```

**Log compression — the log always has two sections:**
```markdown
## Family Summaries
(one entry per architecture family, updated in-place by orchestrator)

## Recent Experiments
(last 10 runs in full detail — oldest dropped when new one added)
```
When Recent Experiments hits 10, one LLM call compresses them into Family Summaries and clears Recent. The log never exceeds ~2 pages regardless of runtime.

**Compression prompt:**
```
Given these 10 recent experiment entries, update the Family Summaries section.
For each architecture family that appeared, write/update one bullet:
  - FamilyName: best_bpb=X, tried N variants, key finding: [one sentence]
Sort by best_bpb ascending. Output ONLY the updated Family Summaries section content.
```

**File whitelist enforcement (runs after every LLM write and after every harness run):**
```python
ALLOWED = {
    "architecture.py", "architecture.py.bak",
    "train_config.py", "train_config.py.bak",
    "harness.py", "orchestrator.py",
    "CONSTRAINTS.md", "exploration_log.md",
    "history.json", "results.tsv", "run.log",
}

def enforce_whitelist():
    for f in EXPLORE_DIR.iterdir():
        if f.name not in ALLOWED and not f.name.startswith("__"):
            print(f"[enforce] deleting rogue file: {f.name}")
            f.unlink() if f.is_file() else shutil.rmtree(f)
```

- [ ] **Step 1: Write orchestrator.py**

```python
"""
orchestrator.py — Autonomous ML researcher. Runs forever until manually stopped.

Per-step loop:
  1. Build LLM context (CONSTRAINTS.md + exploration_log.md + current files)
  2. Call LLM → get new architecture.py or train_config.py code
  3. Enforce file whitelist (delete rogue files)
  4. Run harness — if crash: retry up to MAX_RETRIES with error fed back to LLM
  5. Simulated annealing accept/reject
  6. Update exploration_log.md (append to Recent, compress when full)
  7. Persist SA state, cool temperature

Usage:
  export ANTHROPIC_API_KEY="sk-ant-..."
  export NPROC_PER_NODE=8
  export MAX_WALLCLOCK_SECONDS=600
  python explore/orchestrator.py
"""
import json, math, os, random, shutil, subprocess, time
from datetime import datetime
from pathlib import Path

import litellm

EXPLORE_DIR  = Path(__file__).parent
ARCH_FILE    = EXPLORE_DIR / "architecture.py"
CONFIG_FILE  = EXPLORE_DIR / "train_config.py"
HARNESS_FILE = EXPLORE_DIR / "harness.py"
CONSTRAINTS  = EXPLORE_DIR / "CONSTRAINTS.md"
EXP_LOG      = EXPLORE_DIR / "exploration_log.md"
HISTORY_FILE = EXPLORE_DIR / "history.json"
RESULTS_FILE = EXPLORE_DIR / "results.tsv"

MODEL        = os.environ.get("EXPLORER_MODEL", "claude-sonnet-4-5")
COOLING_RATE = float(os.environ.get("COOLING_RATE", "0.97"))
CONFIG_RATIO = float(os.environ.get("CONFIG_RATIO", "0.30"))
NPROC        = int(os.environ.get("NPROC_PER_NODE", "1"))
TIMEOUT      = int(os.environ.get("HARNESS_TIMEOUT", "720"))
MAX_RETRIES  = 3
COMPRESS_AT  = 10   # compress recent experiments into family summaries at this count

ALLOWED_FILES = {
    "architecture.py", "architecture.py.bak",
    "train_config.py", "train_config.py.bak",
    "harness.py", "orchestrator.py",
    "CONSTRAINTS.md", "exploration_log.md",
    "history.json", "results.tsv", "run.log",
}

# ── LLM ────────────────────────────────────────────────────────────────────

ARCH_SYSTEM = """You are a cutting-edge ML researcher competing in the OpenAI Parameter Golf challenge.
Your job: propose and implement a novel model architecture nobody on the leaderboard has tried.

Read CONSTRAINTS.md and exploration_log.md carefully before proposing.

Think: What unexplored territory exists in ML research? What architectures compress well?
What failed before but could work with a better implementation?

Rules for your output:
1. Start the file with: # HYPOTHESIS: [one sentence — why this will work]
2. Replace the ENTIRE file. No transformer code leftover.
3. ONLY use standard library + torch + numpy. No new imports, no helper files.
4. Output ONLY valid Python. No markdown, no prose outside code comments."""

CONFIG_SYSTEM = """You are an ML engineer tuning hyperparameters for a new architecture.
Read CONSTRAINTS.md and the current architecture.py before adjusting train_config.py.

Think: Does this architecture need different LR, batch size, optimizer? Should you add
get_optimizer() to architecture.py instead of using the default Muon+AdamW split?

Rules:
1. Preserve all REQUIRED fields in Config.
2. Output ONLY valid Python. No markdown."""

FIX_SYSTEM = """You are an ML engineer debugging a failed training run.
You wrote architecture.py or train_config.py and it crashed. Fix the bug.

Rules:
1. Output ONLY the fixed Python file. No markdown, no explanations.
2. Keep the same architecture idea — just fix the implementation bug.
3. Do not add new helper files or imports beyond torch/numpy/stdlib."""

COMPRESS_SYSTEM = """You are summarizing ML experiment results.
Given a list of recent experiments, update the Family Summaries section.
For each architecture family, write exactly one bullet:
  - FamilyName: best_bpb=X.XXXX, N attempts, [one-sentence key finding]
Sort by best_bpb ascending (best first). Include only families that appeared in the experiments.
Output ONLY the bullet list. No headers, no prose."""

INSIGHT_SYSTEM = """Analyze this ML experiment result. Output ONE sentence (max 15 words):
what does this result tell us about this architecture family? Output only the sentence."""


def call_llm(system: str, user: str, max_tokens: int = 4096) -> str | None:
    try:
        resp = litellm.completion(
            model=MODEL,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
            max_tokens=max_tokens,
        )
        text = resp.choices[0].message.content.strip()
        for fence in ("```python\n", "```python", "```\n", "```"):
            if text.startswith(fence):
                text = text[len(fence):]
        if text.endswith("```"):
            text = text[:-3]
        return text.strip()
    except Exception as e:
        print(f"[llm] {type(e).__name__}: {e}")
        return None


# ── File safety ─────────────────────────────────────────────────────────────

def enforce_whitelist():
    """Delete any file in explore/ not on the allowed list."""
    for item in EXPLORE_DIR.iterdir():
        if item.name.startswith("."):
            continue
        if item.name not in ALLOWED_FILES:
            if item.is_dir():
                shutil.rmtree(item)
                print(f"[enforce] removed directory: {item.name}/")
            else:
                item.unlink()
                print(f"[enforce] removed rogue file: {item.name}")


def backup():
    shutil.copy(ARCH_FILE, str(ARCH_FILE) + ".bak")
    shutil.copy(CONFIG_FILE, str(CONFIG_FILE) + ".bak")


def restore():
    shutil.copy(str(ARCH_FILE) + ".bak", ARCH_FILE)
    shutil.copy(str(CONFIG_FILE) + ".bak", CONFIG_FILE)


def extract_hypothesis(code: str) -> str:
    for line in code.splitlines():
        if "HYPOTHESIS:" in line.upper():
            return line.split(":", 1)[1].strip()
    return "No hypothesis stated."


def extract_arch_type(code: str) -> str:
    for line in code.splitlines():
        if line.startswith("class ") and "(" in line:
            return line.split("class ")[1].split("(")[0].strip()
    return "Unknown"


# ── Harness runner with retry ────────────────────────────────────────────────

def run_harness() -> tuple[dict | None, str]:
    """Returns (result_dict_or_None, error_text)."""
    log_path = EXPLORE_DIR / "run.log"
    cmd = f"torchrun --standalone --nproc_per_node={NPROC} {HARNESS_FILE}"
    try:
        with open(log_path, "w") as f:
            subprocess.run(cmd, shell=True, timeout=TIMEOUT,
                           stdout=f, stderr=f, check=False, env=os.environ.copy())
    except subprocess.TimeoutExpired:
        return None, "Timed out after {TIMEOUT}s."

    log = log_path.read_text()
    result = {}
    for line in log.splitlines():
        for key in ("val_bpb", "val_loss", "artifact_mb", "size_ok",
                    "size_penalty", "penalized_bpb", "num_params_M", "num_steps"):
            if line.strip().startswith(f"{key}:"):
                raw = line.split(":", 1)[1].strip()
                result[key] = (raw == "true") if raw in ("true", "false") else _try_float(raw)

    if "penalized_bpb" not in result:
        error_tail = "\n".join(log.splitlines()[-40:])
        return None, error_tail

    return result, ""


def _try_float(s):
    try: return float(s)
    except: return s


def run_with_retry(target_file: Path, original_code: str, system: str, is_arch: bool) -> tuple[dict | None, str, str]:
    """
    Run harness. On crash, feed error back to LLM and retry up to MAX_RETRIES.
    Returns (result, final_code, hypothesis).
    """
    current_code = original_code
    hypothesis   = extract_hypothesis(current_code)
    arch_type    = extract_arch_type(current_code)
    constraints_text = CONSTRAINTS.read_text()

    for attempt in range(MAX_RETRIES):
        enforce_whitelist()
        result, error = run_harness()

        if result is not None:
            return result, current_code, hypothesis

        if attempt == MAX_RETRIES - 1:
            print(f"[retry] failed after {MAX_RETRIES} attempts")
            return None, current_code, hypothesis

        print(f"[retry] attempt {attempt + 1}/{MAX_RETRIES} — asking LLM to fix crash")
        fix_user = (
            f"=== CONSTRAINTS.md ===\n{constraints_text}\n\n"
            f"=== Your code that crashed ===\n{current_code}\n\n"
            f"=== Error (last 40 lines of run.log) ===\n{error}\n\n"
            f"Fix the crash. Output only the corrected Python file."
        )
        fixed = call_llm(FIX_SYSTEM, fix_user)
        if not fixed:
            print("[retry] LLM gave no fix")
            continue

        current_code = fixed
        target_file.write_text(current_code)
        enforce_whitelist()
        hypothesis = extract_hypothesis(current_code)

    return None, current_code, hypothesis


# ── Exploration log ──────────────────────────────────────────────────────────

LOG_HEADER = """# Exploration Log

The LLM reads this before every proposal. It contains:
- **Family Summaries**: best result per architecture family, updated after compression
- **Recent Experiments**: last {n} full experiment records

---

## Family Summaries

*None yet.*

---

## Recent Experiments

*No experiments yet.*
""".format(n=COMPRESS_AT)


def init_log():
    if not EXP_LOG.exists():
        EXP_LOG.write_text(LOG_HEADER)


def _split_log(text: str) -> tuple[str, list[str]]:
    """Split log into (family_summaries_section, list_of_recent_entries)."""
    lines = text.splitlines()
    family_lines, recent_entries, current_entry = [], [], []
    in_family, in_recent = False, False

    for line in lines:
        if line.strip() == "## Family Summaries":
            in_family, in_recent = True, False
            continue
        if line.strip() == "## Recent Experiments":
            in_family, in_recent = False, True
            continue
        if line.startswith("### Experiment") and in_recent:
            if current_entry:
                recent_entries.append("\n".join(current_entry))
            current_entry = [line]
        elif in_recent:
            current_entry.append(line)
        elif in_family and not line.startswith("---"):
            family_lines.append(line)

    if current_entry:
        recent_entries.append("\n".join(current_entry))

    return "\n".join(family_lines).strip(), recent_entries


def _rebuild_log(family_section: str, recent_entries: list[str]) -> str:
    recent_text = "\n\n".join(recent_entries) if recent_entries else "*No recent experiments.*"
    return (
        "# Exploration Log\n\n"
        "The LLM reads this before every proposal.\n\n---\n\n"
        "## Family Summaries\n\n"
        f"{family_section or '*None yet.*'}\n\n"
        "---\n\n"
        "## Recent Experiments\n\n"
        f"{recent_text}\n"
    )


def compress_log_if_needed():
    """When Recent Experiments hits COMPRESS_AT, compress into Family Summaries."""
    text = EXP_LOG.read_text()
    family_section, recent_entries = _split_log(text)

    if len(recent_entries) < COMPRESS_AT:
        return

    print(f"[log] compressing {len(recent_entries)} recent entries into family summaries")
    compress_user = (
        f"Existing family summaries:\n{family_section}\n\n"
        f"Recent experiments to merge:\n" + "\n---\n".join(recent_entries)
    )
    new_family = call_llm(COMPRESS_SYSTEM, compress_user, max_tokens=800)
    if not new_family:
        print("[log] compression LLM call failed, keeping as-is")
        return

    # Keep only last 2 entries after compression so context isn't totally empty
    EXP_LOG.write_text(_rebuild_log(new_family, recent_entries[-2:]))
    print("[log] compression done")


def append_recent(step: int, is_arch: bool, arch_type: str, hypothesis: str,
                  result: dict | None, accepted: bool, temperature: float, insight: str):
    date_str = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    status = "KEPT" if accepted else "REJECTED"
    turn = "architecture" if is_arch else "config-tune"
    if result:
        res_str = (f"val_bpb={result['val_bpb']:.4f}, "
                   f"penalized={result['penalized_bpb']:.4f}, "
                   f"size={result.get('artifact_mb', 0):.2f}MB")
    else:
        res_str = "CRASH"

    entry = (
        f"### Experiment {step} — {date_str} | {status}\n"
        f"**Turn:** {turn} | **Arch:** {arch_type}\n"
        f"**Hypothesis:** {hypothesis}\n"
        f"**Result:** {res_str}\n"
        f"**SA:** {status} (T={temperature:.3f})\n"
        f"**Insight:** {insight}"
    )

    text = EXP_LOG.read_text()
    family_section, recent_entries = _split_log(text)
    recent_entries.append(entry)
    EXP_LOG.write_text(_rebuild_log(family_section, recent_entries))


def get_insight(arch_type, hypothesis, result, accepted) -> str:
    user = (
        f"Arch: {arch_type} | Hypothesis: {hypothesis}\n"
        f"val_bpb: {result.get('val_bpb','?') if result else 'CRASH'} | "
        f"Accepted: {accepted}"
    )
    return call_llm(INSIGHT_SYSTEM, user, max_tokens=40) or "No insight."


# ── TSV log ─────────────────────────────────────────────────────────────────

def ensure_tsv():
    if not RESULTS_FILE.exists():
        RESULTS_FILE.write_text("step\tval_bpb\tpenalized_bpb\tartifact_mb\tstatus\ttype\tarch\n")


def log_tsv(step, result, accepted, is_arch, arch_type):
    ensure_tsv()
    bpb = result.get("val_bpb", 9.99) if result else 9.99
    pen = result.get("penalized_bpb", 9.99) if result else 9.99
    mb  = result.get("artifact_mb", 0.0) if result else 0.0
    status = "keep" if accepted else ("crash" if not result else "discard")
    with open(RESULTS_FILE, "a") as f:
        f.write(f"{step}\t{bpb:.4f}\t{pen:.4f}\t{mb:.2f}"
                f"\t{status}\t{'arch' if is_arch else 'config'}\t{arch_type}\n")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    init_log()
    ensure_tsv()

    if HISTORY_FILE.exists():
        state = json.loads(HISTORY_FILE.read_text())
        print(f"[orchestrator] Resuming from step {state['step']}")
    else:
        state = {"step": 0, "temperature": 1.0,
                 "current_bpb": float("inf"), "best_bpb": float("inf")}
        print("[orchestrator] Fresh start")

    print(f"[orchestrator] model={MODEL} | T={state['temperature']:.3f} | best={state['best_bpb']:.4f}")

    constraints_text = CONSTRAINTS.read_text()

    while True:
        step         = state["step"] + 1
        temperature  = state["temperature"]
        current_bpb  = state["current_bpb"]
        is_arch_turn = random.random() > CONFIG_RATIO

        label = "ARCHITECTURE" if is_arch_turn else "CONFIG-TUNE"
        print(f"\n[step {step}] {label} | T={temperature:.3f} | current={current_bpb:.4f} | best={state['best_bpb']:.4f}")

        # Snapshot before modification
        backup()

        # Build context
        exp_log_text = EXP_LOG.read_text()
        arch_code    = ARCH_FILE.read_text()
        config_code  = CONFIG_FILE.read_text()

        # Call LLM for initial proposal
        if is_arch_turn:
            user_msg = (
                f"=== CONSTRAINTS.md ===\n{constraints_text}\n\n"
                f"=== exploration_log.md ===\n{exp_log_text}\n\n"
                f"=== Current architecture.py ===\n{arch_code}\n\n"
                f"=== train_config.py (reference) ===\n{config_code}\n\n"
                "Propose and implement your next architecture experiment."
            )
            new_code = call_llm(ARCH_SYSTEM, user_msg)
            target_file = ARCH_FILE
        else:
            user_msg = (
                f"=== CONSTRAINTS.md ===\n{constraints_text}\n\n"
                f"=== exploration_log.md ===\n{exp_log_text}\n\n"
                f"=== Current architecture.py ===\n{arch_code}\n\n"
                f"=== Current train_config.py ===\n{config_code}\n\n"
                "Tune train_config.py for the current architecture."
            )
            new_code = call_llm(CONFIG_SYSTEM, user_msg)
            target_file = CONFIG_FILE

        if not new_code:
            print("[orchestrator] No LLM response — skipping step")
            continue

        target_file.write_text(new_code)
        enforce_whitelist()

        arch_type  = extract_arch_type(ARCH_FILE.read_text())
        hypothesis = extract_hypothesis(new_code) if is_arch_turn else "Config tuning"
        print(f"[orchestrator] arch={arch_type} | {hypothesis[:80]}")

        # Run harness with crash retry
        print("[orchestrator] Running harness...")
        result, final_code, hypothesis = run_with_retry(target_file, new_code, FIX_SYSTEM, is_arch_turn)
        enforce_whitelist()  # clean up after harness run too

        if result is None:
            print("[orchestrator] Giving up on this idea — restoring backup")
            restore()
            insight = "Crashed after all retry attempts — implementation bug."
            append_recent(step, is_arch_turn, arch_type, hypothesis, None, False, temperature, insight)
            log_tsv(step, None, False, is_arch_turn, arch_type)
            state["step"] = step
            HISTORY_FILE.write_text(json.dumps(state))
            continue

        new_bpb = result["penalized_bpb"]
        raw_bpb = result.get("val_bpb", new_bpb)
        print(f"[orchestrator] val_bpb={raw_bpb:.4f} | penalized={new_bpb:.4f} | {result.get('artifact_mb',0):.2f}MB")

        # SA accept/reject
        if new_bpb < current_bpb:
            accepted = True
            print(f"[orchestrator] KEPT ✓  improvement={current_bpb - new_bpb:.4f}")
        else:
            delta = new_bpb - current_bpb
            prob  = math.exp(-delta / max(temperature, 1e-6))
            accepted = random.random() < prob
            print(f"[orchestrator] {'KEPT (SA)' if accepted else 'REJECTED'} | worse_by={delta:.4f} | p={prob:.3f}")
            if not accepted:
                restore()

        insight = get_insight(arch_type, hypothesis, result, accepted)

        append_recent(step, is_arch_turn, arch_type, hypothesis,
                      result, accepted, temperature, insight)
        compress_log_if_needed()
        log_tsv(step, result, accepted, is_arch_turn, arch_type)

        if accepted:
            state["current_bpb"] = new_bpb
        if raw_bpb < state["best_bpb"]:
            state["best_bpb"] = raw_bpb
            print(f"[orchestrator] *** NEW BEST: {raw_bpb:.4f} ***")

        state["step"]        = step
        state["temperature"] = temperature * COOLING_RATE
        HISTORY_FILE.write_text(json.dumps(state))
        time.sleep(2)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify syntax locally**

```bash
python -c "import ast; ast.parse(open('explore/orchestrator.py').read()); print('ok')"
```
Expected: `ok`

---

## Task 6: End-to-End Smoke Test (on H100)

- [ ] **Step 1: Run one step with short harness budget**

```bash
cd /workspace/parameter-golf
MAX_WALLCLOCK_SECONDS=120 NPROC_PER_NODE=1 \
ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY \
python explore/orchestrator.py
```

Let it complete one step (LLM proposes architecture → harness runs → logged), then Ctrl+C.

- [ ] **Step 2: Verify the exploration log was updated**

```bash
cat explore/exploration_log.md
```
Expected: the initial "no experiments yet" section plus one new `### Experiment 1` entry with hypothesis, result, insight.

- [ ] **Step 3: Verify architecture.py changed**

```bash
head -5 explore/architecture.py
```
Expected: a `# HYPOTHESIS:` comment at the top describing what the LLM tried.

- [ ] **Step 4: Verify resume works**

```bash
MAX_WALLCLOCK_SECONDS=120 NPROC_PER_NODE=1 ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY \
python explore/orchestrator.py
```
Expected first line: `[orchestrator] Resuming from step 1`

---

## Task 7: Full Overnight Run

- [ ] **Step 1: Launch on H100**

```bash
cd /workspace/parameter-golf
pip install litellm

python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 10

export ANTHROPIC_API_KEY="sk-ant-..."
export NPROC_PER_NODE=8
export MAX_WALLCLOCK_SECONDS=600
export COOLING_RATE=0.97
export EXPLORER_MODEL=claude-sonnet-4-5

nohup python explore/orchestrator.py > explore/orchestrator.log 2>&1 &
echo "PID: $!"
```

- [ ] **Step 2: Monitor**

```bash
tail -f explore/exploration_log.md    # read the LLM's research journal
tail -f explore/results.tsv           # BPB scores as they come in
cat explore/history.json              # SA state
```

- [ ] **Step 3: Morning analysis**

```bash
# Best runs by val_bpb:
(head -1 explore/results.tsv; tail -n +2 explore/results.tsv | sort -t$'\t' -k2 -n) | head -11

# Count kept vs rejected:
grep -c "keep" explore/results.tsv
grep -c "discard" explore/results.tsv

# Read the log to see what the LLM discovered:
cat explore/exploration_log.md
```

---

## How to read the morning results

**What you're looking for:**
- Any non-transformer family that scores < 1.20 BPB = there's a new hill, dig into it
- A family that keeps getting rejected but the BPB is trending down = needs more tuning steps, increase SA temperature for that family
- Crashes = the LLM had a bad implementation idea, the log will tell you what failed

**Next steps after a promising result:**
1. Note which architecture type scored best from `exploration_log.md`
2. Reset `history.json` with high temperature: `{"step": N, "temperature": 0.8, "current_bpb": X, "best_bpb": X}`
3. Rerun focused on that family: the LLM will see the log showing that family worked and keep iterating on it
4. When ready for leaderboard submission: run 3 seeds with `NPROC_PER_NODE=8 MAX_WALLCLOCK_SECONDS=600`

**Realistic overnight numbers:**
- 8xH100, 10min/run: ~13min per step total → ~37 steps overnight
- 70/30 split → ~26 architecture proposals, ~11 config-tune steps
- Expect 5–15 unique architecture families actually tested (some crash, some get config-tuned)
