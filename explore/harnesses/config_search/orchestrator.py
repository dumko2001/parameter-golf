#!/usr/bin/env python3
"""
Autonomous hyperparameter search orchestrator.
Calls MiniMax via OpenCode Go API, proposes one config change at a time,
runs train.py, logs result, keeps if better, reverts if worse.
Runs forever until you Ctrl+C.
"""
import os
import re
import subprocess
import sys
import time
from pathlib import Path



# ── Config ────────────────────────────────────────────────────────────────────
HARNESS_DIR   = Path(__file__).parent
TRAIN_PY      = HARNESS_DIR / "train.py"
RESULTS_TSV   = HARNESS_DIR / "results.tsv"
EXPLORE_LOG   = HARNESS_DIR / "exploration_log.md"
RUN_LOG       = HARNESS_DIR / "run.log"
PROGRAM_MD    = HARNESS_DIR / "program.md"

# API_KEY       = os.environ["OPENCODE_API_KEY"]  # Removed: using opencode CLI directly
# BASE_URL      = "https://opencode.ai/zen/go/v1" # Removed: using opencode CLI directly
MODEL         = "opencode-go/minimax-m2.7"

MAX_RETRIES   = 3   # retries on crash before giving up on an idea

# Train command — 200 steps, lightweight for M1 8GB
TRAIN_CMD = [
    sys.executable, str(TRAIN_PY),
]
TRAIN_ENV = {
    **os.environ,
    "ITERATIONS":        "200",
    "VAL_LOSS_EVERY":    "100",
    "VAL_BATCH_SIZE":    "16384",
    "TRAIN_BATCH_TOKENS":"8192",
    "GRAD_ACCUM_STEPS":  "1",
    "TRAIN_LOG_EVERY":   "50",
    "WARMUP_STEPS":      "10",
    "WARMDOWN_ITERS":    "80",
    "MAX_WALLCLOCK_SECONDS": "300",
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def git(*args) -> str:
    result = subprocess.run(["git", "-C", str(HARNESS_DIR), *args],
                            capture_output=True, text=True)
    return result.stdout.strip()

def extract_hyperparams_class(code: str) -> str:
    """Extract just the Hyperparameters class body from train.py."""
    match = re.search(r"(class Hyperparameters:.*?)(?=\nclass |\n# =====|\Z)",
                      code, re.DOTALL)
    return match.group(1).strip() if match else ""

def replace_hyperparams_class(code: str, new_class: str) -> str:
    """Splice a new Hyperparameters class into train.py."""
    return re.sub(
        r"class Hyperparameters:.*?(?=\nclass |\n# =====|\Z)",
        new_class + "\n",
        code, flags=re.DOTALL
    )

def run_training() -> tuple[float | None, str]:
    """Run train.py. Returns (val_bpb, error_tail). val_bpb is None on crash."""
    with open(RUN_LOG, "w") as log:
        proc = subprocess.run(TRAIN_CMD, env=TRAIN_ENV,
                              stdout=log, stderr=subprocess.STDOUT,
                              cwd=str(HARNESS_DIR))

    log_text = RUN_LOG.read_text()

    # Extract last val_bpb
    matches = re.findall(r"val_bpb[=:\s]+([\d.]+)", log_text)
    if matches:
        return float(matches[-1]), ""

    # Crashed
    error_tail = "\n".join(log_text.splitlines()[-40:])
    return None, error_tail

def read_exploration_log() -> str:
    if EXPLORE_LOG.exists():
        return EXPLORE_LOG.read_text()
    return "No experiments yet."

def append_exploration_log(entry: str):
    with open(EXPLORE_LOG, "a") as f:
        f.write(entry + "\n")

def append_results_tsv(commit: str, val_bpb: float | None, status: str, desc: str):
    bpb_str = f"{val_bpb:.6f}" if val_bpb is not None else "0.000000"
    with open(RESULTS_TSV, "a") as f:
        f.write(f"{commit}\t{bpb_str}\t{status}\t{desc}\n")

def ask_llm(current_class: str, log: str, error: str = "") -> str:
    """Ask LLM to propose a new Hyperparameters class using opencode CLI."""

    system = (
        "You are an autonomous hyperparameter optimization researcher for a small language model. "
        "Your only goal is to minimize val_bpb (validation bits per byte — lower is better). "
        "You change EXACTLY ONE hyperparameter value at a time. "
        "You output ONLY the complete replacement Python class — no explanation, no markdown, "
        "no commentary. Just the raw Python class starting with 'class Hyperparameters:'. "
        "Read the exploration log carefully. Do not repeat experiments that already failed."
    )

    error_block = ""
    if error:
        error_block = f"\n\n## CRASH ERROR FROM LAST ATTEMPT\nFix this and retry:\n```\n{error}\n```"

    user = f"""## Exploration Log (what's been tried so far)
{log}

## Current Hyperparameters class
```python
{current_class}
```
{error_block}

## Your task
Propose the NEXT experiment. Change exactly ONE hyperparameter value.
Pick something that hasn't been tried or builds on a promising result.
Think like a researcher: consider learning rates, momentum, model width/depth ratios,
warmup schedules, GQA ratios, logit capping, rope base, embedding strategies.

Output ONLY the complete new Hyperparameters class as valid Python. No explanation."""

    prompt = f"{system}\n\n{user}"
    
    cmd = [
        "opencode", "run", prompt,
        "--model", MODEL,
        "--thinking", "false"
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        content = result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"[orchestrator] opencode CLI error: {e.stderr}")
        return ""

    # Strip markdown code fences if present
    content = re.sub(r"^```python\s*", "", content, flags=re.MULTILINE)
    content = re.sub(r"^```\s*", "", content, flags=re.MULTILINE)
    content = re.sub(r"\s*```$", "", content, flags=re.MULTILINE)

    return content.strip()

# ── Main loop ─────────────────────────────────────────────────────────────────

def main():
    print("[orchestrator] Starting config_search autoresearch loop")
    print(f"[orchestrator] Harness: {HARNESS_DIR}")

    # Ensure results.tsv has header
    if not RESULTS_TSV.exists() or RESULTS_TSV.stat().st_size == 0:
        RESULTS_TSV.write_text("commit\tval_bpb\tstatus\tdescription\n")

    # Ensure exploration log exists
    if not EXPLORE_LOG.exists():
        EXPLORE_LOG.write_text("# Config Search Exploration Log\n\n")


    # Run baseline first if log is empty
    if "baseline" not in read_exploration_log():
        print("[orchestrator] Running baseline...")
        val_bpb, err = run_training()
        commit = git("rev-parse", "--short", "HEAD")
        if val_bpb is not None:
            best_bpb = val_bpb
            append_results_tsv(commit, val_bpb, "baseline", "baseline — unmodified defaults")
            append_exploration_log(
                f"## baseline | val_bpb={val_bpb:.4f} | commit={commit}\n"
                f"Unmodified defaults. This is the reference point.\n"
            )
            print(f"[orchestrator] Baseline val_bpb={val_bpb:.4f}")
        else:
            print(f"[orchestrator] BASELINE CRASHED:\n{err}")
            print("[orchestrator] Fix train.py before running the loop.")
            sys.exit(1)
    else:
        # Recover best_bpb from results.tsv
        lines = [l for l in RESULTS_TSV.read_text().splitlines()
                 if l and not l.startswith("commit") and "\t" in l]
        kept = [float(l.split("\t")[1]) for l in lines
                if l.split("\t")[2] in ("keep", "baseline")]
        best_bpb = min(kept) if kept else float("inf")
        print(f"[orchestrator] Resuming. Best so far: {best_bpb:.4f}")

    step = 0
    while True:
        step += 1
        print(f"\n[orchestrator] ── Step {step} ──────────────────────────")

        original_code = TRAIN_PY.read_text()
        current_class = extract_hyperparams_class(original_code)

        error_context = ""
        new_class = ""
        val_bpb = None
        status = "discard"
        description = ""

        for attempt in range(1, MAX_RETRIES + 1):
            print(f"[orchestrator] Asking {MODEL} for next experiment (attempt {attempt})...")
            new_class = ask_llm(current_class,
                                read_exploration_log(), error_context)

            # Sanity check: must look like a class
            if not new_class.startswith("class Hyperparameters:"):
                print(f"[orchestrator] LLM output didn't start with class Hyperparameters:, retrying...")
                error_context = "Your previous output didn't start with 'class Hyperparameters:'. Output only the class."
                continue

            # Apply the change
            new_code = replace_hyperparams_class(original_code, new_class)
            TRAIN_PY.write_text(new_code)

            # Extract what changed for the description
            old_lines = set(current_class.splitlines())
            new_lines = set(new_class.splitlines())
            changed = [l.strip() for l in new_lines - old_lines
                       if l.strip() and not l.strip().startswith("#")]
            description = "; ".join(changed[:2]) if changed else "unknown change"

            print(f"[orchestrator] Change: {description}")
            print(f"[orchestrator] Running train.py...")

            val_bpb, error_tail = run_training()

            if val_bpb is not None:
                break  # success

            # Crashed
            print(f"[orchestrator] CRASH (attempt {attempt}/{MAX_RETRIES})")
            print(error_tail[-500:])
            error_context = error_tail
            TRAIN_PY.write_text(original_code)  # revert before retry

            if attempt == MAX_RETRIES:
                status = "crash"
                description = f"crash after {MAX_RETRIES} retries: {description}"

        # Evaluate result
        if val_bpb is not None:
            improved = val_bpb < best_bpb
            status = "keep" if improved else "discard"
            print(f"[orchestrator] val_bpb={val_bpb:.4f} | best={best_bpb:.4f} | {status.upper()}")

            if improved:
                best_bpb = val_bpb
                # Commit the improvement
                git("add", "train.py")
                git("commit", "-m", f"autoresearch: {description[:72]}")
                commit = git("rev-parse", "--short", "HEAD")
            else:
                # Revert
                TRAIN_PY.write_text(original_code)
                commit = git("rev-parse", "--short", "HEAD")
        else:
            # Already reverted above during retry loop
            commit = git("rev-parse", "--short", "HEAD")
            TRAIN_PY.write_text(original_code)

        append_results_tsv(commit, val_bpb, status, description)
        append_exploration_log(
            f"## step {step} | val_bpb={val_bpb:.4f if val_bpb else 'crash'} | "
            f"status={status} | commit={commit}\n"
            f"Change: {description}\n"
        )

        time.sleep(1)  # brief pause so logs flush


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[orchestrator] Stopped by user.")
