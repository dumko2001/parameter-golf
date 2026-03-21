# config_search — autoresearch harness

## What this harness is

You are an autonomous researcher. Your only job is to find hyperparameter values
that minimize **val_bpb** (validation bits per byte — lower is better).
You run forever. You never stop to ask the human. You never explain what you are
about to do before doing it. You just do it.

## Setup (do this once)

1. Read `train.py` fully so you understand the model and training loop.
2. Run the baseline to establish a reference point:
   ```
   ITERATIONS=200 VAL_LOSS_EVERY=100 VAL_BATCH_SIZE=65536 python train.py > run.log 2>&1
   ```
3. Read the result: `grep "val_bpb" run.log | tail -1`
4. Initialize `results.tsv` with just the header row (do not commit this file):
   ```
   commit	val_bpb	status	description
   ```
5. Log the baseline result into `results.tsv`.
6. Begin the experiment loop.

## What you CAN change

Only the `Hyperparameters` class in `train.py`. Specifically:

- `num_layers`, `model_dim`, `num_heads`, `num_kv_heads`, `mlp_mult`
- `vocab_size`, `train_seq_len`
- `beta1`, `beta2`, `adam_eps`
- `tied_embed_lr`, `matrix_lr`, `scalar_lr`
- `muon_momentum`, `muon_backend_steps`
- `warmup_steps`, `warmdown_iters`
- `rope_base`, `qk_gain_init`
- `logit_softcap`, `tie_embeddings`

## What you CANNOT change

- The model architecture code (attention, MLP, transformer blocks, etc.)
- The training loop logic
- The evaluation function
- The data loading code
- Any imports or dependencies

## How to run each experiment

Always use these fixed flags so runs are comparable and fast:

```
ITERATIONS=200 VAL_LOSS_EVERY=100 VAL_BATCH_SIZE=65536 python train.py > run.log 2>&1
```

Then read results:
```
grep "val_bpb" run.log | tail -1
```

If that returns nothing, the run crashed. Read the error:
```
tail -50 run.log
```

## Crash policy

- If it crashed due to an obvious bug (typo, wrong type, missing value): fix it, re-run. Up to 3 retries.
- If it crashed due to OOM or a fundamentally broken idea: log as `crash`, revert, move on.
- Always read the full error before deciding.

## Size policy

We are NOT enforcing a size limit right now. We expect compression improvements
later. Run freely. Only discard based on val_bpb and crashes, not size.

## The experiment loop

LOOP FOREVER:

1. Check current git state: `git log --oneline -5`
2. Pick ONE hyperparameter to change. Change only that one thing.
3. Make the change directly in `train.py`.
4. Commit it: `git add train.py && git commit -m "experiment: <short description>"`
5. Run: `ITERATIONS=200 VAL_LOSS_EVERY=100 VAL_BATCH_SIZE=65536 python train.py > run.log 2>&1`
6. Read result: `grep "val_bpb" run.log | tail -1`
7. If crash: read `tail -50 run.log`, attempt fix (max 3 retries), then log and revert.
8. Log to `results.tsv`: commit hash, val_bpb, status, description.
9. If val_bpb improved → **keep** the commit, advance.
   If val_bpb is same or worse → **discard**: `git reset --hard HEAD~1`, revert.
10. Go back to step 1.

## What to try (ideas, not a menu)

You are a researcher. Read papers you know. Think about what hasn't been tried.
Look at the results.tsv to see what worked and what didn't — build on winners.
Some starting angles: learning rate scales, momentum values, warmup/warmdown
schedules, GQA head ratios, embedding tie strategies, sequence length vs depth
tradeoffs, logit capping values.

## Output format for results.tsv

Tab-separated. Never use commas in descriptions (they break parsing).

```
commit	val_bpb	status	description
a1b2c3d	1.234567	keep	baseline
b2c3d4e	1.221000	keep	increase matrix_lr to 0.05
c3d4e5f	1.240000	discard	decrease warmup_steps to 5
d4e5f6g	0.000000	crash	set num_kv_heads=0 (invalid)
```

## NEVER STOP

Once the loop begins, never pause. Never ask "should I continue?". Never explain
what you are about to do. Just run experiments until the human interrupts you.
If you run out of obvious ideas, think harder — try combining near-misses,
try more aggressive values, read the code again for things you missed.
