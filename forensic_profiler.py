
import os
import sys
import torch
import time
from torch.profiler import profile, record_function, ProfilerActivity

# Add the project directory to sys.path
sys.path.insert(0, os.path.join(os.getcwd(), "compression-iteration-worktree"))
import train_gpt

def profile_model():
    print("=== 2026 FORenSIC PROFILER ===")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    hp = train_gpt.Hyperparameters()
    
    # Initialize the CURRENT CHAMPION MODEL
    model = train_gpt.GPT(
        vocab_size=hp.vocab_size,
        num_layers=hp.num_layers,
        model_dim=hp.model_dim,
        num_heads=hp.num_heads,
        num_kv_heads=hp.num_kv_heads,
        mlp_mult=hp.mlp_mult,
        tie_embeddings=hp.tie_embeddings,
        tied_embed_init_std=hp.tied_embed_init_std,
        rope_base=hp.rope_base,
        logit_softcap=hp.logit_softcap,
        qk_gain_init=hp.qk_gain_init,
    ).to(device).bfloat16()
    
    # Prepare dummy data
    x = torch.randint(0, hp.vocab_size, (8, 1024), device=device)
    y = torch.randint(0, hp.vocab_size, (8, 1024), device=device)
    
    optimizer = train_gpt.Muon(model.parameters(), lr=0.02, momentum=0.95, backend_steps=5)

    print("\nWarmup steps...")
    for _ in range(5):
        loss = model(x, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    print("\nProfiling 1 training step...")
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
        with record_function("model_forward"):
            loss = model(x, y)
        with record_function("model_backward"):
            loss.backward()
        with record_function("optimizer_step"):
            optimizer.step()

    print("\n--- TOP 10 GPU OPERATIONS ---")
    print(prof.key_averages().sort_by("cuda_time_total", ascending=False)[:10])
    
    print("\n--- TOP 10 CPU OPERATIONS (Wait Times) ---")
    print(prof.key_averages().sort_by("cpu_time_total", ascending=False)[:10])

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA not found. Profiling on CPU (Limited visibility).")
    profile_model()
