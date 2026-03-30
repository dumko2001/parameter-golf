import itertools

VOCAB = 1024
BIGRAM_DIM = 112
BIGRAM_VOCAB = 3072
BASELINE_STEP_MS = 83.0
BASELINE_PARAMS = 35.5e6

def calc_params(d, layers, mlp_mult):
    tok_emb = VOCAB * d
    bigram = BIGRAM_VOCAB * BIGRAM_DIM
    c_q = d * d
    c_k = d * (d // 2)
    c_v = d * (d // 2)
    proj = d * d
    attn_params = c_q + c_k + c_v + proj 
    fc = d * int(mlp_mult * d) * 2
    mlp_proj = int(mlp_mult * d) * d
    mlp_params = fc + mlp_proj
    layer_params = attn_params + mlp_params
    return tok_emb + bigram + (layers * layer_params)

def simulate_size_gptqlite(params):
    bits_per_param = 3.56 # established baseline ratio via LZMA
    return (params * bits_per_param) / 8 / (1024 ** 2)

print(f"{'D':<5} | {'Layers':<8} | {'MLP':<6} | {'Params (M)':<12} | {'Size (MB)':<11} | {'Step Time':<12}")
print("-" * 75)

for d in [512, 448, 384]:
    for layers in [11, 14, 16, 20]:
        for mlp_mult in [2.0, 3.0, 4.0]:
            params = calc_params(d, layers, mlp_mult) / 1e6
            size = simulate_size_gptqlite(params * 1e6)
            step_ms = BASELINE_STEP_MS * (params * 1e6 / BASELINE_PARAMS)
            if size <= 15.8: # Must strictly be under 16MB with buffer
                p_str = f"{params:.1f}M"
                g_str = f"{size:.1f} MB"
                st_str = f"{step_ms:.1f} ms"
                print(f"{d:<5} | {layers:<8} | {mlp_mult:<6} | {p_str:<12} | {g_str:<11} | {st_str:<12}")
