import torch
import lzma
import io
import sys
import unittest.mock

# Mock to bypass CUDA requirement in offline testing
mock_cuda = unittest.mock.MagicMock()
mock_cuda.is_available.return_value = False
sys.modules['torch.cuda'] = mock_cuda
sys.modules['flash_attn_interface'] = unittest.mock.MagicMock()

import train_gpt_synth as tgs

def evaluate_sniper():
    h = tgs.Hyperparameters()
    
    print(f"Testing Synthetic Architecture...")
    print(f"Dim: {h.model_dim} | Layers: {h.num_layers} | MLP Mult: {h.mlp_mult}x | MTP Heads: {h.mtp_num_heads} | XSA: {h.xsa_last_n}")
    
    model = tgs.GPT(
        vocab_size=h.vocab_size,
        num_layers=h.num_layers,
        model_dim=h.model_dim,
        num_heads=h.num_heads,
        num_kv_heads=h.num_kv_heads,
        mlp_mult=h.mlp_mult,
        tie_embeddings=h.tie_embeddings,
        tied_embed_init_std=h.tied_embed_init_std,
        logit_softcap=h.logit_softcap,
        rope_base=h.rope_base,
        qk_gain_init=h.qk_gain_init,
        mtp_num_heads=h.mtp_num_heads,
        mtp_loss_weight=h.mtp_loss_weight,
        bigram_vocab_size=h.bigram_vocab_size,
        bigram_dim=h.bigram_dim,
        xsa_last_n=h.xsa_last_n,
        rope_dims=h.rope_dims,
        ln_scale=h.ln_scale,
        dtg=h.dtg_enabled,
        ve_enabled=h.ve_enabled,
        ve_dim=h.ve_dim,
        ve_layers=h.ve_layers,
        gated_attention=h.gated_attention,
        value_residual=h.value_residual,
    )
    
    full_state_dict = model.state_dict()
    export_sd = {k: v for k, v in full_state_dict.items() if "mtp_heads" not in k}
    
    # Calculate exact parameter count being exported
    total_params = sum(t.numel() for t in export_sd.values())
    print(f"Export Parameters: {total_params / 1e6:.1f} M")
    
    unbanked_sd = tgs._unbank_state_dict(export_sd, h.num_layers)
    quant_result, quant_meta = tgs.mixed_quantize_int6(unbanked_sd, {"mlp", "attn"})
    
    quant_buf = io.BytesIO()
    torch.save({"w": quant_result, "m": quant_meta}, quant_buf)
    quant_blob = lzma.compress(quant_buf.getvalue(), preset=6)
    
    size_mb = len(quant_blob) / (1024 ** 2)
    print(f"Untrained Mock Artifact Size: {size_mb:.2f} MB")
    
if __name__ == "__main__":
    evaluate_sniper()
