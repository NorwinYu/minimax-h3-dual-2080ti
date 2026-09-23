#!/usr/bin/env python3
"""Compare dual_block against the REAL FP16-Exact-Fix block forward.

The earlier probe compared against the stock block, which is NOT what the model
runs: with the Star7 loader the block forward is the FP16-Exact wrapper (FP32
residual stream). This one calls patch_model() first, so both sides run the
patched forward -- the same thing the sampler runs.

    .venv/bin/python fp16_dual_probe.py
"""
import os
import sys

COMFY = os.path.expanduser("~/ComfyUI")
sys.path.insert(0, COMFY)
sys.path.insert(0, os.path.join(COMFY, "custom_nodes", "minimax-h3-chunk-star7"))
sys.path.insert(0, os.path.join(COMFY, "custom_nodes", "comfyui-h3-ulysses2"))

import torch  # noqa: E402

CKPT = "minimax_h3_ref2va_pruned_w4a8_mixed.safetensors"
S = 17260
DEV_A, DEV_B = "cuda:0", "cuda:1"


def main():
    import folder_paths
    import adaptive_loader as al
    import comfy.model_management as mm
    from comfy.ldm.minimax.model import rope_rotation_table
    from dual_block import dual_block

    path = folder_paths.get_full_path_or_raise("diffusion_models", CKPT)
    al._neutralize_process_wide_h3_conflicts()
    p1 = al._load_h3_native_fp16(path)
    p2 = al._load_h3_native_fp16(path)

    p1.patch_model(device_to=torch.device(DEV_A))
    p2.patch_model(device_to=torch.device(DEV_B))
    dm_a = p1.model.diffusion_model
    dm_b = p2.model.diffusion_model
    blk_a, blk_b = dm_a.blocks[0], dm_b.blocks[0]
    print(f"block forward types: a={type(blk_a.forward).__name__} b={type(blk_b.forward).__name__}")
    print(f"out_proj a={type(blk_a.attn.out_proj.forward).__name__} "
          f"b={type(blk_b.attn.out_proj.forward).__name__}")

    torch.manual_seed(0)
    # fp16 input, as _forward's assembly produces
    x = (torch.randn(S, dm_a.hidden_size, dtype=torch.float32) * 0.5).half().to(DEV_A)
    t_vals = torch.tensor([0.05, 0.5], dtype=torch.float32, device=DEV_A)
    table = mm.cast_to(dm_a.adaln_t_table, device=DEV_A)
    pos = t_vals.clamp(0.0, 1.0) * (table.shape[0] - 1)
    i0 = pos.floor().long().clamp(max=table.shape[0] - 2)
    t_emb = torch.lerp(table[i0], table[i0 + 1], (pos - i0).unsqueeze(1))
    pos_ids = torch.zeros(S, 3, dtype=torch.float64)
    rf_a = rope_rotation_table(dm_a.rope_freqs(pos_ids, DEV_A), torch.float16)

    third = S // 3
    segs = [(0, third, 0), (third, S, 1)]

    with torch.cuda.device(0):
        y_ref = blk_a(x.clone(), t_emb, segs, rf_a, transformer_options={})
    torch.cuda.synchronize()
    print(f"reference (patched blk.forward): {tuple(y_ref.shape)} {y_ref.dtype} "
          f"finite={bool(torch.isfinite(y_ref).all())} "
          f"absmax={y_ref.float().abs().max().item():.4f}")

    try:
        y = dual_block(blk_a, blk_b, DEV_A, DEV_B, x.clone(), t_emb, segs, rf_a)
        torch.cuda.synchronize()
        d = (y.float() - y_ref.float()).abs()
        print(f"dual_block                   : {tuple(y.shape)} {y.dtype} "
              f"finite={bool(torch.isfinite(y).all())}")
        print(f"max|diff|={d.max().item():.3e}  mean|diff|={d.mean().item():.3e}  "
              f"rel={d.max().item() / max(y_ref.float().abs().max().item(), 1e-9):.3e}")
    except Exception as e:                                   # noqa: BLE001
        import traceback
        traceback.print_exc()
        print(f"dual_block FAILED {type(e).__name__}: {e}")
        return

    # ---- timing: is the fp32-residual dual block faster with RESIDENT weights? ----
    import statistics
    def timed(fn, repeats=3, warmup=1):
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        ts = []
        for _ in range(repeats):
            a, b = torch.cuda.Event(True), torch.cuda.Event(True)
            a.record(); fn(); b.record()
            torch.cuda.synchronize()
            ts.append(a.elapsed_time(b))
        return statistics.median(ts)

    with torch.cuda.device(0):
        t1 = timed(lambda: blk_a(x.clone(), t_emb, segs, rf_a, transformer_options={}))
    t2 = timed(lambda: dual_block(blk_a, blk_b, DEV_A, DEV_B, x.clone(), t_emb, segs, rf_a))
    print(f"[timing] patched blk.forward (1 GPU)  {t1:8.2f} ms")
    print(f"[timing] dual_block (2 GPUs, resident) {t2:8.2f} ms   speedup {t1 / t2:.2f}x   (S={S})")


if __name__ == "__main__":
    main()
