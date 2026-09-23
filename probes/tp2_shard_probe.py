#!/usr/bin/env python3
"""TP2 feasibility #2: can the quantized weight be sharded (keeping 4-bit) and
dequantize to the exact same values as the full weight's slices?

TP splits are of two kinds:
  * N-split (output dim): qkv_proj [21504,5376]->2x[10752,5376], fc1 [28672,5376]->2x[14336,5376]
  * K-split (input dim):  out_proj [5376,7168]->2x[5376,3584], fc2 [5376,14336]->2x[5376,7168]

For each, slice qdata + scale + s_channel (+ correction) consistently, dequantize,
and compare against the full weight's dequantize() sliced the same way.

    .venv/bin/python tp2_shard_probe.py
"""
import dataclasses
import os
import sys

COMFY = os.path.expanduser("~/ComfyUI")
sys.path.insert(0, COMFY)
sys.path.insert(0, os.path.join(COMFY, "custom_nodes", "minimax-h3-chunk-star7"))

import torch  # noqa: E402

CKPT = "minimax_h3_ref2va_pruned_w4a8_mixed.safetensors"


def shard_and_check(name, w, dim):
    """Split a quantized weight along `dim` (0=N, 1=K) and compare dequantized halves."""
    from comfy_kitchen.tensor.w4a8_int8 import AsymW4A8Int8Layout as L

    qdata, scale, s_channel, correction, codebook = L.get_plain_tensors(w)
    full = w.dequantize()
    n, k = full.shape
    print(f"\n=== {name} [{n},{k}] split dim={dim} ===")
    print(f"  qdata {tuple(qdata.shape)} {qdata.dtype} | scale {tuple(scale.shape)} | "
          f"s_channel {tuple(s_channel.shape)} | correction "
          f"{tuple(correction.shape) if correction is not None else None}")

    params = w._params
    if dim == 0:                      # N-split: output channels
        mid = n // 2
        halves = [
            (qdata[:mid], scale[:mid], s_channel[:mid],
             correction[:, :mid] if correction is not None else None,
             params),
            (qdata[mid:], scale[mid:], s_channel[mid:],
             correction[:, mid:] if correction is not None else None,
             params),
        ]
        refs = [full[:mid], full[mid:]]
    else:                             # K-split: input dim, must respect convrot groups
        # K must split at a multiple of convrot_groupsize so the rotation stays valid
        cg = params.convrot_groupsize
        mid = (k // 2) // cg * cg
        g = params.group_size
        q_groups = k // g
        scale_cols = mid // g
        halves = [
            (qdata[:, :mid // 2], scale[:, :scale_cols], s_channel, correction, params),
            (qdata[:, mid // 2:], scale[:, scale_cols:], s_channel, correction, params),
        ]
        refs = [full[:, :mid], full[:, mid:]]

    for idx, (qd, sc, sch, corr, p) in enumerate(halves):
        new_params = dataclasses.replace(
            p,
            scale=sc, s_channel=sch, correction=corr,
            orig_shape=tuple(refs[idx].shape),
        )
        got = L.dequantize(qd, new_params)
        ref = refs[idx]
        d = (got.float() - ref.float()).abs()
        ok = d.max().item() == 0.0
        print(f"  half{idx}: got {tuple(got.shape)} ref {tuple(ref.shape)} "
              f"max|diff|={d.max().item():.3e} {'BIT-EXACT' if ok else 'DIFF'} "
              f"ref_absmax={ref.float().abs().max().item():.4f}")


def main():
    import folder_paths
    import adaptive_loader as al

    path = folder_paths.get_full_path_or_raise("diffusion_models", CKPT)
    al._neutralize_process_wide_h3_conflicts()
    dm = al._load_h3_native_fp16(path).model.diffusion_model
    blk = dm.blocks[0]

    shard_and_check("qkv_proj (N)", blk.attn.qkv_proj.weight, 0)
    shard_and_check("out_proj (K)", blk.attn.out_proj.weight, 1)
    shard_and_check("fc1 (N)", blk.mlp.fc1.weight, 0)
    shard_and_check("fc2 (K)", blk.mlp.fc2.weight, 1)


if __name__ == "__main__":
    main()
