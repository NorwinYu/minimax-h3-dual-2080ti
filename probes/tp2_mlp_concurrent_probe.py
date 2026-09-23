#!/usr/bin/env python3
"""Is the TP2 MLP serialized, or does it just slow down when both GPUs run it?

Measures the real MLP chain (fc1 + swiglu + fc2) from block 0's sharded weights:
  dev0 alone / dev1 alone / both launched back-to-back with one sync.

    .venv/bin/python tp2_mlp_concurrent_probe.py
"""
import os
import statistics
import sys
import time

COMFY = os.path.expanduser("~/ComfyUI")
sys.path.insert(0, COMFY)
sys.path.insert(0, os.path.join(COMFY, "custom_nodes", "minimax-h3-chunk-star7"))
sys.path.insert(0, os.path.join(COMFY, "custom_nodes", "comfyui-h3-tp2"))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

CKPT = "minimax_h3_ref2va_pruned_w4a8_mixed.safetensors"
S = 17260
H = 5376
KC2 = 256.0


def bench(fn, r=5, w=2):
    for _ in range(w):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(r):
        t0 = time.perf_counter(); fn(); torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1000)
    return statistics.median(ts)


def main():
    import folder_paths
    import adaptive_loader as al
    from tp2 import _shard_n, _shard_k, _lin

    path = folder_paths.get_full_path_or_raise("diffusion_models", CKPT)
    al._neutralize_process_wide_h3_conflicts()
    p = al._load_h3_native_fp16(path)
    p.patch_model(device_to=torch.device("cuda:0"))
    mlp = p.model.diffusion_model.blocks[0].mlp

    D0, D1 = "cuda:0", "cuda:1"
    torch.cuda.set_device(0)
    torch.cuda.init()

    ffn = mlp.fc1.weight.shape[0] // 2
    g0 = list(range(0, ffn // 2)) + list(range(2 * (ffn // 2), 3 * (ffn // 2)))
    g1 = list(range(ffn // 2, ffn)) + list(range(3 * (ffn // 2), 4 * (ffn // 2)))
    fc1_0, fc1_1 = _shard_n(mlp.fc1.weight, g0, g1, D0, D1)
    fc2_0, fc2_1 = _shard_k(mlp.fc2.weight, ffn // 2, D0, D1)

    h0 = torch.randn(S, H, dtype=torch.float16, device=D0)
    h1 = h0.to(D1)

    def fc1(dev_i, h, w):
        with torch.cuda.device(dev_i):
            return F.linear(h, w)

    def mlp(dev_i, h, w1, w2):
        with torch.cuda.device(dev_i):
            g = _lin(h, w1, None)
            gate, up = g.chunk(2, dim=-1)
            act = F.silu(gate.to(torch.float32)) * up.to(torch.float32)
            return _lin((act / KC2).to(torch.float16), w2, None)

    print("=== fc1 半量 GEMM ===")
    t_f0 = bench(lambda: fc1(0, h0, fc1_0))
    t_f1 = bench(lambda: fc1(1, h1, fc1_1))
    def fc1_both():
        fc1(0, h0, fc1_0)
        fc1(1, h1, fc1_1)
    t_fb = bench(fc1_both)
    print(f"  dev0 单独          {t_f0:7.1f} ms")
    print(f"  dev1 单独          {t_f1:7.1f} ms")
    print(f"  两卡背靠背并发     {t_fb:7.1f} ms   理想 max={max(t_f0,t_f1):.1f}  串行和={t_f0+t_f1:.1f}")

    print("\n=== 完整 MLP 链 (fc1 + swiglu + fc2) ===")
    t_m0 = bench(lambda: mlp(0, h0, fc1_0, fc2_0))
    t_m1 = bench(lambda: mlp(1, h1, fc1_1, fc2_1))
    def mlp_both():
        mlp(0, h0, fc1_0, fc2_0)
        mlp(1, h1, fc1_1, fc2_1)
    t_mb = bench(mlp_both)
    print(f"  dev0 单独          {t_m0:7.1f} ms")
    print(f"  dev1 单独          {t_m1:7.1f} ms")
    print(f"  两卡背靠背并发     {t_mb:7.1f} ms   理想 max={max(t_m0,t_m1):.1f}  串行和={t_m0+t_m1:.1f}")

    def overlap(single_max, both):
        ideal_gain = single_max
        actual_gain = (t_m0 + t_m1 - both)
        return actual_gain / ideal_gain * 100 if ideal_gain else 0.0

    print(f"\n  重叠效率 = 实际省下 / 理想可省 = {overlap(max(t_m0,t_m1), t_mb):.0f}%  "
          f"(0% = 完全串行, 100% = 完全重叠)")
    print(f"  与节点内实测 328ms 对比: {t_mb:.1f} ms")


if __name__ == "__main__":
    main()
