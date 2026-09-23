#!/usr/bin/env python3
"""Why does the CUTLASS-backed W4A8 GEMM serialize across the two GPUs?

Compares, with both devices synced correctly:
  A) dense fp16 GEMM of the same shape          (baseline: should overlap)
  B) the real W4A8 shard                        (suspect)
  C) the real W4A8 shard from two host threads  (host-serialization test)

    .venv/bin/python tp2_w4a8_concurrency_probe.py
"""
import os
import statistics
import sys
import threading
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


def sync_all():
    for i in range(torch.cuda.device_count()):
        torch.cuda.synchronize(i)


def bench(fn, r=6, w=3):
    for _ in range(w):
        fn()
    sync_all()
    ts = []
    for _ in range(r):
        t0 = time.perf_counter(); fn(); sync_all()
        ts.append((time.perf_counter() - t0) * 1000)
    return statistics.median(ts)


def report(name, f0, f1):
    a, b = bench(f0), bench(f1)
    c = bench(lambda: (f0(), f1()))
    ideal = max(a, b)
    ov = (a + b - c) / ideal * 100 if ideal else 0.0
    print(f"  {name:28s} dev0 {a:7.1f} | dev1 {b:7.1f} | 并发 {c:7.1f} | "
          f"理想 {ideal:6.1f} | 串行 {a+b:6.1f} | 重叠 {ov:5.1f}%")


def main():
    import folder_paths
    import adaptive_loader as al
    from tp2 import _shard_n

    path = folder_paths.get_full_path_or_raise("diffusion_models", CKPT)
    al._neutralize_process_wide_h3_conflicts()
    p = al._load_h3_native_fp16(path)
    p.patch_model(device_to=torch.device("cuda:0"))
    fc1 = p.model.diffusion_model.blocks[0].mlp.fc1.weight

    ffn = fc1.shape[0] // 2
    q = ffn // 2
    g0 = list(range(0, q)) + list(range(2 * q, 3 * q))
    g1 = list(range(q, 2 * q)) + list(range(3 * q, 4 * q))
    w0, w1 = _shard_n(fc1, g0, g1, "cuda:0", "cuda:1")
    print(f"  fc1 原权重 {tuple(fc1.shape)}  transposed={fc1._params.transposed}  "
          f"orig_dtype={fc1._params.orig_dtype}  correction={fc1._params.correction is None and 'None' or 'present'}")

    d0 = w0.dequantize()
    d1 = d0.to("cuda:1")
    print(f"  反量化后半量权重大小 {tuple(d0.shape)} {d0.dtype}")

    x0 = torch.randn(S, H, dtype=torch.float16, device="cuda:0")
    x1 = x0.to("cuda:1")

    print("\n=== A. dense fp16 GEMM（同形状、只谈重叠能力）===")
    report("fp16 F.linear",
           lambda: F.linear(x0, d0),
           lambda: F.linear(x1, d1))

    print("\n=== B. 真 W4A8 shard（当前节点的路径）===")
    report("W4A8 F.linear",
           lambda: F.linear(x0, w0),
           lambda: F.linear(x1, w1))

    print("\n=== C. 真 W4A8，两个 host 线程分别发射 ===")
    ta = threading.Thread(target=lambda: F.linear(x0, w0))
    tb = threading.Thread(target=lambda: F.linear(x1, w1))
    for _ in range(2):
        ta = threading.Thread(target=lambda: F.linear(x0, w0)); ta.start(); ta.join()
        tb = threading.Thread(target=lambda: F.linear(x1, w1)); tb.start(); tb.join()
    sync_all()
    ts = []
    for _ in range(6):
        ta = threading.Thread(target=lambda: F.linear(x0, w0))
        tb = threading.Thread(target=lambda: F.linear(x1, w1))
        t0 = time.perf_counter(); ta.start(); tb.start(); ta.join(); tb.join(); sync_all()
        ts.append((time.perf_counter() - t0) * 1000)
    c = statistics.median(ts)
    a = bench(lambda: F.linear(x0, w0))
    b = bench(lambda: F.linear(x1, w1))
    print(f"  {'W4A8 双线程':28s} dev0 {a:7.1f} | dev1 {b:7.1f} | 并发 {c:7.1f} | "
          f"理想 {max(a,b):6.1f} | 串行 {a+b:6.1f} | 重叠 {(a+b-c)/max(a,b)*100:5.1f}%")

    print("\n=== D. W4A8 但输入很小（看是否与 M 有关）===")
    xs0 = torch.randn(2048, H, dtype=torch.float16, device="cuda:0")
    xs1 = xs0.to("cuda:1")
    report("W4A8 M=2048",
           lambda: F.linear(xs0, w0),
           lambda: F.linear(xs1, w1))


if __name__ == "__main__":
    main()
