#!/usr/bin/env python3
"""Which stage of the TP2 MLP refuses to overlap across the two GPUs?

Breaks the block-0 MLP into cast / rms+mod / fc1 / swiglu / fc2 and measures each
stage alone on one GPU versus both GPUs launched back-to-back. Samples board
power during the concurrent full-MLP case to check for a power wall.

    .venv/bin/python tp2_mlp_stage_probe.py
"""
import os
import statistics
import subprocess
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
KC2 = 256.0


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
    a = bench(f0)
    b = bench(f1)
    c = bench(lambda: (f0(), f1()))
    ideal = max(a, b)
    ov = (a + b - c) / ideal * 100 if ideal else 0.0
    print(f"  {name:22s} dev0 {a:7.1f} | dev1 {b:7.1f} | 并发 {c:7.1f} | "
          f"理想 {ideal:6.1f} | 串行 {a+b:6.1f} | 重叠 {ov:5.1f}%")


def sample_power(stop, out):
    while not stop.is_set():
        try:
            r = subprocess.run(
                ["nvidia-smi", "--query-gpu=power.draw,utilization.gpu,clocks.sm",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=3)
            out.append(r.stdout.strip().replace("\n", " || "))
        except Exception:
            pass
        time.sleep(0.15)


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

    ffn = mlp.fc1.weight.shape[0] // 2
    q = ffn // 2
    g0 = list(range(0, q)) + list(range(2 * q, 3 * q))
    g1 = list(range(q, 2 * q)) + list(range(3 * q, 4 * q))
    fc1_0, fc1_1 = _shard_n(mlp.fc1.weight, g0, g1, D0, D1)
    fc2_0, fc2_1 = _shard_k(mlp.fc2.weight, q, D0, D1)

    h0 = torch.randn(S, H, dtype=torch.float16, device=D0)
    h1 = h0.to(D1)
    xf0, xf1 = h0.to(torch.float32), h1.to(torch.float32)
    w0 = torch.ones(H, dtype=torch.float32, device=D0)
    w1 = w0.to(D1)
    # swiglu input: the fc1 output, [S, ffn] fp16, chunked into gate/up
    g0t = torch.randn(S, ffn, dtype=torch.float16, device=D0)
    g1t = g0t.to(D1)

    def swiglu(g):
        gate, up = g.chunk(2, dim=-1)
        return F.silu(gate.to(torch.float32)) * up.to(torch.float32)

    def rms_mod(xf, w):
        r = torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + 1e-5)
        return (xf * r * w).mul_(1.0 + w).add_(w)

    print("\n=== MLP 各段：单卡 vs 双卡并发 ===")
    report("0. fp16->fp32 cast",
           lambda: h0.to(torch.float32),
           lambda: h1.to(torch.float32))
    report("1. rms+mod fp32",
           lambda: rms_mod(xf0, w0),
           lambda: rms_mod(xf1, w1))
    report("2. fc1 W4A8 (N-split)",
           lambda: _lin(h0, fc1_0, None),
           lambda: _lin(h1, fc1_1, None))
    report("3. swiglu fp32",
           lambda: swiglu(g0t),
           lambda: swiglu(g1t))
    a0 = g0t.new_empty(S, q)
    a1 = g1t.new_empty(S, q)
    report("4. fc2 W4A8 (K-split)",
           lambda: _lin(a0, fc2_0, None),
           lambda: _lin(a1, fc2_1, None))

    def mlp(i, h, w1_, w2_):
        with torch.cuda.device(i):
            g = _lin(h, w1_, None)
            act = swiglu(g)
            return _lin((act / KC2).to(torch.float16), w2_, None)

    report("5. 完整 MLP",
           lambda: mlp(0, h0, fc1_0, fc2_0),
           lambda: mlp(1, h1, fc1_1, fc2_1))

    stop = threading.Event()
    samples = []
    th = threading.Thread(target=sample_power, args=(stop, samples), daemon=True)
    th.start()
    c = bench(lambda: (mlp(0, h0, fc1_0, fc2_0), mlp(1, h1, fc1_1, fc2_1)), r=8)
    stop.set(); th.join(timeout=2)
    rows = []
    for row in samples:
        parts = row.split(" || ")
        if len(parts) == 2:
            rows.append([[float(v) for v in parts[k].split(",")] for k in range(2)])
    print(f"\n  并发完整 MLP {c:.1f} ms 期间功耗采样 ({len(rows)} 次):")
    for k in range(2):
        if rows:
            print(f"    GPU{k} 功耗 max {max(r[k][0] for r in rows):6.1f} W  "
                  f"利用率 max {max(r[k][1] for r in rows):3.0f}%  "
                  f"SM {max(r[k][2] for r in rows):4.0f} MHz")


if __name__ == "__main__":
    main()
