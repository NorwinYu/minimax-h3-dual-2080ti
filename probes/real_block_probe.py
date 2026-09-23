#!/usr/bin/env python3
"""Ulysses2 on the REAL H3 weights: one real DiTBlock, split across two GPUs.

The synthetic prototype proved the mechanism with fp16 tensors. This one drives
the actual checkpoints through the actual block code -- real AsymW4A8Int8Layout
weights, the real fused rms+rope, the real attention backend -- so the speedup
number is measured on the thing the sampler will run.

    .venv/bin/python real_block_probe.py
"""
import os
import statistics
import sys
import time as _time

COMFY = os.path.expanduser("~/ComfyUI")
sys.path.insert(0, COMFY)
sys.path.insert(0, os.path.join(COMFY, "custom_nodes", "minimax-h3-chunk-star7"))

import torch  # noqa: E402

CKPT = "minimax_h3_ref2va_pruned_w4a8_mixed.safetensors"
S = 17260           # the film's real packed sequence


def load(dm_path):
    import adaptive_loader as al
    al._neutralize_process_wide_h3_conflicts()
    return al._load_h3_native_fp16(dm_path).model.diffusion_model


def place(blk, dev):
    """Move one block's weights to a device (what dynamic loading does per block)."""
    for name in ("norm1", "norm2", "attn", "mlp", "adaln_proj"):
        sub = getattr(blk, name)
        sub.to(dev)
        for p in sub.parameters(recurse=True):
            if getattr(p, "_layout_cls", None) is not None:
                p.data = p.data.to(dev)
    return blk


def mod_segments_for(S):
    return [(0, S, 0)]


def rope(model, S, dev):
    from comfy.ldm.minimax.model import rope_rotation_table
    pos = torch.zeros(S, 3, dtype=torch.float64)
    return rope_rotation_table(model.rope_freqs(pos, dev), torch.float16)



PH = {}
NCALL = [0]


def _sync_all():
    torch.cuda.synchronize(0)
    torch.cuda.synchronize(1)


def _snap(name, t0):
    if os.environ.get("NOPHASE"):
        return
    _sync_all()
    PH[name] = PH.get(name, 0.0) + (_time.perf_counter() - t0) * 1000.0

def block_direct(blk, x, rf):
    """The block's attn+MLP path without the per-token adaln modulation.

    Adaln is local per token (as are norm/rope), so it does not participate in
    the cross-GPU split; the pruned curve form of `adaln_proj` also needs the
    model's own forward to build its 8-dim input, which is out of scope here.
    Everything that the split changes -- attention and the MLP -- is included.
    """
    q, k, v = qkv_rope(blk, blk.norm1(x), rf)
    y = x + attn_out(blk, q, k, v)
    return y + blk.mlp(blk.norm2(y))


def qkv_rope(blk, x, rf):
    """norm1 + qkv + the model's own fused rms/rope, returning q,k,v [S,heads,hd]."""
    from comfy.ldm.minimax.model import Attention  # noqa: F401
    att = blk.attn
    h = att.qkv_proj(x)
    q, k, v = h.split(att.heads * att.head_dim, dim=-1)
    s = x.shape[0]
    q = q.view(1, s, att.heads, att.head_dim)
    k = k.view(1, s, att.heads, att.head_dim)
    import comfy.model_management as mm
    import comfy.quant_ops as qo
    qw = mm.cast_to(att.q_norm.weight, device=x.device)
    kw = mm.cast_to(att.k_norm.weight, device=x.device)
    rot = rf.shape[-3] * 2
    q, k = qo.ck.rms_rope_split_half_(q, k, rf, qw, kw, epsilon=att.q_norm.eps, rot_dim=rot)
    return q[0], k[0], v.view(s, att.heads, att.head_dim)


def attn_core(q, k, v):
    """Raw attention, NO out_proj.

    out_proj mixes all heads (inner = heads*head_dim), so it cannot run on a
    device that only holds half the heads -- the Ulysses gather has to happen
    first. Keeping it out of here is what makes the split correct.
    """
    from comfy.ldm.modules.attention import AttentionTensorContainer, optimized_attention
    # heads must be the head count actually present: the split path carries half
    # the heads per device, so q.shape[1] is 28 there and 56 in the reference.
    heads = q.shape[1]
    qq = AttentionTensorContainer(q.transpose(0, 1).unsqueeze(0))
    kk = AttentionTensorContainer(k.transpose(0, 1).unsqueeze(0))
    vv = AttentionTensorContainer(v.transpose(0, 1).unsqueeze(0))
    o = optimized_attention(qq, kk, vv, heads, mask=None, skip_reshape=True,
                            transformer_options={})
    return o.squeeze(0)


def block_direct(blk, x, rf):
    with torch.cuda.device(x.device):
        q, k, v = qkv_rope(blk, blk.norm1(x), rf)
        y = x + blk.attn.out_proj(attn_core(q, k, v))
        return y + blk.mlp(blk.norm2(y))


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


def main():
    import folder_paths
    path = folder_paths.get_full_path_or_raise("diffusion_models", CKPT)
    print(f"loading {CKPT}")
    da = load(path)
    db = load(path)
    print(f"loaded two independent models: {da is not db}")

    bA = place(da.blocks[0], "cuda:0")
    bB = place(db.blocks[0], "cuda:1")
    print(f"block0 A on cuda:0, B on cuda:1  "
          f"(qkv dev {bA.attn.qkv_proj.weight.device} / {bB.attn.qkv_proj.weight.device})")

    torch.manual_seed(0)
    x = (torch.randn(S, da.hidden_size, dtype=torch.float32) * 0.1).half()
    t_emb = torch.zeros(1, 2688, dtype=torch.float32)
    segs = mod_segments_for(S)
    rf_a = rope(da, S, "cuda:0")
    rf_b = rope(db, S, "cuda:1")

    # ---- reference: the real block, whole sequence, one GPU --------------
    xd0 = x.to("cuda:0")
    try:
        y_ref = block_direct(bA, xd0, rf_a)
        torch.cuda.synchronize()
        print(f"reference block : shape={tuple(y_ref.shape)} "
              f"finite={bool(torch.isfinite(y_ref).all())} "
              f"absmax={y_ref.float().abs().max().item():.4f}")
    except Exception as e:                                    # noqa: BLE001
        print(f"reference block : FAILED {type(e).__name__}: {e}")
        return

    # ---- Ulysses2: split the sequence, head-split attention across GPUs ---
    s0 = S // 2
    h2 = bA.attn.heads // 2

    s1 = S - s0
    hd = bA.attn.head_dim

    def ulysses_block():
        # Each device's ops need CUDA's current device set to that device: the
        # custom rms/rope op goes through DLPack and refuses a mismatched index.
        # Cross-device ordering uses stream waits, not cudaSynchronize: a global
        # sync drains both pipelines and destroys the overlap the split exists for.
        st0 = torch.cuda.current_stream(0)
        st1 = torch.cuda.current_stream(1)
        NCALL[0] += 1
        _t = _time.perf_counter()
        with torch.cuda.device(0):
            x0 = x[:s0].to("cuda:0")
            rf0 = rf_a[:, :s0].contiguous()
            q0, k0, v0 = qkv_rope(bA, bA.norm1(x0), rf0)
        with torch.cuda.device(1):
            x1 = x[s0:].to("cuda:1")
            rf1 = rf_b[:, s0:].contiguous()
            q1, k1, v1 = qkv_rope(bB, bB.norm1(x1), rf1)
        # ---- Ulysses scatter: exchange the off-half heads over NVLink ----
        _snap("qkv", _t); _t = _time.perf_counter()
        st0.wait_stream(st1)                       # dev0 will read q1/k1/v1
        st1.wait_stream(st0)                       # dev1 will read q0/k0/v0; both
        # buffers are filled before either side consumes, so the two directions
        # are in flight together instead of one after the other.
        with torch.cuda.device(0):
            rq0 = torch.empty(s1, h2, hd, dtype=q0.dtype, device="cuda:0")
            rk0 = torch.empty_like(rq0)
            rv0 = torch.empty_like(rq0)
            rq0.copy_(q1[:, :h2], non_blocking=True)
            rk0.copy_(k1[:, :h2], non_blocking=True)
            rv0.copy_(v1[:, :h2], non_blocking=True)
        with torch.cuda.device(1):
            rq1 = torch.empty(s0, h2, hd, dtype=q0.dtype, device="cuda:1")
            rk1 = torch.empty_like(rq1)
            rv1 = torch.empty_like(rq1)
            rq1.copy_(q0[:, h2:], non_blocking=True)
            rk1.copy_(k0[:, h2:], non_blocking=True)
            rv1.copy_(v0[:, h2:], non_blocking=True)
        _snap("scatter", _t); _t = _time.perf_counter()
        with torch.cuda.device(0):
            Q0 = torch.cat([q0[:, :h2], rq0]); K0 = torch.cat([k0[:, :h2], rk0]); V0 = torch.cat([v0[:, :h2], rv0])
        with torch.cuda.device(1):
            Q1 = torch.cat([rq1, q1[:, h2:]]); K1 = torch.cat([rk1, k1[:, h2:]]); V1 = torch.cat([rv1, v1[:, h2:]])
        with torch.cuda.device(0):
            O0 = attn_core(Q0, K0, V0)             # [S, h2*hd]
        with torch.cuda.device(1):
            Q1 = torch.cat([rq1, q1[:, h2:]]); K1 = torch.cat([rk1, k1[:, h2:]]); V1 = torch.cat([rv1, v1[:, h2:]])
            O1 = attn_core(Q1, K1, V1)
        # ---- Ulysses gather: full heads back on each device's own tokens ----
        _snap("attention", _t); _t = _time.perf_counter()
        st0.wait_stream(st1)                       # dev0 will read O1
        st1.wait_stream(st0)                       # dev1 will read O0 -- MUST come
        # before dev0's out_proj/MLP, otherwise it serially waits for them and the
        # two GPUs' biggest phase (MLP) never overlaps.
        with torch.cuda.device(0):
            g0 = torch.empty(s0, h2 * hd, dtype=O0.dtype, device="cuda:0")
            g0.copy_(O1[:s0], non_blocking=True)
        with torch.cuda.device(1):
            g1 = torch.empty(s1, h2 * hd, dtype=O0.dtype, device="cuda:1")
            g1.copy_(O0[s0:], non_blocking=True)
        _snap("g.gather", _t); _t = _time.perf_counter()
        with torch.cuda.device(0):
            o0 = torch.cat([O0[:s0], g0], dim=-1)  # [s0, heads*hd]
            y0 = x0 + bA.attn.out_proj(o0)
        with torch.cuda.device(1):
            o1 = torch.cat([g1, O1[s0:]], dim=-1)  # [s1, heads*hd]
            y1 = x1 + bB.attn.out_proj(o1)
        _snap("g.outproj", _t); _t = _time.perf_counter()
        with torch.cuda.device(0):
            y0 = y0 + bA.mlp(bA.norm2(y0))
        with torch.cuda.device(1):
            y1 = y1 + bB.mlp(bB.norm2(y1))
        _snap("g.mlp", _t); _t = _time.perf_counter()
        st0.wait_stream(st1)                       # dev0 gathers y1
        with torch.cuda.device(0):
            out = torch.cat([y0, y1.to("cuda:0", non_blocking=True)], dim=0)
        _snap("g.finalcat", _t)
        return out

    try:
        y = ulysses_block()
        torch.cuda.synchronize()
        d = (y.float() - y_ref.float()).abs().to("cuda:0")
        print(f"ulysses2 block  : shape={tuple(y.shape)} "
              f"max|diff|={d.max().item():.3e}  "
              f"ref_absmax={y_ref.float().abs().max().item():.4f}")
    except Exception as e:                                    # noqa: BLE001
        print(f"ulysses2 block  : FAILED {type(e).__name__}: {e}")
        return

    t1 = timed(lambda: block_direct(bA, xd0, rf_a))
    t2 = timed(ulysses_block)
    print(f"[timing] real block single GPU  {t1:8.2f} ms")
    print(f"[timing] real block Ulysses2    {t2:8.2f} ms   speedup {t1 / t2:.2f}x   (S={S})")
    if PH:
        n = max(1, NCALL[0])
        print(f"[phases] per-call ms (over {n} calls):")
        for k, v in PH.items():
            print(f"    {k:12s} {v / n:8.2f} ms/call")
        print(f"    {'SUM':12s} {sum(PH.values()) / n:8.2f} ms/call   "
              f"(measured total {t2:.2f})")

    # ---- where does the missing parallelism go? --------------------------
    # Isolate pure compute concurrency from the all-to-all: run each half on
    # its own GPU with no exchange at all.
    half = S // 2
    xh0 = x[:half].to("cuda:0")
    xh1 = x[half:].to("cuda:1")
    rh0 = rf_a[:, :half].contiguous()
    rh1 = rf_b[:, half:].contiguous()

    def wall(fn, repeats=3, warmup=1):
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        ts = []
        for _ in range(repeats):
            a = _time.perf_counter(); fn(); torch.cuda.synchronize()
            ts.append((_time.perf_counter() - a) * 1000)
        return statistics.median(ts)

    a0 = wall(lambda: block_direct(bA, xh0, rh0))
    a1 = wall(lambda: block_direct(bB, xh1, rh1))

    def both():
        with torch.cuda.device(0):
            block_direct(bA, xh0, rh0)
        with torch.cuda.device(1):
            block_direct(bB, xh1, rh1)

    ab = wall(both)
    print(f"[diag] half S={half} on cuda:0 alone   {a0:8.2f} ms")
    print(f"[diag] half S={half} on cuda:1 alone   {a1:8.2f} ms")
    print(f"[diag] both halves concurrently        {ab:8.2f} ms  "
          f"(ideal {max(a0, a1):.2f}, sum {a0 + a1:.2f})")
    print(f"[diag] full S on one GPU               {t1:8.2f} ms")
    print(f"[diag] -> concurrency efficiency {max(a0, a1) / ab * 100:.0f}%  |  "
          f"split compute ceiling {t1 / ab:.2f}x")


if __name__ == "__main__":
    main()
