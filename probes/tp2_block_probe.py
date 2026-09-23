#!/usr/bin/env python3
"""TP2 full block: shard every linear layer, allreduce the row-parallel ones, and
compare against the real FP16-Exact block forward.

Layout facts that matter (all verified from the model source):
  * qkv_proj [21504,5376] -> 28+28 heads (column-parallel, concat)
  * out_proj [5376,7168]  -> row-parallel, the two halves SUM
  * fc1 [28672,5376] = [gate(14336) | up(14336)] interleaved -> each TP rank takes
    gate[0:7168]+up[0:7168]  (i.e. rows [0:7168] ++ [14336:21504])
  * fc2 [5376,14336]      -> row-parallel over the FFN dim, halves SUM
  * FP16-Exact: out_proj scales its input by /64 then *64; MLP scales by /256 then *256
    (K_OUT_PROJ=64, K_FC2=256). The residual stream is fp32 throughout.

    .venv/bin/python tp2_block_probe.py
"""
import dataclasses
import os
import statistics
import sys

COMFY = os.path.expanduser("~/ComfyUI")
sys.path.insert(0, COMFY)
sys.path.insert(0, os.path.join(COMFY, "custom_nodes", "minimax-h3-chunk-star7"))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

CKPT = "minimax_h3_ref2va_pruned_w4a8_mixed.safetensors"
LAYOUT = "AsymW4A8Int8Layout"
KO, KC2 = 64.0, 256.0
S = 17260          # the film's real packed sequence


def _qt(qdata, params):
    from comfy_kitchen.tensor.base import QuantizedTensor
    return QuantizedTensor(qdata, LAYOUT, params)


def _fp(t):
    """fp8 scales cannot be indexed on CPU; fp8->fp32 is lossless."""
    return t.float() if isinstance(t, torch.Tensor) and t.dtype == torch.float8_e4m3fn else t


def shard_n(w, rows0, rows1, dev0, dev1):
    """Column-parallel: split output rows (possibly interleaved for fc1)."""
    from comfy_kitchen.tensor.w4a8_int8 import AsymW4A8Int8Layout as L
    qdata, scale, s_channel, correction, codebook = L.get_plain_tensors(w)
    p = w._params
    scale, s_channel = _fp(scale), _fp(s_channel)
    correction = _fp(correction)
    outs = []
    for rows, dev in ((rows0, dev0), (rows1, dev1)):
        rows = list(rows)
        pr = dataclasses.replace(
            p, scale=scale[rows].to(dev), s_channel=s_channel[rows].to(dev),
            correction=correction[:, rows].to(dev) if correction is not None else None,
            orig_shape=(len(rows), w.shape[1]))
        outs.append(_qt(qdata[rows].to(dev), pr))
    return outs


def shard_k(w, mid, dev0, dev1):
    """Row-parallel: split input columns."""
    from comfy_kitchen.tensor.w4a8_int8 import AsymW4A8Int8Layout as L
    qdata, scale, s_channel, correction, codebook = L.get_plain_tensors(w)
    p = w._params
    scale = _fp(scale)
    g = p.group_size
    sc = mid // g
    p0 = dataclasses.replace(p, scale=scale[:, :sc].to(dev0), orig_shape=(w.shape[0], mid))
    p1 = dataclasses.replace(p, scale=scale[:, sc:].to(dev1), orig_shape=(w.shape[0], w.shape[1] - mid))
    return (_qt(qdata[:, :mid // 2].to(dev0), p0),
            _qt(qdata[:, mid // 2:].to(dev1), p1))


def _rms(t, w):
    return t * torch.rsqrt(t.float().pow(2).mean(-1, keepdim=True) + 1e-5).to(t.dtype) * w


def _qkv_rope(att, qkv_w, qn, kn, h, rf, dev, heads):
    import comfy.model_management as mm
    import comfy.quant_ops as qo
    qkv = F.linear(h.to(dev), qkv_w.to(dev))
    q, k, v = qkv.split(heads * att.head_dim, dim=-1)
    s = h.shape[0]
    q = q.view(1, s, heads, att.head_dim)
    k = k.view(1, s, heads, att.head_dim)
    qw = mm.cast_to(qn, device=dev)
    kw = mm.cast_to(kn, device=dev)
    q, k = qo.ck.rms_rope_split_half_(q, k, rf.to(dev), qw, kw,
                                      epsilon=att.q_norm.eps, rot_dim=rf.shape[-3] * 2)
    return q[0], k[0], v.view(s, heads, att.head_dim)


def _attn(q, k, v):
    from comfy.ldm.modules.attention import AttentionTensorContainer, optimized_attention
    heads = q.shape[1]
    o = optimized_attention(AttentionTensorContainer(q.transpose(0, 1).unsqueeze(0)),
                            AttentionTensorContainer(k.transpose(0, 1).unsqueeze(0)),
                            AttentionTensorContainer(v.transpose(0, 1).unsqueeze(0)),
                            heads, mask=None, skip_reshape=True, transformer_options={})
    return o.squeeze(0)


def main():
    import folder_paths
    import adaptive_loader as al
    from comfy.ldm.minimax.model import _mod_gate, _mod_scale_shift, rope_rotation_table

    path = folder_paths.get_full_path_or_raise("diffusion_models", CKPT)
    al._neutralize_process_wide_h3_conflicts()
    patcher = al._load_h3_native_fp16(path)
    D0, D1 = "cuda:0", "cuda:1"
    torch.cuda.set_device(0)
    # apply the FP16-Exact object patches so the reference block is the real one
    # (fp32 residual + K_OUT_PROJ/K_FC2 scaling), not the unpatched base block.
    patcher.patch_model(device_to=torch.device(D0))
    dm = patcher.model.diffusion_model
    blk = dm.blocks[0]
    att, mlp = blk.attn, blk.mlp
    H, heads, hd = dm.hidden_size, att.heads, att.head_dim
    h2 = heads // 2
    ffn_half = mlp.fc1.weight.shape[0] // 4          # 28672/4 = 7168

    # --- shard weights ---
    # qkv output is [q(7168) | k(7168) | v(7168)], each 56*128. Head-parallel TP takes
    # heads 0..27 (resp 28..55) from EACH of q/k/v -> three interleaved slices.
    full = heads * hd
    qh = h2 * hd
    rows0 = list(range(0, qh)) + list(range(full, full + qh)) + list(range(2 * full, 2 * full + qh))
    rows1 = list(range(qh, full)) + list(range(full + qh, 2 * full)) + list(range(2 * full + qh, 3 * full))
    qkv0, qkv1 = shard_n(att.qkv_proj.weight, rows0, rows1, D0, D1)
    op0, op1 = shard_k(att.out_proj.weight, h2 * hd, D0, D1)            # 3584
    # fc1 interleaved: gate[0:ffn_half] ++ up[0:ffn_half]
    rows0 = list(range(0, ffn_half)) + list(range(2 * ffn_half, 3 * ffn_half))
    rows1 = list(range(ffn_half, 2 * ffn_half)) + list(range(3 * ffn_half, 4 * ffn_half))
    fc10, fc11 = shard_n(mlp.fc1.weight, rows0, rows1, D0, D1)
    fc20, fc21 = shard_k(mlp.fc2.weight, ffn_half, D0, D1)              # 7168

    # replicate tiny params
    n1 = [blk.norm1.weight.detach().to(d) for d in (D0, D1)]
    n2 = [blk.norm2.weight.detach().to(d) for d in (D0, D1)]
    qn = [att.q_norm.weight.detach().to(d) for d in (D0, D1)]
    kn = [att.k_norm.weight.detach().to(d) for d in (D0, D1)]
    adaln = blk.adaln_proj          # already on D0 via patch_model

    torch.manual_seed(0)
    x = (torch.randn(S, H, dtype=torch.float32) * 0.5).half().to(D0)
    t_vals = torch.tensor([0.05, 0.5], dtype=torch.float32, device=D0)
    table = dm.adaln_t_table.to(D0)
    pos = t_vals.clamp(0.0, 1.0) * (table.shape[0] - 1)
    i0 = pos.floor().long().clamp(max=table.shape[0] - 2)
    t_emb = torch.lerp(table[i0], table[i0 + 1], (pos - i0).unsqueeze(1))
    pos_ids = torch.zeros(S, 3, dtype=torch.float64)
    rf = rope_rotation_table(dm.rope_freqs(pos_ids, D0), torch.float16)
    third = S // 3
    segs = [(0, third, 0), (third, S, 1)]

    with torch.cuda.device(0):
        y_ref = blk(x.clone(), t_emb, segs, rf, transformer_options={})
    torch.cuda.synchronize()
    print(f"reference blk.forward : {tuple(y_ref.shape)} {y_ref.dtype} "
          f"finite={bool(torch.isfinite(y_ref).all())} absmax={y_ref.float().abs().max().item():.4f}")

    def tp2(xin, t_emb, segs, rf):
        st0, st1 = torch.cuda.current_stream(0), torch.cuda.current_stream(1)
        m0 = adaln(t_emb)                 # adaln lives on D0; mods are tiny
        m1 = tuple(v.to(D1) for v in m0)
        with torch.cuda.device(0):
            x0 = xin.to(torch.float32)
            h = _mod_scale_shift(_rms(x0, n1[0]), m0[0], m0[1], segs).to(torch.float16)
            q0, k0, v0 = _qkv_rope(att, qkv0, qn[0], kn[0], h, rf, D0, h2)
        with torch.cuda.device(1):
            x1 = xin.to(D1, non_blocking=True).to(torch.float32)
            h = _mod_scale_shift(_rms(x1, n1[1]), m1[0], m1[1], segs).to(torch.float16)
            q1, k1, v1 = _qkv_rope(att, qkv1, qn[1], kn[1], h, rf, D1, h2)
        with torch.cuda.device(0):
            O0 = _attn(q0, k0, v0)                       # [S, h2*hd]
            p0 = F.linear((O0 / KO).to(torch.float16), op0)
        with torch.cuda.device(1):
            O1 = _attn(q1, k1, v1)
            p1 = F.linear((O1 / KO).to(torch.float16), op1)
        st0.wait_stream(st1); st1.wait_stream(st0)
        with torch.cuda.device(0):
            ao = (p0 + p1.to(D0)).to(torch.float32) * KO
            x0 = _mod_gate(x0, m0[2], ao, segs)
            h = _mod_scale_shift(_rms(x0, n2[0]), m0[3], m0[4], segs).to(torch.float16)
            g = F.linear(h, fc10)                        # [S, 2*ffn_half]
            gate, up = g.chunk(2, dim=-1)
            act = (torch.nn.functional.silu(gate.to(torch.float32)) * up.to(torch.float32))
            mm0 = F.linear((act / KC2).to(torch.float16), fc20)
        with torch.cuda.device(1):
            x1 = _mod_gate(x1, m1[2], ao.to(D1), segs)   # broadcast the same attention residual
            h = _mod_scale_shift(_rms(x1, n2[1]), m1[3], m1[4], segs).to(torch.float16)
            g = F.linear(h.to(D1), fc11)
            gate, up = g.chunk(2, dim=-1)
            act = (torch.nn.functional.silu(gate.to(torch.float32)) * up.to(torch.float32))
            mm1 = F.linear((act / KC2).to(torch.float16), fc21)
        st0.wait_stream(st1); st1.wait_stream(st0)
        with torch.cuda.device(0):
            mo = (mm0 + mm1.to(D0)).to(torch.float32) * KC2
            return _mod_gate(x0, m0[5], mo, segs)

    y = tp2(x, t_emb, segs, rf)
    torch.cuda.synchronize()
    d = (y.float() - y_ref.float()).abs()
    print(f"tp2_block              : {tuple(y.shape)} {y.dtype} finite={bool(torch.isfinite(y).all())}")
    print(f"  max|diff|={d.max().item():.4e}  mean|diff|={d.mean().item():.4e}  "
          f"rel_max={d.max().item() / y_ref.float().abs().max().item():.4f}")

    def timed(fn, r=5, w=2):
        for _ in range(w):
            fn()
        torch.cuda.synchronize()
        ts = []
        for _ in range(r):
            a, b = torch.cuda.Event(True), torch.cuda.Event(True)
            a.record(); fn(); b.record(); torch.cuda.synchronize()
            ts.append(a.elapsed_time(b))
        return statistics.median(ts)

    with torch.cuda.device(0):
        t1 = timed(lambda: blk(x.clone(), t_emb, segs, rf, transformer_options={}))
    t2 = timed(lambda: tp2(x, t_emb, segs, rf))
    print(f"[timing] real block (1 GPU) {t1:8.2f} ms | tp2 (2 GPUs) {t2:8.2f} ms | speedup {t1 / t2:.2f}x  (S={S})")


if __name__ == "__main__":
    main()
