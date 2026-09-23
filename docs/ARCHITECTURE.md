# Architecture: MiniMax H3 TP2 on dual RTX 2080 Ti

## The model

MiniMax H3 (ref2va, pruned, W4A8) is a joint video+audio flow transformer:

| | value |
|---|---|
| DiT logical params | 20.11 B (11.56 B MLP + 7.71 B attention + refiner/others) |
| blocks / hidden | 50 / 5376 |
| attention | 56 heads × 128 |
| FFN | 14336 |
| quant | W4A8 — int4 weights (2/byte), fp8 group scales (group 16), fp32 channel scale, Lloyd-Max codebook `[16]`, ConvRot 256 |
| disk | 10.96 GiB (qkv layer: 55 MiB packed vs 220 MiB fp16) |
| VAE | 16× spatial compression; patch `(1,2,2)` |

## Why TP2 is needed, and its ceiling

The 2080 Ti (sm_75) has **no BF16/FP8 tensor cores**, so W4A8 is mandatory to fit
the 20 B model. Two 22 GB cards split every block's four linears:

```
qkv_proj [21504,5376]  -> each rank: q/k/v heads 0..27   (column-parallel, 3 interleaved slices)
out_proj [5376,7168]   -> row-parallel, halves SUM       (allreduce)
fc1      [28672,5376]  -> gate[0:7168] ++ up[0:7168]     (column-parallel)
fc2      [5376,14336]  -> row-parallel, halves SUM       (allreduce)
```

Residual stays **fp32** (FP16-Exact): out_proj scales input `/64` then `*64`, MLP
`/256` then `*256`. The symmetric allreduce gives each rank the other's partial and
sums locally, so no separate broadcast copy.

TP2 halves only the **weight** side: both cards still read the full activation
sequence `[S, 5376]`, plus a duplicated fp32 residual and allreduce syncs. So the
speedup is bounded by `1/(1 - X/2)` with `X ≈ 0.5 → ~1.33×` — which is exactly the
1.31× measured before the fix below.

## The real bottleneck: W4A8 GEMM blocks its host thread

The first TP2 implementation reached 34.0 s/step (1.31×) but never saturated the
GPUs (mean util 64% / 33%). Per-phase profiling blamed the MLP, but the offline
microbenchmarks all scaled "perfectly". That turned out to be a measurement bug:
`torch.cuda.synchronize()` with no argument syncs **only the current device**, so
concurrent runs were systematically under-measured.

With both devices synced, the truth was:

```
dense fp16 GEMM (same shape)     dev0 61.9 | dev1 58.0 | concurrent  61.0 | overlap 95%  ✓
W4A8 GEMM (comfy-kitchen)        dev0 103.3| dev1 96.8 | concurrent 194.7 | overlap  5%  ✗
W4A8 GEMM from 2 host threads    dev0 106.5| dev1 97.1 | concurrent 106.8 | overlap 91%  ✓
```

The W4A8 GEMM **blocks the calling host thread until its kernel retires**, so a
single host thread can never keep two GPUs' GEMMs in flight at once. One thread per
rank fixes it (`_par` in `tp2.py`), because each thread carries its own current
device and stream.

This one change took TP2 from **34.0 s/step (1.31×)** to **21.0 s/step (2.12×)** at
864×480, with mean GPU util 83% / 44%.

Two follow-on subtleties, both handled in `tp2.py`:

1. `torch.inference_mode()` is **thread-local**; a worker thread runs outside it,
   so in-place ops on the caller's inference tensors raise. `_run` re-enters the
   caller's mode.
2. `torch.cuda.current_stream` is thread-local too; each worker sets its own
   device + stream explicitly (`with torch.cuda.device(i), torch.cuda.stream(st)`).

## Native resolution envelope

Native canvas is **1344×768** (the R2V node default). The packed sequence is
`S ≈ 3.3e-4 × W × H × frames`:

| 1344×768 | S | sampling | end-to-end |
|---|---|---|---|
| 73 frames (3.04 s) | 24 737 | 32.5 s/step | ✅ |
| 90 frames (3.75 s) | ~30 000 | 40.2 s/step | ✅ (desktop card at 94% VRAM) |
| 107 frames | ~35 700 | 34–35 s/step | ❌ decode stage times out |

## Diagnostic probes

`probes/` contains the exact experiments behind the numbers above. The decisive one
is `tp2_w4a8_concurrency_probe.py` (fp16 vs W4A8, single vs concurrent vs two
threads).
