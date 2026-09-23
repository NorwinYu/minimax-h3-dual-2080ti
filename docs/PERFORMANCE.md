# Performance

All numbers measured on this exact machine, real renders. No estimated or
synthetic figures.

## Hardware

- 2× NVIDIA RTX 2080 Ti, 22 GB (modified), **sm_75 (Turing)**
- NVLink active (NV2, 2 links × 25.781 GB/s)
- Driver 580.173.02, CUDA 13.0, PyTorch 2.9.1+cu130
- Two cards on **different NUMA nodes**
- Power limit 250 W during the runs below

## Smoke sample (640×352 × 124 frames × 4 steps, fl2va + 4-step Turbo LoRA)

Single workflow `examples/smoke_t2v.json`, text-to-video, no reference image.

| metric | single card | TP2 |
|---|---|---|
| per-step | ~51 s | ~14 s |
| end-to-end (4 steps) | 243 s | 82 s |
| speedup | 1.00× | **2.97×** |
| GPU util | GPU0 idle / GPU1 97% | GPU0 57% / GPU1 72% |
| peak VRAM | GPU1 19071 MiB | GPU0 11579 / GPU1 19681 MiB |
| peak temp | GPU1 78 °C (97% for 4 min) | GPU0 67 °C / GPU1 75 °C (1.4 min) |

## Film (864×480 × 124 frames, ref2va)

| | s/step | speedup |
|---|---|---|
| single card, chunked | 44.6 | 1.00× |
| TP2 v1 | 34.0 | 1.31× |
| **TP2 + one-thread-per-rank** | **19.8** | **2.25×** |

Per-block phase times after the fix (50-block loop):

```
qkv+attn 211 ms | mlp 196 ms | final allreduce 13 ms  => 420 ms/block
```

## Native 1344×768

| frames | s/step | per-shot (4 steps) |
|---|---|---|
| 73 | 32.5 | ~195 s |
| 90 | 40.2 | ~220 s |

## Power 180 W vs 250 W

Same I2V benchmark (864×480 × 124 frames × 2 steps):

| | 180 W | 250 W |
|---|---|---|
| end-to-end | 98.3 s | 92.4 s (−6%) |
| mean SM clock | 1494 MHz | 1658 MHz (+11%) |
| mean power | 150 W | 195 W |

The cards are mildly power-limited at 180 W, but the workload is memory/launch
bound, so the higher clock only buys ~6% end-to-end.

## Quality gate

The thread fix is a pure scheduling change — output is **bit-exact** identical
(frame MAE 0.0000) before/after. Temporal-stability metrics and caption QA are the
regression gates; see `docs/FILM_PIPELINE.md`.
