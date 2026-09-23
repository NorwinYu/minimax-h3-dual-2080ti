# MiniMax H3 · 双 RTX 2080 Ti

[![CI](https://github.com/NorwinYu/minimax-h3-dual-2080ti/actions/workflows/ci.yml/badge.svg)](https://github.com/NorwinYu/minimax-h3-dual-2080ti/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

在两张 **22 GB 改显存的 RTX 2080 Ti（sm_75，NVLink）**上本地运行 **MiniMax H3**（20B 视频+音频联合流式 Transformer）。W4A8 量化让它装得下，张量并行（TP2）让两张卡同时干活。

## 性能对比（同一样本，实测）

样本 `examples/smoke_t2v.json`：640×352 × 124 帧 × 4 步，fl2va + 4-step Turbo LoRA，**纯文生视频，不需要参考图**。

| 指标 | 单卡（chunked） | TP2（双卡） |
|---|---|---|
| 每步耗时 | ~51 s | **~14 s** |
| 端到端（4 步） | 243 s | **82 s** |
| 加速比 | 1× | **2.97×** |
| GPU 利用率 | GPU0 闲置 / GPU1 97% | GPU0 57% / GPU1 72% |
| 显存峰值 | GPU1 19071 MiB | GPU0 11579 / GPU1 19681 MiB |
| 温度峰值 | GPU1 **78 °C**（满载 4 分钟） | GPU0 67 °C / GPU1 75 °C（仅 1.4 分钟） |

> 单卡把全部计算压在一张卡上，GPU1 满载 4 分钟到 78 °C；TP2 把负载分摊到两张卡、快 3 倍，单卡高温时长缩短到 **1/3**。

更大分辨率（成片 864×480 × 124 帧，ref2va）：单卡 44.6 s/step → TP2 **19.8 s/step（2.25×）**。

## 参数

### 模型

| 项 | 值 |
|---|---|
| DiT 参数量 / 层数 / hidden | 20.11 B / 50 / 5376 |
| 注意力 | 56 头 × 128 |
| FFN | 14336 |
| qkv_proj / out_proj | [21504,5376] / [5376,7168] |
| fc1 / fc2 | [28672,5376] / [5376,14336] |
| 量化 | W4A8：int4（2/字节）· group 16 · ConvRot 256 · fp8 组 scale · fp32 通道 scale · Lloyd-Max 码本 `[16]` |
| DiT 磁盘占用 | 10.96 GiB（fl2va 11.68 GiB） |
| VAE | 16× 空间压缩，patch `(1,2,2)` |

### 运行时

| 项 | 值 |
|---|---|
| 分辨率 | 640×352（smoke）/ 864×480 / 1344×768（原生） |
| 每镜帧数 | 124（480p）/ 73（原生） |
| 步数 / 采样器 | 4 / euler |
| LoRA | `minimax_h3_fl2v_turbo_4step` / `minimax_h3_ref2v_turbo_4step_v0.1` |
| 注意力后端 | `comfy_kitchen_int8` |
| 种子 | 42，每镜 +1000 |
| TP2 副本 | `cuda:1`（物理桌面卡） |
| TP2 切分 | qkv/out/fc1 列并行 · fc2 行并行 · 对称 allreduce |
| 精度 | FP16-Exact：fp32 残差，out_proj `/64`→`*64`，MLP `/256`→`*256` |

### 启动

```bash
export CUDA_VISIBLE_DEVICES=1,0          # GPU1 在前 → cuda:0 是计算卡
python main.py --listen 127.0.0.1 --port 8188 \
    --reserve-vram 1.5 --vram-headroom 0.5 \
    --disable-pinned-memory --fast-disk
```

`CUDA_VISIBLE_DEVICES=1,0` 很关键：`--cuda-device 1` 只会暴露一张卡，把节点里的所有 `cuda:0` 悄悄坍缩到同一张卡。

## 为什么「一卡一线程」是核心

第一版 TP2 只做到 1.31×，GPU 一半在空转。根因不是切权重，而是 **comfy-kitchen 的 W4A8 GEMM 会阻塞调用它的 host 线程直到 kernel 跑完**——单线程永远无法让两张卡的 GEMM 同时在飞：

```
dense fp16 GEMM             dev0 61.9 | dev1 58.0 | 并发  61.0 | 重叠 95%  ✓
W4A8 GEMM                   dev0 103.3| dev1 96.8 | 并发 194.7 | 重叠  5%  ✗
W4A8 GEMM，一卡一线程       dev0 106.5| dev1 97.1 | 并发 106.8 | 重叠 91%  ✓
```

给每个 rank 一个独立 host 线程（`tp2.py` 里的 `_par`）就补上了这块。这个改动是纯调度优化——输出逐比特一致（帧 MAE 0.0000）。详见 `docs/ARCHITECTURE.md`。

## 硬件

- 2× RTX 2080 Ti，22 GB（改显存），sm_75（Turing）
- NVLink 桥（双 link）
- ≥ 32 GB 内存，驱动 580.x，CUDA 13.0，PyTorch 2.9.1+cu130

## 安装

1. **ComfyUI**，cu130 版 PyTorch 环境——`scripts/setup_env.sh` 记录了中国可用的镜像安装方式（torch 2.9.1+cu130）。
2. **自定义节点**（许可见 `THIRD_PARTY_NOTICES.md`）：`comfyui-multigpu`、`minimax-h3-chunk-star7`、`minimax-h3-fp16-exact-star7`、`comfyui-minimax-h3-turbo`、`minimax-h3-audio-t8`（可选），以及本仓库的 **`comfyui-h3-tp2`**（`custom_nodes/` 下）。
3. **模型**（放进 `ComfyUI/models/`，遵守 MiniMax 的模型许可，勿提交）：
   - `diffusion_models/minimax_h3_fl2va_pruned_w4a8_mixed.safetensors`（或 ref2va）
   - `loras/minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors`
   - `text_encoders/qwen3vl_4b_fp8_scaled.safetensors`
   - `clip_projections/mmh3-4b-ClipProj-v3.1.safetensors`
   - `vae/minimax_h3_video_vae_fp16.safetensors`、`vae/minimax_h3_audio_vae_fp32.safetensors`
4. **一行 ComfyUI 核心改动**——`docs/COMFYUI_PATCH.md`（多卡模型的 aimdo 退出）。

## 快速开始

```bash
# 1. 启动 ComfyUI（双卡）
CUDA_VISIBLE_DEVICES=1,0 python main.py --listen 127.0.0.1 --port 8188 \
    --reserve-vram 1.5 --vram-headroom 0.5 --disable-pinned-memory --fast-disk

# 2. 跑 smoke sample（文生视频，无需任何参考图/个人素材）
python scripts/bench.py examples/smoke_t2v.json smoke

# 3. 跑自己的成片（R2V，先照 examples/manifest.template.json 改成自己的参考图 + 剧本）
python scripts/run_film.py --manifest examples/manifest.template.json --tp2
```

## 目录结构

```
custom_nodes/comfyui-h3-tp2/   TP2 节点（切分 + 一卡一线程 forward）
scripts/                       成片流水线、QA、ASR、基准、启动器
probes/                        支撑架构结论的诊断实验
examples/smoke_t2v.json        自包含 smoke sample（文生视频）
examples/manifest.template.json  成片 manifest 模板（R2V，自备参考图）
docs/                          架构 / 性能 / 流水线 / ComfyUI 补丁
```

根目录还有 `AGENTS.md` / `CLAUDE.md`，给 AI 编程代理用（含命令、约束、踩坑清单）。

## 许可与第三方

本仓库代码为 MIT（见 `LICENSE`）。ComfyUI 及若干自定义节点为 GPL / MIT / Apache，见 `THIRD_PARTY_NOTICES.md`。模型权重不在本仓库，遵循各自许可。
