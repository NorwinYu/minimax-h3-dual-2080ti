# CLAUDE.md

Claude Code 在本仓库工作时的记忆与约束。完整的坑清单见 `AGENTS.md`，这里保留最关键的、最容易犯错的。

## 项目

MiniMax H3（20B 视频+音频联合流式 Transformer）在**双 RTX 2080 Ti 22GB（sm_75，NVLink）**上的本地运行时。核心是 `custom_nodes/comfyui-h3-tp2/` 的 TP2 张量并行节点（W4A8 量化 + 一卡一线程），加一套成片流水线（`scripts/`）和自包含 smoke sample（`examples/smoke_t2v.json`）。

## 常用命令

```bash
# 静态校验（无 GPU 也能跑；脚本用 ComfyUI 的 venv，不是系统 python）
find . -name "*.py" -not -path "*__pycache__*" -exec python -m py_compile {} +
python -c "import json,glob; [json.load(open(f)) for f in glob.glob('examples/*.json')]"

# 有 GPU：冒烟测试（文生视频，无需参考图）
python scripts/bench.py examples/smoke_t2v.json smoke

# 有 GPU：跑成片（先照模板改 manifest.template.json 的 reference_image + 剧本）
python scripts/run_film.py --manifest examples/manifest.template.json --tp2
```

## 环境与设备编号

- 脚本要在装好 ComfyUI 的机器上运行，用 ComfyUI 的 `.venv/bin/python`。
- `CUDA_VISIBLE_DEVICES=1,0`：物理 GPU0 = 桌面卡（ComfyUI `cuda:1`，编码器/VAE），物理 GPU1 = 计算卡（ComfyUI `cuda:0`，DiT）。`bench.py` 报的是**物理**编号。
- 功率上限 250W（180W 会轻微降频，端到端慢约 6%）。

## 硬性约束

1. **不提交**：`*.safetensors`、`*.mp4`/`*.wav`/`*.png`/`*.mp3`、参考图、日志、`__pycache__/`、`*_production.json`。
2. **不出现私人信息**：默认路径用环境变量（`H3_ASR_MODEL`、`H3_COMFYUI_LOG`、`COMFYUI_INPUT`、`COMFYUI_OUTPUT`），不写死个人 `~/xxx` 路径。
3. **许可**：本仓库代码 MIT；第三方见 `THIRD_PARTY_NOTICES.md`。**别把 GPL 源码拷贝进来**——那行 ComfyUI aimdo 补丁只以"编辑指引"存在于 `docs/COMFYUI_PATCH.md`。
4. **语言**：代码注释和 README 是中文，`docs/` 是英文。不要统一成一种语言。

## 最容易踩的坑（改代码前先看）

- **W4A8 GEMM 阻塞 host 线程**：comfy-kitchen 的 W4A8 GEMM 会阻塞调用线程直到 kernel 结束。TP2 必须"一卡一线程"（`tp2.py` 的 `_par`/`_run`）。改回单线程会从 2.97× 掉到 1.31×。
- **`torch.inference_mode()` 线程局部**：worker 线程默认不在 inference_mode 里，对主线程 inference tensor 做 inplace 会抛异常。`_run` 里必须重进。
- **`torch.cuda.synchronize()` 无参只同步当前卡**：跨卡并发测量必须 `for i in range(device_count()): synchronize(i)`，否则并发耗时被低估。
- **ComfyUI 模型缓存**：切"单卡 vs TP2"对比必须**重启 ComfyUI 清缓存**，否则第二次还是上一次的模型。
- **别和 ComfyUI 并发跑探针**：31GB RAM 会互相 OOM 污染。
- **turbo LoRA 是 4 步**：测速可 2 步，出片必须 4 步，否则主体模糊闪现。
- **kill 姿势**：别用 `pgrep -f 'main.py --listen'`（会匹配到自己）；用 `ss -ltnp | grep :8188` 拿 PID。`readlink -f /proc/PID/exe` 对 venv 会解析成 `python3.12`，按 `== "python"` 匹配会漏。

## 性能基准（勿凭印象改数字）

- smoke（640×352×124f×4 步）：单卡 243s，TP2 **82s（2.97×）**。
- 成片（864×480×124f）：单卡 44.6 s/step，TP2 19.8 s/step（2.25×）。
- 温度：单卡 GPU1 峰值 78°C（满载 4 分钟）；TP2 67/75°C（1.4 分钟）。
- 线程化改动是纯调度优化，输出逐比特一致（帧 MAE 0.0000）。
