# AGENTS.md

面向 AI 编程代理的项目指引。改动代码前先读这里。

## 这是什么

MiniMax H3（20B 视频+音频联合流式 Transformer）在 **双 RTX 2080 Ti 22GB（sm_75，NVLink）** 上的本地运行时：

- **`custom_nodes/comfyui-h3-tp2/`** —— TP2 张量并行自定义节点（W4A8 量化 + 一卡一线程）
- **`scripts/`** —— 成片流水线（manifest → 渲染 → QA → ASR）、基准、启动器
- **`probes/`** —— 支撑架构结论的诊断实验
- **`examples/`** —— 自包含 smoke sample（文生视频）+ 成片 manifest 模板

## 语言与文档约定

- **代码注释和 README 是中文**；`docs/` 深度文档是英文。**不要**把两边统一成一种语言。
- 提交信息用简短祈使句：`Add …` / `Fix …` / `Make …` / `Use …` / `Remove …` / `Update …`。

## 无 GPU 也能做的校验

本仓库的脚本需要在装好 ComfyUI 的机器上运行（用 ComfyUI 的 `.venv/bin/python`，不是系统 python）。没有 GPU 时只做静态校验：

```bash
# 语法检查（不 import torch/comfy，纯语法）
find . -name "*.py" -not -path "*__pycache__*" -exec python -m py_compile {} +

# JSON 合法性（examples/ 下的 manifest 和 workflow）
python -c "import json,glob; [json.load(open(f)) for f in glob.glob('examples/*.json')]"
```

有 GPU 时的冒烟测试：

```bash
python scripts/bench.py examples/smoke_t2v.json smoke      # 文生视频，无需参考图
python scripts/run_film.py --manifest examples/manifest.template.json --tp2   # 成片
```

## 硬性约束（违反会出错）

1. **不提交**：模型权重（`*.safetensors`）、生成媒体（`*.mp4`/`*.wav`/`*.png`/`*.mp3`）、参考图、日志、`__pycache__/`、`*_production.json`。`.gitignore` 已覆盖，别改松。
2. **不出现私人信息**：默认路径用环境变量（`H3_ASR_MODEL`、`H3_COMFYUI_LOG`、`COMFYUI_INPUT`、`COMFYUI_OUTPUT`），别写死 `~/xxx` 之类的个人路径。
3. **许可**：本仓库代码 MIT；ComfyUI 等第三方见 `THIRD_PARTY_NOTICES.md`。**不要**把 GPL 源码（尤其 ComfyUI 核心）拷贝进来——那行 aimdo 补丁只以"编辑指引"形式存在于 `docs/COMFYUI_PATCH.md`。
4. **设备编号**：`CUDA_VISIBLE_DEVICES=1,0` 下，物理 GPU0 = 桌面卡（ComfyUI `cuda:1`，跑编码器/VAE），物理 GPU1 = 计算卡（ComfyUI `cuda:0`，跑 DiT）。`bench.py` 报告的是**物理** GPU 编号。

## 硬骨头 / 最容易踩的坑（本仓库独有的血泪）

- **W4A8 GEMM 阻塞 host 线程**：comfy-kitchen 的 W4A8 GEMM 会阻塞调用它的线程直到 kernel 结束。所以 TP2 必须"一卡一线程"（`tp2.py` 的 `_par` + `_run`）。把它改回单线程会从 **2.97× 掉回 1.31×**。
- **`torch.inference_mode()` 是线程局部的**：worker 线程默认不在 inference_mode 里，对主线程的 inference tensor 做 inplace 会抛 `Inplace update to inference tensor outside InferenceMode is not allowed`。`_run` 里必须重进调用者的模式。
- **`torch.cuda.synchronize()` 不带参数只同步当前卡**：并发/跨卡测量必须显式 `for i in range(device_count()): synchronize(i)`，否则并发耗时会系统性低估（这也是早期"重叠率 100%"假象的根源）。
- **ComfyUI 模型缓存**：同一 loader 链会复用缓存的模型对象。切换"单卡 vs TP2"对比时**必须重启 ComfyUI 清缓存**，否则第二次跑的其实还是上一次的（实测出现过"单卡"其实还是 TP2）。
- **不要和 ComfyUI 并发跑探针**：本机 31GB RAM，probe 和 ComfyUI 同时加载模型会互相 OOM 污染结果。
- **turbo LoRA 是 4 步**：2 步渲染会时序不稳定（主体模糊闪现）。**测速可用 2 步，出片必须 4 步**。
- **进程清理**：kill ComfyUI 时别用 `pgrep -f 'main.py --listen'`（会匹配到自己的 shell）；用 `ss -ltnp | grep :8188` 拿 PID 再 kill。
- **`readlink -f /proc/PID/exe` 对 venv python 会解析成 `python3.12`**，按 `== "python"` 匹配会漏掉。按 cmdline 匹配。

## 目录结构

```
custom_nodes/comfyui-h3-tp2/   TP2 节点（切分 + 一卡一线程 forward）
scripts/                       成片流水线、QA、ASR、基准、启动器
probes/                        支撑架构结论的诊断实验
examples/                      smoke sample + manifest 模板
docs/                          架构 / 性能 / 流水线 / ComfyUI 补丁（英文）
```

## 性能基准（勿凭印象改数字）

- smoke sample（640×352 × 124 帧 × 4 步）：单卡 243s，TP2 **82s（2.97×）**。
- 成片（864×480 × 124 帧）：单卡 44.6 s/step，TP2 19.8 s/step（2.25×）。
- 温度：单卡 GPU1 峰值 78°C（满载 4 分钟）；TP2 67/75°C（1.4 分钟）。
- 功率上限实测用 250W（180W 会轻微降频，端到端慢约 6%）。
