# 贡献指南

欢迎贡献。动手前请先读 `AGENTS.md`（本仓库的约束和踩坑清单）。

## 报告问题

开 issue 时请附上：

- 硬件（显卡型号/显存/驱动版本）与 `CUDA_VISIBLE_DEVICES` 设置
- 复现用的命令或 workflow（如 `examples/smoke_t2v.json`）
- ComfyUI 日志里报错附近的一段

## 提交改动

1. Fork 本仓库，从 `main` 开分支。
2. 改动前先跑本地校验（无 GPU 也能跑）：

   ```bash
   find . -name "*.py" -not -path "*__pycache__*" -exec python -m py_compile {} +
   python -c "import json,glob; [json.load(open(f)) for f in glob.glob('examples/*.json')]"
   for f in scripts/*.sh; do bash -n "$f"; done
   ```

3. 提交信息用简短祈使句：`Add …` / `Fix …` / `Make …` / `Use …` / `Remove …` / `Update …`。
4. 一次一个行为；不要顺手改无关文件。

## 边界

- **绝不提交**：模型权重（`*.safetensors`）、生成媒体（`*.mp4`/`*.wav`/`*.png`/`*.mp3`）、参考图、日志、`__pycache__/`、`*_production.json`。
- **不写死私人路径**：默认路径用环境变量（`H3_ASR_MODEL`、`COMFYUI_INPUT` 等）。
- **不引入 GPL 源码**：本仓库代码是 MIT；ComfyUI 核心的改动只以编辑指引形式放在 `docs/COMFYUI_PATCH.md`。
- **语言**：代码注释和 README 用中文，`docs/` 深度文档用英文，不要混着改。

## 性能数字

改动影响性能时，用 `scripts/bench.py examples/smoke_t2v.json` 在同一样本上对比，并把数据写进 PR 描述（不要凭印象改 `docs/PERFORMANCE.md` 或 README 里的数字）。

## 许可

提交即表示同意在本仓库 MIT 许可下分发你的改动（见 `LICENSE`）。
