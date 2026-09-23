# Third-Party Notices

This project is a thin integration layer around several third-party projects.
The code in this repository is original (MIT-licensed, see `LICENSE`); the
projects below are **runtime dependencies** and are distributed under their
own licenses. You must install them separately.

| Project | License | Used for | Notes |
|---|---|---|---|
| [ComfyUI](https://github.com/comfyanonymous/ComfyUI) | GPL-3.0 | Execution engine, model management, nodes | This repo ships **no ComfyUI code**; see `docs/COMFYUI_PATCH.md` for a required one-line edit. |
| [comfy-kitchen](https://github.com/Comfy-Org/comfy_kitchen) | (Comfy Org) | `AsymW4A8Int8Layout`, `QuantizedTensor`, W4A8 GEMM, int8 attention | Ships with ComfyUI; imported by `comfyui-h3-tp2`. |
| [ComfyUI-MultiGPU](https://github.com/comfyanonymous/ComfyUI-MultiGPU) | GPL-3.0 | P2P-aware DLPack guard | Installed as `comfyui-multigpu`. |
| [minimax-h3-chunk-star7](https://github.com/...) | MIT | `MiniMaxH3ActivationChunkStar7` chunked forward, `adaptive_loader` | Required custom node. |
| [minimax-h3-fp16-exact-star7](https://github.com/...) | MIT | `MiniMaxH3FP16ExactFixStar7` (FP16-Exact companion) | Required custom node. |
| [comfyui-minimax-h3-turbo](https://github.com/...) | Apache-2.0 | Turbo LoRA loading | Required custom node. |
| [minimax-h3-audio-t8](https://github.com/...) | GPL-3.0-or-later | Native voice / dialogue workflows (optional) | Only needed for the dialogue extras. |
| [MiniMax H3](https://huggingface.co/MiniMaxAI) | model license | The DiT / VAE / encoder / LoRA weights | Weights are **not** in this repo; download from MiniMax and review their license. |
| [OpenAI Whisper](https://github.com/openai/whisper) | MIT | Local dialogue verification (`scripts/asr_check.py`) | Model downloaded on demand via `hf-mirror`; see `docs/FILM_PIPELINE.md`. |
| [PyAV](https://github.com/PyAV-Org/PyAV), [Pillow](https://python-pillow.org), [transformers](https://github.com/huggingface/transformers), [safetensors](https://github.com/huggingface/safetensors) | permissive | Video frame extraction, image crops, ASR | Listed in `requirements.txt`. |

## Why MIT, when ComfyUI is GPL?

- Every `.py` file in this repo is original code written for this project; no
  ComfyUI (or other GPL) source is copied or vendored.
- `comfyui-h3-tp2` **imports** and, at runtime, monkeypatches ComfyUI internals.
  That is runtime interoperability, not derivation, so it stays MIT here. If your
  legal reading differs, treat `custom_nodes/` as GPL-3.0-or-later to match ComfyUI.
- The one change that touches ComfyUI core (`comfy/ldm/minimax/model.py`) is
  **not distributed**: it is described as an instruction in
  `docs/COMFYUI_PATCH.md`, so this repository never redistributes GPL code.

If you fork, replace the copyright holder in `LICENSE` and re-verify the model
weights' own terms before redistributing anything.
