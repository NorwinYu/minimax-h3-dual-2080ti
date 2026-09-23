#!/usr/bin/env python3
"""Build the micro-film's R2V workflows from film_manifest.json.

One source of truth: edit prompts/shots in the manifest, not in the JSON graphs.

  python3 make_film.py            # write wf_film_<slug>.json for every shot
  python3 make_film.py --doc      # also emit FILM_SCRIPT.md (Chinese shooting script)
  python3 make_film.py --ref X    # preview which reference image would be used

Every shot uses MiniMaxH3ReferenceToVideo (character reference, free scene). First-frame
chaining is deliberately NOT used: it pins the composition, so it can only continue one
continuous take, not cut between scenes.
"""
import argparse, json, os

HERE = os.path.dirname(os.path.abspath(__file__))
MANIFEST = os.path.join(HERE, "film_manifest.json")


def expand(m, text):
    """Expand {KFC_BUCKET}/{ROOM}/{CHAR} so every shot uses byte-identical prop wording."""
    for k, v in (m.get("props") or {}).items():
        text = text.replace("{" + k + "}", v)
    return text


def build_shot(m, shot, ref_image, seed, env_image=None):
    """API-format R2V graph for one shot. Up to two references: character + set."""
    w, h, n = m["width"], m["height"], m["frames_per_shot"]
    ref2 = shot.get("ref2") or (env_image if shot.get("use_environment", True) else None)
    use_env = bool(ref2)
    wf = {
        "1":  {"class_type": "MiniMaxH3FP16LoaderStar7",
               "_meta": {"title": f"DiT: {m['dit']}"},
               "inputs": {"unet_name": m["dit"]}},
        "1b": {"class_type": "MiniMaxH3FP16ExactFixStar7",
               "inputs": {"model": ["1", 0], "enabled": True}},
        "2":  {"class_type": "LoraLoaderModelOnly",
               "inputs": {"model": ["1b", 0], "lora_name": m["lora"], "strength_model": 1.0}},
        "3":  {"class_type": "MiniMaxH3ActivationChunkStar7",
               "_meta": {"title": f"分块 + 注意力 {m['attention']}"},
               "inputs": {"model": ["2", 0], "chunk_tokens": 8192, "auto_halve_on_oom": True,
                          "verbose": True, "mlp_chunk_tokens": 4096,
                          "disable_dynamic_prefetch": "off", "qkv_chunk_tokens": 4096,
                          "out_proj_chunk_tokens": 4096, "reuse_mlp_weights": True,
                          "attention_backend": m["attention"]}},
        "4":  {"class_type": "MiniMaxH3SigmaShift",
               "_meta": {"title": "视频 shift 12 / 音频 shift 3"},
               "inputs": {"model": ["3", 0], "shift_video": 12.0, "shift_audio": 3.0}},
        "5":  {"class_type": "ClipProjLoader",
               "_meta": {"title": "4B 编码器 + 投影（钉在 GPU0）"},
               "inputs": {"clip_name": m["encoder"], "type": "auto",
                          "projection": m["projection"], "device": "cuda:1", "mode": "resident"}},
        "6":  {"class_type": "VAELoaderMultiGPU",
               "inputs": {"vae_name": "minimax_h3_video_vae_fp16.safetensors", "device": "cuda:1"}},
        "7":  {"class_type": "VAELoaderMultiGPU",
               "inputs": {"vae_name": "minimax_h3_audio_vae_fp32.safetensors", "device": "cuda:1"}},
        "8":  {"class_type": "MiniMaxH3ReferenceToVideo",
               "_meta": {"title": f"{shot['start']} {shot['beat']}｜{shot['shot']}"},
               "inputs": {"clip": ["5", 0], "vae": ["6", 0], "audio_vae": ["7", 0],
                          "prompt": expand(m, shot["prompt"]), "width": w, "height": h,
                          "length": n, "ref_image_size": m["ref_image_size"],
                          "ref_images.ref_image_0": ["17", 0]}},
        "9":  {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
        "10": {"class_type": "BasicGuider",
               "inputs": {"model": ["4", 0], "conditioning": ["8", 0]}},
        "11": {"class_type": "BasicScheduler",
               "inputs": {"model": ["4", 0], "scheduler": "simple",
                          "steps": m["steps"], "denoise": 1.0}},
        "12": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
        "13": {"class_type": "SamplerCustomAdvanced",
               "inputs": {"noise": ["9", 0], "guider": ["10", 0], "sampler": ["12", 0],
                          "sigmas": ["11", 0], "latent_image": ["8", 1]}},
        "14": {"class_type": "MiniMaxH3ChunkedDecodeStar7",
               "_meta": {"title": "视频+音频解码（VAE 在 GPU0）"},
               "inputs": {"av_latent": ["13", 0], "video_vae": ["6", 0], "audio_vae": ["7", 0]}},
        "15": {"class_type": "CreateVideo",
               "inputs": {"images": ["14", 0], "fps": float(m["fps"]), "audio": ["14", 1]}},
        "16": {"class_type": "SaveVideo",
               "inputs": {"video": ["15", 0], "filename_prefix": f"film_{shot['slug']}",
                          "format": "auto", "codec": "auto"}},
        "17": {"class_type": "LoadImage",
               "_meta": {"title": f"角色参考图: {ref_image}"},
               "inputs": {"image": ref_image}},
    }
    if use_env:
        wf["18"] = {"class_type": "LoadImage",
                    "_meta": {"title": f"第二参考图: {ref2}"},
                    "inputs": {"image": ref2}}
        wf["8"]["inputs"]["ref_images.ref_image_1"] = ["18", 0]
        # a person's identity has to ride in on a picture; the prompt alone drifts across shots.
        # Appended, not prepended: mentioning <Picture 2> first made the model treat the
        # person as the primary reference and drop the main subject.
        wf["8"]["inputs"]["prompt"] = wf["8"]["inputs"]["prompt"] + (
            " <Picture 2> is the human character reference and drives that person's face, build "
            "and costume only; keep them identical to Picture 2.")
    return wf


def add_tp2(wf, replica_device="cuda:1"):
    """Run the DiT tensor-parallel across two GPUs instead of the chunked single-GPU path."""
    wf["105"] = {"class_type": "MiniMaxH3TP2",
                 "_meta": {"title": f"TP2 双卡张量并行（副本 {replica_device}）"},
                 "inputs": {"model": ["4", 0], "replica_device": replica_device}}
    wf["10"]["inputs"]["model"] = ["105", 0]
    wf["11"]["inputs"]["model"] = ["105", 0]
    return wf


def build_qa(m, frame_image, seed=0):
    """Caption one frame with the 4B vision encoder and write the text to disk.

    This is the QA gate: it turns 'I cannot see the output' into a checkable string,
    which is exactly the failure mode that let a wrong shot-1 pass earlier.
    """
    return {
        "1": {"class_type": "ClipProjDeviceLoader",
              "_meta": {"title": "原样编码器（不做投影）用于描述"},
              "inputs": {"clip_name": m["encoder"], "type": "auto",
                         "device": "cuda:1", "mode": "resident"}},
        "2": {"class_type": "LoadImage", "inputs": {"image": frame_image}},
        "3": {"class_type": "ClipProjGenerate",
              "_meta": {"title": "对画面做事实性描述"},
              "inputs": {"clip": ["1", 0],
                         "system": "You are a precise, literal image describer. Describe only what is visible.",
                         "prompt": "Describe this image in one factual sentence: the main subject, what it is doing, and the setting.",
                         "max_length": 96, "temperature": 0.2, "top_p": 0.9, "top_k": 40,
                         "seed": seed, "image": ["2", 0]}},
        "4": {"class_type": "SaveText",
              "_meta": {"title": "描述落盘，供脚本比对"},
              "inputs": {"text": ["3", 0], "filename_prefix": "qa/caption", "format": "txt"}},
    }


def write_doc(m, path):
    lines = [f"# 《{m['title']}》 *{m['title_en']}*", "",
             f"**一句话故事**：{m['logline']}", "",
             f"**规格**：{m['runtime_seconds']} 秒 · {len(m['shots'])} 镜 × {m['frames_per_shot']/m['fps']:.3f} 秒 · "
             f"{m['width']}×{m['height']} @ {m['fps']}fps · 原生 32 kHz 立体声", "",
             f"**技术**：{m['dit']} + {m['encoder']} + {m['projection']}｜"
             f"注意力 {m['attention']}｜{m['steps']} 步 + Turbo LoRA", "",
             "| # | 时间 | 幕/节拍 | 画面与运镜 | 声音 |", "|---|---|---|---|---|"]
    for i, s in enumerate(m["shots"], 1):
        lines.append(f"| {i} | {s['start']} | {s['act']}·{s['beat']} | {s['shot']} | {s['audio']} |")
    lines += ["", "## 分镜提示词", ""]
    for i, s in enumerate(m["shots"], 1):
        lines += [f"### 镜 {i} · {s['slug']}（{s['start']}｜{s['act']}·{s['beat']}）", "",
                  f"> {s['prompt']}", "",
                  f"自动校验关键词：`{'`, `'.join(s['expect'])}`", ""]
    lines += ["## 结构说明", "",
              "- **第一幕（0:00–0:20）建立**：主角登场 + 目标确立（守住炸鸡桶）。",
              "- **第二幕（0:20–0:40）冲突**：外部——鸽子来偷；内部——柯基打开桶、叼出最后一块、又放回去。内心戏是让角色立体的关键。",
              "- **第三幕（0:40–1:00）低谷与解决**：黄昏无人归来 → 夜里钥匙转动、门开、冲向门口。",
              "- 全片**不出现人类**，始终留在柯基视角——主角是狗，不是人。",
              "- 所有镜头用 **R2V 角色参考**，不用首帧接力：镜头之间是「剪辑」，首帧接力会把构图锁死。", ""]
    open(path, "w").write("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", help="角色参考图文件名（需已在 ComfyUI/input/）")
    ap.add_argument("--doc", action="store_true", help="额外导出 FILM_SCRIPT.md")
    ap.add_argument("--only", help="只生成指定 slug")
    ap.add_argument("--env", default="set_living_room.png",
                    help="场景设定图（ComfyUI/input/ 内），用于统一室内环境")
    a = ap.parse_args()

    m = json.load(open(MANIFEST))
    ref = a.ref or m["reference_image"]

    for i, shot in enumerate(m["shots"]):
        if a.only and shot["slug"] != a.only:
            continue
        seed = m["base_seed"] + i * 1000
        wf = build_shot(m, shot, ref, seed, env_image=a.env)
        path = os.path.join(HERE, f"wf_film_{shot['slug']}.json")
        json.dump(wf, open(path, "w"), indent=2, ensure_ascii=False)
        print(f"  {shot['slug']:16s} seed={seed:<6} ref={ref:20s} -> {os.path.basename(path)}")

    if a.doc:
        write_doc(m, os.path.join(HERE, "FILM_SCRIPT.md"))
        print("  已导出 FILM_SCRIPT.md")

    print(f"\n  {len(m['shots'])} 镜 × {m['frames_per_shot']/m['fps']:.3f}s = {m['runtime_seconds']}s")


if __name__ == "__main__":
    main()
