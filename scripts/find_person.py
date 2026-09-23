#!/usr/bin/env python3
"""Crop a full frame into a grid, caption every tile, and report where the subject is.

A person reference has to be a tight crop: a full 1344x768 scene frame is fed to the model
as scene tokens, and it then copies the room instead of the face (measured: the woman
disappeared from the shot entirely). This finds the tile that is mostly the subject.

    .venv/bin/python find_person.py --image /path/to/frame.png --want woman
"""
import argparse
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

INDIR = os.environ.get("COMFYUI_INPUT", os.path.expanduser("~/ComfyUI/input"))
OUTDIR = os.environ.get("COMFYUI_OUTPUT", os.path.expanduser("~/ComfyUI/output"))


def caption_wf(encoder, png_name, question):
    return {
        "1": {"class_type": "ClipProjDeviceLoader",
              "inputs": {"clip_name": encoder, "type": "auto",
                         "device": "cuda:1", "mode": "resident"}},
        "2": {"class_type": "LoadImage", "inputs": {"image": png_name}},
        "3": {"class_type": "ClipProjGenerate",
              "inputs": {"clip": ["1", 0],
                         "system": "You are a precise, literal image describer. Describe only what is visible.",
                         "prompt": question,
                         "max_length": 80, "temperature": 0.2, "top_p": 0.9, "top_k": 40,
                         "seed": 0, "image": ["2", 0]}},
        "4": {"class_type": "SaveText",
              "inputs": {"text": ["3", 0], "filename_prefix": "qa/tile", "format": "txt"}},
    }


def main():
    from run_film import submit, wait           # noqa: E402
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--want", default="woman")
    ap.add_argument("--encoder", default="qwen3vl_4b_fp8_scaled.safetensors")
    ap.add_argument("--cols", type=int, default=3)
    ap.add_argument("--rows", type=int, default=2)
    a = ap.parse_args()

    from PIL import Image
    im = Image.open(a.image)
    W, H = im.size
    print(f"源图 {W}x{H}  目标主体: {a.want}")

    tiles = []
    for r in range(a.rows):
        for c in range(a.cols):
            x0, x1 = int(W * c / a.cols), int(W * (c + 1) / a.cols)
            y0, y1 = int(H * r / a.rows), int(H * (r + 1) / a.rows)
            # 1/3 重叠，避免把人正好切在缝上
            x0 = max(0, x0 - (x1 - x0) // 4); x1 = min(W, x1 + (x1 - x0) // 4)
            y0 = max(0, y0 - (y1 - y0) // 4); y1 = min(H, y1 + (y1 - y0) // 4)
            tiles.append((f"tile_{r}{c}", im.crop((x0, y0, x1, y1))))

    q = ("Describe this image in one factual sentence: the main subject, what it is doing, "
         "and the setting.")
    for name, t in tiles:
        p = os.path.join(INDIR, f"{name}.png")
        t.save(p)
        before = set(glob.glob(os.path.join(OUTDIR, "qa", "tile*.txt")))
        pid, err = submit(caption_wf(a.encoder, f"{name}.png", q))
        if not pid:
            print(f"  {name}: 提交失败 {err}"); continue
        st, _, _ = wait(pid, timeout=300)
        after = set(glob.glob(os.path.join(OUTDIR, "qa", "tile*.txt"))) - before
        if st != "success" or not after:
            print(f"  {name}: 失败 {st}"); continue
        txt = open(max(after, key=os.path.getmtime), encoding="utf-8", errors="ignore").read().strip()
        hit = a.want.lower() in txt.lower()
        print(f"  {'✔' if hit else '·'} {name} ({t.size[0]}x{t.size[1]}): {txt}")


if __name__ == "__main__":
    main()
