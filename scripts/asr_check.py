#!/usr/bin/env python3
"""Transcribe a rendered shot's audio with a local whisper-small so dialogue is checkable.

H3 writes speech into the target audio, but nothing else on this machine can read it back;
without this the `<d>` tags are unverifiable claims.

    .venv/bin/python asr_check.py /path/to/shot.mp4
"""
import argparse
import os
import re
import subprocess
import sys
import tempfile

MODEL = os.environ.get("H3_ASR_MODEL", os.path.expanduser("~/models/whisper-small"))


def transcribe(mp4, language="zh"):
    import torch
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    wav = os.path.join(tempfile.mkdtemp(), "a.wav")
    subprocess.run(["ffmpeg", "-v", "error", "-i", mp4, "-vn", "-ac", "1", "-ar", "16000",
                    "-y", wav], check=True)
    proc = WhisperProcessor.from_pretrained(MODEL)
    model = WhisperForConditionalGeneration.from_pretrained(MODEL, dtype=torch.float32)
    model.eval()
    import numpy as np
    import wave
    with wave.open(wav) as w:
        n, sr = w.getnframes(), w.getframerate()
        pcm = np.frombuffer(w.readframes(n), dtype=np.int16)
    audio = pcm.astype(np.float32) / 32768.0
    feats = proc(audio, sampling_rate=sr, return_tensors="pt").input_features
    ids = model.generate(feats, language=language, task="transcribe", max_new_tokens=200)
    return proc.batch_decode(ids, skip_special_tokens=True)[0].strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mp4", nargs="+")
    ap.add_argument("--manifest", default=os.path.join(os.path.dirname(__file__), "..", "examples", "manifest.json"))
    ap.add_argument("--lang", default="zh")
    a = ap.parse_args()

    import json
    shots = {}
    if os.path.exists(a.manifest):
        m = json.load(open(a.manifest))
        for s in m["shots"]:
            d = re.search(r"<d>\[Mandarin\](.*?)</d>", s["prompt"])
            shots[s["slug"]] = d.group(1) if d else None

    for mp4 in a.mp4:
        slug = re.search(r"film_(\w+?)_\d+_\.mp4", os.path.basename(mp4))
        slug = slug.group(1) if slug else ""
        want = shots.get(slug)
        text = transcribe(mp4, a.lang)
        print(f"=== {os.path.basename(mp4)}")
        if want:
            print(f"  期望台词: 「{want}」")
            norm = lambda s: re.sub(r"[^\w]", "", s)
            same = norm(want) == norm(text)
            print(f"  转写结果: 「{text}」   {'✔ 完全一致' if same else '✗ 有差异'}")
        else:
            print(f"  （该镜无对白）转写: 「{text}」")


if __name__ == "__main__":
    main()
