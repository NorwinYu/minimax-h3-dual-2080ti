#!/usr/bin/env python3
"""Caption several frames of each rendered shot to check prop/scene stability over time.

The single mid-frame QA gate can miss a prop that appears late, so this samples the whole
shot and prints every caption; a prop that drifts or vanishes shows up in the text.

    .venv/bin/python multiframe_qa.py [--only <slug>]
"""
import argparse
import json
import os
import sys


HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from run_film import INPUT_DIR, caption, qa_check   # noqa: E402

OUT_DIR = os.environ.get("COMFYUI_OUTPUT", os.path.expanduser("~/ComfyUI/output"))
MANIFEST = os.path.join(HERE, "film_manifest.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    ap.add_argument("--manifest", default=MANIFEST)
    ap.add_argument("--frac", default="0.1,0.35,0.6,0.9")
    a = ap.parse_args()

    m = json.load(open(a.manifest))
    fracs = [float(x) for x in a.frac.split(",")]
    for shot in m["shots"]:
        slug = shot["slug"]
        if a.only and slug != a.only:
            continue
        mp4 = next((os.path.join(OUT_DIR, f) for f in sorted(os.listdir(OUT_DIR))
                    if f.startswith(f"film_{slug}_") and f.endswith(".mp4")), None)
        if not mp4:
            print(f"=== {slug}: 没有渲染输出，跳过")
            continue
        print(f"=== {slug}  expect={shot.get('expect')}")
        import av
        c = av.open(mp4)
        frames = list(c.decode(video=0))
        c.close()
        for fr in fracs:
            png = os.path.join(INPUT_DIR, f"qa_{slug}_{int(fr * 100)}.png")
            idx = min(len(frames) - 1, int(len(frames) * fr))
            frames[idx].to_image().save(png)
            text, err = caption(m, png)
            if err:
                print(f"   帧{idx:3d} ({fr:.0%}): 字幕失败 {err}")
                continue
            hits, ratio, bad = qa_check(text, shot.get("expect"), shot.get("must_not"))
            flag = "✗✗" if bad else ("✔" if ratio >= 0.66 else "·")
            print(f"   {flag} 帧{idx:3d} ({fr:.0%}) {ratio:4.0%} 负向{bad}")
            print(f"      {text}")


if __name__ == "__main__":
    main()
