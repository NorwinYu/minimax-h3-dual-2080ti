#!/usr/bin/env python3
"""Drive 《留给你的那块》: render every shot with R2V, QA each one, then finish the film.

  python3 run_film.py --ref my_character.png        # full run (copies ref into ComfyUI/input/)
  python3 run_film.py --ref my_character.png --only <slug>
  python3 run_film.py --finish-only             # just re-concat whatever exists
  python3 run_film.py --qa-only                 # re-caption existing shots, no rendering

Why the QA step exists: the model cannot see images, so a shot whose prompt was ignored
looks identical to a correct one until a human watches it. Captioning the middle frame with
the 4B vision encoder turns that into a checkable string, and the manifest's expect[] list is
the keyword set to check against. A miss is reported loudly, never silently accepted.
"""
import argparse, glob, json, os, re, shutil, subprocess, sys, tempfile, time, urllib.request

import av
import imageio_ffmpeg

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from make_film import MANIFEST, build_shot, build_qa, add_tp2    # noqa: E402

API = "http://127.0.0.1:8188"
INPUT_DIR = os.environ.get("COMFYUI_INPUT", os.path.expanduser("~/ComfyUI/input"))
OUT_DIR = os.environ.get("COMFYUI_OUTPUT", os.path.expanduser("~/ComfyUI/output"))


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(os.path.join(HERE, "film_driver.log"), "a") as f:
        f.write(line + "\n")


def submit(wf):
    req = urllib.request.Request(f"{API}/prompt",
        data=json.dumps({"prompt": wf, "client_id": "film"}).encode(),
        headers={"Content-Type": "application/json"})
    try:
        return json.load(urllib.request.urlopen(req, timeout=90))["prompt_id"], None
    except urllib.error.HTTPError as e:
        return None, e.read().decode()[:500]


def wait(pid, timeout=3600):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            h = json.load(urllib.request.urlopen(f"{API}/history/{pid}", timeout=10))
        except Exception:
            h = {}
        if h:
            e = list(h.values())[0]
            if e.get("status", {}).get("completed"):
                return e["status"].get("status_str"), e.get("outputs", {}), time.time() - t0
        time.sleep(5)
    return "TIMEOUT", {}, time.time() - t0


def out_mp4(outputs):
    for o in outputs.values():
        for k, v in o.items():
            if k == "images" and isinstance(v, list) and v:
                p = os.path.join(OUT_DIR, v[0].get("subfolder", ""), v[0]["filename"])
                return p if os.path.exists(p) else None
    return None


def grab_frame(mp4, which, png):
    """which: 'first' | 'mid' | 'last'"""
    c = av.open(mp4)
    frames = list(c.decode(video=0))
    c.close()
    if not frames:
        return None
    idx = {"first": 0, "mid": len(frames) // 2, "last": len(frames) - 1}[which]
    frames[idx].to_image().save(png)
    return png


def caption(m, png):
    """Run the QA graph; return (caption_text, error)."""
    wf = build_qa(m, os.path.basename(png))
    before = set(glob.glob(os.path.join(OUT_DIR, "qa", "caption*.txt")))
    pid, err = submit(wf)
    if not pid:
        return None, f"提交失败 {err}"
    status, _, _ = wait(pid, timeout=900)
    if status != "success":
        return None, f"QA 状态 {status}"
    after = set(glob.glob(os.path.join(OUT_DIR, "qa", "caption*.txt"))) - before
    if not after:
        return None, "没找到描述文件"
    newest = max(after, key=os.path.getmtime)
    return open(newest, encoding="utf-8", errors="ignore").read().strip(), None


def qa_check(text, expect, must_not=None):
    """Positive scene anchors + negative gate. A must_not hit is a hard fail.

    The negative list exists because the first film passed QA at "100%" on captions that
    literally read 'a chicken in a metal bucket' — the positive words 'chicken' and
    'bucket' matched while the prop had drifted to a live bird in a tin can.
    """
    low = (text or "").lower()

    def says(w):
        # word boundaries, not substring: naive matching flagged "sitting" as "tin".
        # A trailing plural is the same anchor, so "ear" matches the caption's "ears".
        return re.search(rf"(?<![a-z]){re.escape(w.lower())}s?(?![a-z])", low) is not None

    hits = [w for w in expect if says(w)]
    bad = [w for w in (must_not or []) if says(w)]
    ratio = len(hits) / len(expect) if expect else 0.0
    return hits, ratio, bad


def finish(files, out_path, m):
    """Concat losslessly, then one pass of fades + EBU R128 loudness normalisation."""
    exe = imageio_ffmpeg.get_ffmpeg_exe()
    dur = len(files) * m["frames_per_shot"] / m["fps"]
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        for p in files:
            f.write(f"file '{p}'\n")
        lst = f.name
    vf = f"fade=t=in:st=0:d=0.7,fade=t=out:st={dur-1.0:.2f}:d=0.9"
    af = (f"loudnorm=I=-16:TP=-1.5:LRA=11,"
          f"afade=t=in:st=0:d=0.7,afade=t=out:st={dur-1.0:.2f}:d=0.9")
    cmd = [exe, "-y", "-f", "concat", "-safe", "0", "-i", lst,
           "-vf", vf, "-af", af,
           "-c:v", "libx264", "-crf", "18", "-preset", "medium", "-pix_fmt", "yuv420p",
           "-c:a", "aac", "-b:a", "192k", "-ar", str(m["fps"] * 0 + 32000),
           "-movflags", "+faststart", out_path]
    r = subprocess.run(cmd, capture_output=True, text=True)
    os.unlink(lst)
    if r.returncode != 0:
        log(f"成片失败: {r.stderr[-700:]}")
        return None
    c = av.open(out_path)
    v, a = c.streams.video[0], c.streams.audio[0]
    info = dict(frames=v.frames, seconds=float(v.duration * v.time_base),
                sr=a.codec_context.sample_rate, ch=a.codec_context.layout.nb_channels,
                mb=os.path.getsize(out_path) / 1e6)
    c.close()
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", help="角色参考图路径（会复制进 ComfyUI/input/）")
    ap.add_argument("--only", help="只处理指定 slug")
    ap.add_argument("--finish-only", action="store_true")
    ap.add_argument("--qa-only", action="store_true", help="只补做 QA，不渲染")
    ap.add_argument("--skip-qa", action="store_true")
    ap.add_argument("--tp2", action="store_true", help="DiT 走双卡张量并行")
    ap.add_argument("--replica-device", default="cuda:1")
    ap.add_argument("--out", default="h3_microfilm.mp4")
    ap.add_argument("--manifest", default=MANIFEST, help="改用其他 manifest")
    a = ap.parse_args()

    m = json.load(open(a.manifest))
    prod_path = a.manifest.replace("_manifest.json", "_production.json")
    ref = m["reference_image"]
    env = m.get("environment_image", "set_living_room.png")
    if a.ref:
        cand = [os.path.abspath(os.path.expanduser(a.ref)),
                os.path.join(INPUT_DIR, os.path.basename(a.ref)),
                os.path.expanduser(f"~/Downloads/{os.path.basename(a.ref)}")]
        src = next((c for c in cand if os.path.exists(c)), None)
        if src is None:
            log(f"参考图不存在，找过这些位置: {cand}")
            return 1
        ref = os.path.basename(src)
        dst = os.path.join(INPUT_DIR, ref)
        if os.path.abspath(src) != os.path.abspath(dst):
            shutil.copy2(src, dst)
        else:
            log(f"参考图已在 ComfyUI/input/，无需复制")
        m["reference_image"] = ref
        json.dump(m, open(a.manifest, "w"), indent=2, ensure_ascii=False)
        log(f"参考图已就位: {ref}（{os.path.getsize(os.path.join(INPUT_DIR, ref))/1e6:.2f} MB）")

    prod = json.load(open(prod_path)) if os.path.exists(prod_path) else {"shots": {}}

    if not a.finish_only:
        log(f"=== 《{m['title']}》开始 · {len(m['shots'])} 镜 · 参考图 {ref} ===")
        for i, shot in enumerate(m["shots"]):
            slug = shot["slug"]
            if a.only and slug != a.only:
                continue
            seed = m["base_seed"] + i * 1000
            prefix = f"film_{slug}"
            rec = prod["shots"].setdefault(slug, {})

            existing = sorted(glob.glob(os.path.join(OUT_DIR, f"{prefix}_*.mp4")))
            if existing and not a.qa_only:
                rec["file"] = existing[-1]
                log(f"{slug}: 已有输出，跳过渲染")
            elif not a.qa_only:
                wf = build_shot(m, shot, ref, seed, env_image=env)
                if a.tp2:
                    add_tp2(wf, a.replica_device)
                pid, err = submit(wf)
                if not pid:
                    log(f"{slug}: 提交失败 {err}")
                    return 1
                log(f"{slug}: 已提交 {pid[:8]}（seed {seed}）…")
                status, outputs, dt = wait(pid)
                mp4 = out_mp4(outputs)
                if status != "success" or not mp4:
                    log(f"{slug}: 失败 status={status}")
                    return 1
                rec.update(file=mp4, seed=seed, seconds=round(dt, 1))
                log(f"{slug}: 完成 {os.path.basename(mp4)} 用时 {dt:.0f}s")

            if not a.skip_qa and rec.get("file"):
                png = os.path.join(INPUT_DIR, f"qa_{slug}.png")
                if grab_frame(rec["file"], "mid", png):
                    txt, err = caption(m, png)
                    if err:
                        rec["qa"] = {"error": err}
                        log(f"{slug}: QA 失败 {err}")
                    else:
                        hits, ratio, bad = qa_check(txt, shot["expect"], shot.get("must_not"))
                        rec["qa"] = {"caption": txt, "hits": hits, "ratio": round(ratio, 2),
                                     "must_not_hits": bad}
                        if bad:
                            flag = "✗✗ 负向命中"
                        elif ratio >= 0.6:
                            flag = "✔"
                        elif ratio >= 0.4:
                            flag = "△"
                        else:
                            flag = "✗"
                        extra = f"  ⚠负向命中{bad}" if bad else ""
                        log(f"{slug}: QA {flag} {ratio:.0%} 命中{hits}{extra} | “{txt[:110]}”")
            json.dump(prod, open(prod_path, "w"), indent=2, ensure_ascii=False)

    files = [prod["shots"][s["slug"]]["file"] for s in m["shots"]
             if s["slug"] in prod["shots"] and prod["shots"][s["slug"]].get("file")]
    if len(files) != len(m["shots"]):
        log(f"只找到 {len(files)}/{len(m['shots'])} 镜，暂不成片")
        return 1
    log(f"全部 {len(files)} 镜就绪，成片（淡入淡出 + 响度归一化 -16 LUFS）…")
    info = finish(files, os.path.join(OUT_DIR, a.out), m)
    if info:
        log(f"成片完成: {a.out} | {info['frames']} 帧 | {info['seconds']:.2f}s | "
            f"{info['sr']}Hz {info['ch']}声道 | {info['mb']:.1f} MB")
    log("=== 结束 ===")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        import traceback
        tb = traceback.format_exc()
        print(tb, flush=True)
        with open(os.path.join(HERE, "film_driver.log"), "a") as f:
            f.write(tb + "\n")
        sys.exit(1)
