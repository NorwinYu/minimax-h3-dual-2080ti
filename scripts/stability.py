#!/usr/bin/env python3
"""Temporal stability metric — objective proxy for warping / "uncanny" motion.

A sudden distortion (mouth ballooning open, limb melting) shows up as a spike in the
frame-to-frame difference. This measures that spike, so step-count and pipeline changes
can be compared with numbers instead of impressions.

  motion[i]  = mean |frame[i] - frame[i-1]|            (how much is moving)
  spike      = max / median(motion)                    (how abrupt the worst jump is)
  jerk       = mean |motion[i] - motion[i-1]| / median (how jerky the motion profile is)
  frozen     = fraction of frames with motion < 0.05   (stuck / stalled frames)

Usage: stability.py <mp4> [<mp4> ...]      or      stability.py --pair dirA dirB shotname
"""
import glob, os, sys

import av
import numpy as np


def motion_profile(path, width=432):
    c = av.open(path)
    prev = None
    out = []
    for fr in c.decode(video=0):
        a = fr.to_ndarray(format="rgb24").astype(np.float32)
        # downscale by simple slicing for speed; full res adds nothing to the metric
        if a.shape[1] > width:
            step = a.shape[1] // width
            a = a[:, ::step, :]
        if prev is not None:
            out.append(float(np.abs(a - prev).mean()))
        prev = a
    c.close()
    return np.array(out)


def report(path, label=None):
    m = motion_profile(path)
    if len(m) < 3:
        print(f"  {label or path}: 帧数不足")
        return None
    med = float(np.median(m))
    spike = float(m.max() / med) if med > 1e-6 else float("inf")
    jerk = float(np.abs(np.diff(m)).mean() / med) if med > 1e-6 else float("inf")
    frozen = float((m < 0.05).mean())
    p95 = float(np.percentile(m, 95))
    r = dict(label=label or os.path.basename(path), frames=len(m) + 1,
             motion=float(m.mean()), median=med, p95=p95, max=float(m.max()),
             spike=spike, jerk=jerk, frozen=frozen)
    print(f"  {r['label']:34s} 帧{r['frames']:4d} | 平均运动 {r['motion']:5.2f} "
          f"中位 {med:5.2f} p95 {p95:6.2f} 峰值 {r['max']:6.2f} | "
          f"突变倍率 {spike:5.2f} | 抖动 {jerk:5.2f} | 冻结帧 {frozen*100:4.1f}%")
    return r


def main():
    if len(sys.argv) >= 4 and sys.argv[1] == "--pair":
        dirA, dirB, shot = sys.argv[2], sys.argv[3], sys.argv[4]
        def pick(d):
            # 只要正式镜头文件，排除 key_/test_/set_/anchor_ 等中间产物
            cands = [f for f in glob.glob(os.path.join(d, f"*{shot}*.mp4"))
                     if not os.path.basename(f).startswith(("key_", "test_", "set_", "anchor_"))]
            return sorted(cands)
        a, b = pick(dirA), pick(dirB)
        print(f"=== 同镜头对比: {shot} ===")
        ra = report(a[-1], f"旧 {os.path.basename(a[-1])}") if a else None
        rb = report(b[-1], f"新 {os.path.basename(b[-1])}") if b else None
        if ra and rb:
            print(f"  → 突变倍率 {ra['spike']:.2f} → {rb['spike']:.2f} "
                  f"({'改善' if rb['spike'] < ra['spike'] else '变差'} "
                  f"{abs(1-rb['spike']/ra['spike'])*100:.0f}%)")
            print(f"  → 抖动     {ra['jerk']:.2f} → {rb['jerk']:.2f} "
                  f"({'改善' if rb['jerk'] < ra['jerk'] else '变差'} "
                  f"{abs(1-rb['jerk']/ra['jerk'])*100:.0f}%)")
        return
    for p in sys.argv[1:]:
        report(p)


if __name__ == "__main__":
    main()
