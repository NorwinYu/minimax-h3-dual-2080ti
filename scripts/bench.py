#!/usr/bin/env python3
"""Submit a workflow, sample BOTH GPUs throughout, report time + utilization.

Usage: bench.py <workflow.json> [label]
Physical GPU 0 = desktop card (comfy cuda:1, hosts the text encoder)
Physical GPU 1 = compute card (comfy cuda:0, hosts the DiT)
"""
import json, subprocess, sys, time, urllib.request, re, os

API = "http://127.0.0.1:8188"
LOG = os.environ.get("H3_COMFYUI_LOG", os.path.expanduser("~/.h3-2080ti/comfyui.log"))

def gpus():
    out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu",
                          "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
    r = []
    for line in out.strip().splitlines():
        i, m, u = [x.strip() for x in line.split(",")]
        r.append((int(i), int(m), int(u)))
    return r

def submit(path):
    wf = json.load(open(path))
    req = urllib.request.Request(f"{API}/prompt",
        data=json.dumps({"prompt": wf}).encode(),
        headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=90))["prompt_id"]

def logsize():
    try: return os.path.getsize(LOG)
    except OSError: return 0

def wait(pid, label):
    samples = []
    t0 = time.time()
    while True:
        samples.append((time.time() - t0, gpus()))
        try:
            h = json.load(urllib.request.urlopen(f"{API}/history/{pid}", timeout=10))
        except Exception:
            h = {}
        if h:
            e = list(h.values())[0]
            st = e.get("status", {})
            if st.get("completed"):
                return time.time() - t0, samples, e, st.get("status_str")
        if time.time() - t0 > 3600:
            return time.time() - t0, samples, {}, "TIMEOUT"
        time.sleep(2)

def steps_from_log(mark):
    """Per-step timings emitted by the Star7 chunk node."""
    try:
        txt = open(LOG, "rb").read()[mark:].decode("utf-8", "ignore")
    except Exception:
        return []
    txt = re.sub(r"\x1b\[[0-9;]*m", "", txt)
    return [float(x) for x in re.findall(r"step \d+/\d+ -> \S+\s+\|\s+([0-9.]+)s/it", txt)]

def main():
    path = sys.argv[1]
    label = sys.argv[2] if len(sys.argv) > 2 else os.path.basename(path)
    mark = logsize()
    pid = submit(path)
    print(f"  [{label}] 提交 {pid[:8]} …", flush=True)
    dt, samples, entry, st = wait(pid, label)
    steps = steps_from_log(mark)
    # utilization over the sampling window (skip first 10 s of model loading)
    win = [s for t, s in samples if t > 10] or [s for _, s in samples]
    def agg(idx):
        mem = [g[idx][1] for g in win if len(g) > idx]
        utl = [g[idx][2] for g in win if len(g) > idx]
        return (max(mem) if mem else 0, sum(utl)/len(utl) if utl else 0, max(utl) if utl else 0)
    m0, u0, p0 = agg(0)
    m1, u1, p1 = agg(1)
    total = re.findall(r"Prompt executed in ([0-9.]+) seconds",
                       open(LOG, "rb").read()[mark:].decode("utf-8", "ignore"))
    print(f"     状态={st}  采样步耗时={[round(s,1) for s in steps]}")
    print(f"     总耗时={dt:.1f}s" + (f"  (ComfyUI 计时 {total[-1]}s)" if total else ""))
    print(f"     物理GPU0 (桌面卡/编码器): 峰值显存 {m0} MiB, 平均利用率 {u0:.0f}%, 峰值 {p0}%")
    print(f"     物理GPU1 (计算卡/DiT)   : 峰值显存 {m1} MiB, 平均利用率 {u1:.0f}%, 峰值 {p1}%")
    outs = [v for o in entry.get("outputs", {}).values() for k, v in o.items() if k == "images"]
    if outs: print(f"     输出: {outs[-1]}")
    return dt

if __name__ == "__main__":
    main()
