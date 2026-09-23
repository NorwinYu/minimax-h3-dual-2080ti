#!/usr/bin/env python3
"""Pre-flight a ComfyUI API-format workflow against the live /object_info.

Checks, without submitting anything:
  * every class_type exists
  * every required input is supplied
  * no unknown inputs (autogrow prefix keys such as ref_image_0 are accepted)
  * every link points at an existing node and a valid output index
  * combo widget values are among the offered options

Usage: validate_wf.py <workflow.json> [--api http://127.0.0.1:8188]
"""
import json, sys, urllib.request

def main():
    path = sys.argv[1]
    api = "http://127.0.0.1:8188"
    if "--api" in sys.argv:
        api = sys.argv[sys.argv.index("--api") + 1]
    wf = json.load(open(path))
    wf = {k: v for k, v in wf.items() if not k.startswith("_")}
    oi = json.load(urllib.request.urlopen(f"{api}/object_info", timeout=90))

    # autogrow: collect every "<prefix><n>" key the node family accepts
    errs = []
    for nid, node in wf.items():
        ct = node["class_type"]
        x = oi.get(ct)
        if not x:
            errs.append(f"节点 {nid}: 未知 class_type {ct}")
            continue
        req = x["input"].get("required", {}) or {}
        opt = x["input"].get("optional", {}) or {}
        grow = set()
        for k, v in list(req.items()) + list(opt.items()):
            meta = v[1] if len(v) > 1 and isinstance(v[1], dict) else {}
            tmpl = meta.get("template") or {}
            if v[0] == "COMFY_AUTOGROW_V3" and tmpl.get("prefix"):
                grow.add((k, tmpl["prefix"], tmpl.get("max", 0)))
        provided = set(node["inputs"])
        for k in req:
            if k not in provided:
                errs.append(f"节点 {nid} ({ct}): 缺少必需输入 {k}")
        for k in provided:
            if k in req or k in opt:
                continue
            # autogrow slots arrive as dotted paths: <parent>.<prefix><n>
            if any(k.startswith(f"{parent}.{p}") and k[len(parent)+1+len(p):].isdigit()
                   for parent, p, _ in grow):
                continue
            errs.append(f"节点 {nid} ({ct}): 未知输入 {k}")

        for k, v in node["inputs"].items():
            if isinstance(v, list) and len(v) == 2 and isinstance(v[0], str):
                if v[0] not in wf:
                    errs.append(f"节点 {nid}.{k}: 引用不存在的节点 {v[0]}")
                    continue
                outs = oi[wf[v[0]]["class_type"]].get("output") or []
                if v[1] >= len(outs):
                    errs.append(f"节点 {nid}.{k}: 输出索引 {v[1]} 超范围 "
                                f"({wf[v[0]]['class_type']} 只有 {len(outs)} 个输出)")

        for k, v in node["inputs"].items():
            if k not in req or not isinstance(v, str):
                continue
            spec = req[k]
            if isinstance(spec[0], list):
                if v not in spec[0]:
                    errs.append(f"节点 {nid}.{k}: 值 {v!r} 不在选项中")
            elif spec[0] == "COMBO":
                opts = (spec[1] or {}).get("options")
                if isinstance(opts, list) and opts and isinstance(opts[0], str) and v not in opts:
                    errs.append(f"节点 {nid}.{k}: 值 {v!r} 不在 COMBO 选项中")

    print(f"预检 {path}: {len(wf)} 个节点")
    if errs:
        for e in errs:
            print("  ✗", e)
        sys.exit(1)
    print("  ✔ 全部通过（节点类存在、必需输入齐全、autogrow 键合法、连线与 combo 取值有效）")

if __name__ == "__main__":
    main()
