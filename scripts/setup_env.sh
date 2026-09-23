#!/usr/bin/env bash
# ============================================================================
#  ComfyUI environment install — sudo-free path
#
#  Why this is not the plain `python3 -m venv` + `pip install torch` from the plan:
#    * python3-venv/ensurepip are absent  -> venv created with --without-pip,
#      pip 26.2.1 unpacked straight from the wheel (sha256 verified)
#    * download.pytorch.org measured 21 KB/s -> torch comes from the aliyun
#      cu130 mirror at ~3.6 MB/s; PyPI deps come from TUNA
#  Version choice: torch 2.9.1 + torchvision 0.24.1 + torchaudio 2.9.1 (cu130,
#  cp312) — the combination the RTX 2080 Ti field handbook validated, and the
#  one whose comfy-kitchen hardware dequantisation path is the source of the
#  reported speed (torchaudio stops at 2.11, so the newest torch cannot be used).
# ============================================================================
set -uo pipefail
cd "$HOME/ComfyUI" || exit 1
LOG="$HOME/.h3-2080ti/setup.log"
exec >> "$LOG" 2>&1
echo "================ setup_env.sh started $(date '+%F %T') ================"

UA="Mozilla/5.0 (X11; Linux x86_64) Chrome/126 Safari/537.36"
WHEELS="$HOME/wheels"
MIRROR="https://mirrors.aliyun.com/pytorch-wheels/cu130"
TUNA="https://pypi.tuna.tsinghua.edu.cn/simple"
mkdir -p "$WHEELS"

echo "--- 1) 下载 cu130 wheel（aliyun，含续传）---"
for f in \
  "torch-2.9.1%2Bcu130-cp312-cp312-manylinux_2_28_x86_64.whl" \
  "torchvision-0.24.1%2Bcu130-cp312-cp312-manylinux_2_28_x86_64.whl" \
  "torchaudio-2.9.1%2Bcu130-cp312-cp312-manylinux_2_28_x86_64.whl" ; do
  out="$WHEELS/$(python3 -c "import urllib.parse,sys;print(urllib.parse.unquote(sys.argv[1]))" "$f")"
  echo "  -> $(basename "$out")"
  curl -L -C - -A "$UA" --retry 8 --retry-delay 5 --retry-all-errors \
       --max-time 3600 -o "$out" "$MIRROR/$f"
  echo "     大小: $(stat -c%s "$out" 2>/dev/null) 字节"
done

echo "--- 2) 安装 torch 三元组（依赖走清华 PyPI）---"
.venv/bin/python -m pip install --no-cache-dir \
  "$WHEELS"/torch-2.9.1+cu130-cp312-cp312-manylinux_2_28_x86_64.whl \
  "$WHEELS"/torchvision-0.24.1+cu130-cp312-cp312-manylinux_2_28_x86_64.whl \
  "$WHEELS"/torchaudio-2.9.1+cu130-cp312-cp312-manylinux_2_28_x86_64.whl \
  -i "$TUNA" 2>&1 | tail -25

echo "--- 3) 安装 ComfyUI requirements（清华 PyPI）---"
.venv/bin/python -m pip install --no-cache-dir -r requirements.txt -i "$TUNA" 2>&1 | tail -30

echo "--- 4) 门禁检查 ---"
.venv/bin/python - <<'PY'
import importlib, sys
print("python      :", sys.version.split()[0])
try:
    import torch
    print("torch       :", torch.__version__, "| cuda build:", torch.version.cuda)
    print("arch_list   :", torch.cuda.get_arch_list())
    print("sm_75 在列表:", "sm_75" in torch.cuda.get_arch_list())
    print("cuda 可用   :", torch.cuda.is_available())
    print("显卡数量    :", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        print(f"  gpu{i}      :", torch.cuda.get_device_name(i), "cap", torch.cuda.get_device_capability(i))
    try:
        print("P2P 0->1    :", torch.cuda.can_device_access_peer(0, 1))
    except Exception as e:
        print("P2P 检查失败:", e)
except Exception as e:
    print("!! torch 导入/初始化失败:", type(e).__name__, e)
try:
    import comfy_kitchen as ck
    print("comfy_kitchen:", getattr(ck, "__version__", "?"), ck.__file__)
except Exception as e:
    print("!! comfy_kitchen 导入失败:", type(e).__name__, e)
PY

echo "================ setup_env.sh finished $(date '+%F %T') ================"
