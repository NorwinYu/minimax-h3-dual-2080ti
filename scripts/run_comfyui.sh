#!/usr/bin/env bash
# ============================================================================
#  Start ComfyUI for MiniMax H3 on the NVLink 2080 Ti pair.
#
#  GPU split (see plan §4):
#    GPU 1 = DiT + sampling   (only ~4 MiB of desktop lives there)
#    GPU 0 = text encoder + projection + both VAEs (pinned via ClipProj/MultiGPU)
#
#  No --lowvram: the 12.5 GB W4A8 DiT fits in 22 GB, and forcing offload costs
#  multiples of runtime. --reserve-vram/--vram-headroom guard the small
#  remaining headroom; --disable-pinned-memory avoids the known host-buffer
#  read/OOM failure path.
# ============================================================================
set -uo pipefail
cd "$HOME/ComfyUI" || exit 1
LOG="$HOME/.h3-2080ti/comfyui.log"
mkdir -p "$HOME/.h3-2080ti"

if pgrep -f "main.py --listen 127.0.0.1 --port 8188" >/dev/null; then
  echo "ComfyUI 已在运行 (pid $(pgrep -f 'main.py --listen 127.0.0.1 --port 8188' | head -1))"
  exit 0
fi

# CUDA_VISIBLE_DEVICES=1,0  -- NOT --cuda-device 1.
# --cuda-device 1 restricts ComfyUI to ONE device, which silently makes every
# "cuda:0" inside custom nodes that same card, so GPU 0 sits idle for the whole
# run. Exposing both (GPU 1 first) gives ComfyUI two devices where index 0 is the
# clean card (DiT compute) and index 1 is the desktop card (encoder + VAEs pinned).
export CUDA_VISIBLE_DEVICES=1,0
# ComfyUI-MultiGPU's p2p_registry does ctypes.CDLL("libcudart.so") — the unversioned
# name. The venv only ships libcudart.so.13, so with two GPUs visible its P2P path
# fails and every model load dies. The symlink + LD_LIBRARY_PATH below fix that.
CUDALIB="$HOME/ComfyUI/.venv/lib/python3.12/site-packages/nvidia/cu13/lib"
[ -e "$CUDALIB/libcudart.so" ] || ln -sf libcudart.so.13 "$CUDALIB/libcudart.so"
export LD_LIBRARY_PATH="$CUDALIB:${LD_LIBRARY_PATH:-}"
nohup .venv/bin/python main.py \
  --listen 127.0.0.1 --port 8188 \
  --reserve-vram 1.5 --vram-headroom 0.5 \
  --disable-pinned-memory \
  --fast-disk \
  > "$LOG" 2>&1 &

echo "ComfyUI 启动中 (pid $!)，日志: $LOG"
for i in $(seq 1 60); do
  if curl -s -m 3 -o /dev/null "http://127.0.0.1:8188/system_stats"; then
    echo "就绪（${i}s）: http://127.0.0.1:8188"
    exit 0
  fi
  sleep 1
done
echo "60s 内未就绪，请查看 $LOG"
exit 1
