#!/usr/bin/env bash
# HiDream-O1-Image headless setup (pod). Clones the OFFICIAL repo + pinned deps.
# Run from /workspace. Assumes venv /venv/main active (source /workspace/env.sh).
set -euo pipefail
cd "$(dirname "$0")"
PY="${PY:-/venv/main/bin/python}"

# 1. clone official repo (source of truth for inference.py + models/pipeline.py)
if [ ! -d HiDream-O1-Image ]; then
  git clone https://github.com/HiDream-ai/HiDream-O1-Image.git
fi

# 2. deps — repo pins transformers==4.57.1 and needs torch>=2.10 (2.9.x is broken).
#    NOTE: this may upgrade torch on the pod — check the base image already has torch>=2.10.
"$PY" -m pip install -r HiDream-O1-Image/requirements.txt
# Blackwell (RTX 6000 / 5090): keep numpy/scipy sane or things won't import.
"$PY" -m pip install "numpy==2.4.6" "scipy>=1.13" || true

# 3. flash-attn (highly recommended). If the build fails, patch pipeline.py to disable it.
if ! "$PY" -c "import flash_attn" 2>/dev/null; then
  echo "[setup] installing flash-attn…"
  if ! "$PY" -m pip install flash-attn --no-build-isolation; then
    echo "[setup] flash-attn install FAILED → patching models/pipeline.py line 341 use_flash_attn False"
    # robust: replace the flag wherever it appears (line number may drift across commits)
    sed -i 's/"use_flash_attn": *True/"use_flash_attn": False/g' HiDream-O1-Image/models/pipeline.py
    grep -n '"use_flash_attn"' HiDream-O1-Image/models/pipeline.py || true
  fi
fi

mkdir -p models results
echo "[setup] done. next: bash dl_hidream_o1.sh"
