#!/usr/bin/env bash
# HiDream-O1-Image headless setup (pod). Clones the OFFICIAL repo + pinned deps.
# Run from /workspace. Assumes venv /venv/main active (source /workspace/env.sh).
set -euo pipefail
cd "$(dirname "$0")"
PY="${PY:-/venv/main/bin/python}"

# 1. clone official repo — ⚠️ Dev-2604 needs the DEV branch (README model table):
#    main's inference.py is for the older Dev/full (bf16, noise_clip_std 2.5) and
#    produces visibly LESS DETAILED images with the 2604 checkpoint (verified A/B).
#    dev branch = torch_dtype float32 + noise_scale 8.0/clip_std 8.0 defaults.
if [ ! -d HiDream-O1-Image ]; then
  git clone --branch dev https://github.com/HiDream-ai/HiDream-O1-Image.git
else
  git -C HiDream-O1-Image fetch origin dev && git -C HiDream-O1-Image checkout dev
fi

# 2. deps — repo pins transformers==4.57.1 and needs torch>=2.10 (2.9.x is broken).
#    NOTE: this may upgrade torch on the pod — check the base image already has torch>=2.10.
"$PY" -m pip install -r HiDream-O1-Image/requirements.txt
# Blackwell (RTX 6000 / 5090): keep numpy/scipy sane or things won't import.
"$PY" -m pip install "numpy==2.4.6" "scipy>=1.13" || true

# 3. flash-attn — OPTIONAL and largely moot on the dev branch: dev runs the model in
#    fp32 (flash-attn kernels are fp16/bf16-only) and the SDPA fallback path does
#    2.37 it/s @2048² on RTX 6000 Blackwell — don't burn an hour compiling it.
#    ⚠️ dev branch hard-imports flash_attn unless FA_VERSION=auto (env). We patch the
#    pipeline flag off AND always export FA_VERSION=auto when running (see runner).
if ! "$PY" -c "import flash_attn" 2>/dev/null; then
  echo "[setup] flash-attn absent → patching use_flash_attn=False (SDPA path, quality-identical)"
  sed -i 's/"use_flash_attn": *True/"use_flash_attn": False/g' HiDream-O1-Image/models/pipeline.py
  grep -n '"use_flash_attn"' HiDream-O1-Image/models/pipeline.py || true
fi

mkdir -p models results
echo "[setup] done. next: bash dl_hidream_o1.sh"
