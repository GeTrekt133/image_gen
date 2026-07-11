#!/usr/bin/env bash
# Download HiDream-O1-Image-Dev-2604 weights (HF, no conversion needed).
# Needs HF_TOKEN in env (source /workspace/secrets.env). Uses HF Xet (fast).
set -euo pipefail
cd "$(dirname "$0")"
PY="${PY:-/venv/main/bin/python}"
"$PY" -m pip install -q "huggingface_hub[hf_xet]" >/dev/null 2>&1 || true
DL(){ "$PY" -m huggingface_hub download "$1" --local-dir "$2" ${HF_TOKEN:+--token "$HF_TOKEN"}; }
# fallback if the module entrypoint differs:
command -v hf >/dev/null 2>&1 && DL(){ hf download "$1" --local-dir "$2"; }

# the checkpoint we use (distilled, 28-step, model_type=dev)
DL HiDream-ai/HiDream-O1-Image-Dev-2604 models/hidream-o1-dev-2604

# optional — full model (50-step, higher quality, slower):
# DL HiDream-ai/HiDream-O1-Image models/hidream-o1-full
# optional — prompt refiner (heavy Gemma-4-31B; skip unless needed):
# DL HiDream-ai/Prompt-Refine models/prompt-refine

echo "[dl] done. sanity-check it's FULL weights (not a LoRA):"
du -sh models/hidream-o1-dev-2604 2>/dev/null || true
ls -la models/hidream-o1-dev-2604 | head
