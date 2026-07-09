#!/usr/bin/env bash
# LTX-2.3 video-branch model downloads (idempotent). Saves gemma fp8 under the
# name the workflow expects (gemma_3_12B_it_clean_fp8.safetensors).
set -uo pipefail
source /workspace/env.sh
M=/workspace/ComfyUI/models
mkdir -p "$M"/{checkpoints,text_encoders} "$M/loras/ltxv/ltx2"
dlf(){ local repo="$1" path="$2" dest="$3" name="$4"; local out="$dest/$name"
  [ -s "$out" ] && { echo "[skip] $name ($(numfmt --to=iec $(stat -c%s "$out")))"; return 0; }
  echo "[dl ] $name <- $repo/$path"; local t0=$SECONDS
  hf download "$repo" "$path" --local-dir /workspace/ltx_dl >/tmp/ltx_dl.log 2>&1 \
    && { mkdir -p "$dest"; mv -f "/workspace/ltx_dl/$path" "$out"; echo "  ok $(numfmt --to=iec $(stat -c%s "$out")) in $((SECONDS-t0))s"; } \
    || { echo "  FAIL $name"; tail -3 /tmp/ltx_dl.log; }
}
echo "== LTX-2.3 dev fp8 checkpoint (28G) =="
dlf Lightricks/LTX-2.3-fp8 ltx-2.3-22b-dev-fp8.safetensors "$M/checkpoints" ltx-2.3-22b-dev-fp8.safetensors
echo "== Gemma-3-12B fp8 text encoder (13G) -> clean_fp8 name =="
dlf Pavpif/ltx2-gemma3-text-encoder model_gemma_3_12B_it_fp8_e4m3fn.safetensors "$M/text_encoders" gemma_3_12B_it_clean_fp8.safetensors
echo "== distilled LoRA 384-1.1 =="
dlf Lightricks/LTX-2.3 ltx-2.3-22b-distilled-lora-384-1.1.safetensors "$M/loras/ltxv/ltx2" ltx-2.3-22b-distilled-lora-384-1.1.safetensors
echo "== union IC-LoRA ref0.5 =="
dlf Lightricks/LTX-2.3-22b-IC-LoRA-Union-Control ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors "$M/loras/ltxv/ltx2" ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors
rm -rf /workspace/ltx_dl 2>/dev/null
echo "LTX_MODELS_DONE"
