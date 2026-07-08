#!/usr/bin/env bash
set -uo pipefail
source /workspace/env.sh
M=/workspace/ComfyUI/models
mkdir -p "$M"/{checkpoints,latent_upscale_models,text_encoders} "$M/loras/ltxv/ltx2"
dlf(){ local repo="$1" path="$2" dest="$3" name="$4"; local out="$dest/$name"
  [ -s "$out" ] && { echo "[skip] $name"; return 0; }
  echo "[dl ] $name <- $repo/$path"
  hf download "$repo" "$path" --local-dir /workspace/ltx_dl >/tmp/ltx_$$.log 2>&1 \
    && { mkdir -p "$dest"; mv -f "/workspace/ltx_dl/$path" "$out"; echo "  ok $(numfmt --to=iec $(stat -c%s "$out"))"; } \
    || { echo "  ❌ FAIL $name"; tail -3 /tmp/ltx_$$.log; }
}
echo "== checkpoints (fp8: Pro=dev, Fast=distilled) =="
dlf Lightricks/LTX-2.3-fp8 ltx-2.3-22b-dev-fp8.safetensors       "$M/checkpoints" ltx-2.3-22b-dev-fp8.safetensors
dlf Lightricks/LTX-2.3-fp8 ltx-2.3-22b-distilled-fp8.safetensors "$M/checkpoints" ltx-2.3-22b-distilled-fp8.safetensors
echo "== distilled LoRA + upscalers =="
dlf Lightricks/LTX-2.3 ltx-2.3-22b-distilled-lora-384-1.1.safetensors "$M/loras/ltxv/ltx2" ltx-2.3-22b-distilled-lora-384-1.1.safetensors
dlf Lightricks/LTX-2.3 ltx-2.3-spatial-upscaler-x2-1.1.safetensors    "$M/latent_upscale_models" ltx-2.3-spatial-upscaler-x2-1.1.safetensors
dlf Lightricks/LTX-2.3 ltx-2.3-temporal-upscaler-x2-1.0.safetensors   "$M/latent_upscale_models" ltx-2.3-temporal-upscaler-x2-1.0.safetensors
echo "== IC-LoRA (motion-track + union) + pose (19b) =="
dlf Lightricks/LTX-2.3-22b-IC-LoRA-Motion-Track-Control ltx-2.3-22b-ic-lora-motion-track-control-ref0.5.safetensors "$M/loras/ltxv/ltx2" ltx-2.3-22b-ic-lora-motion-track-control-ref0.5.safetensors
dlf Lightricks/LTX-2.3-22b-IC-LoRA-Union-Control        ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors        "$M/loras/ltxv/ltx2" ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors
dlf Lightricks/LTX-2-19b-IC-LoRA-Pose-Control           ltx-2-19b-ic-lora-pose-control.safetensors                  "$M/loras/ltxv/ltx2" ltx-2-19b-ic-lora-pose-control.safetensors
echo "== Gemma-3-12b text encoder (folder, per README) =="
if [ ! -s "$M/text_encoders/gemma-3-12b-it-qat-q4_0-unquantized/model.safetensors.index.json" ] && [ ! -d "$M/text_encoders/gemma-3-12b-it-qat-q4_0-unquantized" ]; then
  hf download google/gemma-3-12b-it-qat-q4_0-unquantized --local-dir "$M/text_encoders/gemma-3-12b-it-qat-q4_0-unquantized" >/tmp/gemma.log 2>&1 && echo "  gemma ok" || { echo "  ❌ gemma FAIL"; tail -3 /tmp/gemma.log; }
else echo "[skip] gemma folder"; fi
rm -rf /workspace/ltx_dl 2>/dev/null
echo "LTX_DL_DONE"
