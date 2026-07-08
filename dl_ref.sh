#!/usr/bin/env bash
set -uo pipefail; source /workspace/env.sh; M=/workspace/ComfyUI/models
dl(){ repo="$1"; path="$2"; dest="$3"; name="$4"; out="$dest/$name"
  [ -s "$out" ] && { echo "[skip] $name"; return; }
  echo "[dl] $name <- $repo/$path"
  hf download "$repo" "$path" --local-dir /workspace/rdl >/tmp/rdl.log 2>&1 \
    && { mkdir -p "$dest"; mv -f "/workspace/rdl/$path" "$out"; echo "  ok $(numfmt --to=iec $(stat -c%s "$out"))"; } \
    || { echo "  FAIL $name"; tail -2 /tmp/rdl.log; }
}
dl unsloth/LTX-2.3-GGUF "ltx-2.3-22b-dev-Q4_K_M.gguf"              "$M/unet"          "ltx-2-3-22b-dev-Q4_K_M.gguf"
dl unsloth/LTX-2.3-GGUF "vae/ltx-2.3-22b-dev_video_vae.safetensors" "$M/vae"          "LTX23_video_vae_bf16.safetensors"
dl unsloth/LTX-2.3-GGUF "vae/ltx-2.3-22b-dev_audio_vae.safetensors" "$M/vae"          "LTX23_audio_vae_bf16.safetensors"
dl onewayomni/ltx-2.3_text_projection_bf16.safetensors "ltx-2.3_text_projection_bf16.safetensors" "$M/text_encoders" "ltx-2.3_text_projection_bf16.safetensors"
dl Lightricks/LTX-2.3 "ltx-2.3-22b-distilled-lora-384.safetensors" "$M/loras/ltxv/ltx2" "ltx-2.3-22b-distilled-lora-384.safetensors"
# gemma fp4 (single file in inflatebot repo)
GF=$(curl -s "https://huggingface.co/api/models/inflatebot/LTX23-gemma-3-12b-it-orthogonal-reflection-bounded-ablation-v4-fp4_mixed" | python3 -c "import sys,json;print([s['rfilename'] for s in json.load(sys.stdin).get('siblings',[]) if s['rfilename'].endswith('.safetensors')][0])" 2>/dev/null)
echo "gemma fp4 file: $GF"
dl "inflatebot/LTX23-gemma-3-12b-it-orthogonal-reflection-bounded-ablation-v4-fp4_mixed" "$GF" "$M/text_encoders" "gemma_3_12B_it_fp4_mixed.safetensors"
# spatial upscaler 1.0 (workflow wants -1.0; copy our 1.1 as 1.0 fallback + try real 1.0)
[ -s "$M/latent_upscale_models/ltx-2.3-spatial-upscaler-x2-1.0.safetensors" ] || cp -f "$M/latent_upscale_models/ltx-2.3-spatial-upscaler-x2-1.1.safetensors" "$M/latent_upscale_models/ltx-2.3-spatial-upscaler-x2-1.0.safetensors" 2>/dev/null
rm -rf /workspace/rdl 2>/dev/null
echo "REF_DL_DONE"
