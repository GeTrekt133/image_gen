#!/usr/bin/env bash
# Закачка ровно того, что нужно для Z-Image finish+NSFW прогона.
set -uo pipefail
source /workspace/env.sh
M=/workspace/ComfyUI/models
mkdir -p "$M"/{vae,text_encoders,diffusion_models,loras,upscale_models,ultralytics/bbox}

dl_hf(){ local repo="$1" path="$2" dest="$3" name="$4"; local out="$dest/$name"
  if [ -s "$out" ]; then echo "[skip] $name"; return 0; fi
  echo "[dl ] $name  <-  $repo/$path"
  hf download "$repo" "$path" --local-dir /workspace/hf_dl >/tmp/hf_dl.log 2>&1 \
    && { mkdir -p "$dest"; mv -f "/workspace/hf_dl/$path" "$out"; echo "  ok $(numfmt --to=iec $(stat -c%s "$out"))"; } \
    || { echo "  ❌ FAIL $name"; tail -3 /tmp/hf_dl.log; return 1; }
}
civitai_url(){ curl -s -H "Authorization: Bearer $CIVITAI_TOKEN" "https://civitai.com/api/v1/models/$1" | jq -r '.modelVersions[0].files[0].downloadUrl'; }
dl_civitai_curl(){ local mid="$1" name="$2"; local out="$M/loras/$name"
  if [ -s "$out" ]; then echo "[skip] $name"; return 0; fi
  local url; url=$(civitai_url "$mid")
  echo "[dl ] $name (civitai $mid)"
  curl -sL "${url}?token=${CIVITAI_TOKEN}" -o "$out" --max-time 900 \
    && [ -s "$out" ] && echo "  ok $(numfmt --to=iec $(stat -c%s "$out"))" || echo "  ❌ FAIL $name"
}
# версионный URL напрямую (2812128 = Radiant Realism Pro v2), только curl (aria2 403 на B2)
dl_civitai_ver(){ local vid="$1" name="$2"; local out="$M/loras/$name"
  if [ -s "$out" ]; then echo "[skip] $name"; return 0; fi
  echo "[dl ] $name (civitai ver $vid)"
  curl -sL "https://civitai.com/api/download/models/${vid}?token=${CIVITAI_TOKEN}" -o "$out" --max-time 900 \
    && [ -s "$out" ] && echo "  ok $(numfmt --to=iec $(stat -c%s "$out"))" || echo "  ❌ FAIL $name"
}

echo "==== Z-Image finish DL $(date -u +%H:%M:%SZ) ===="
dl_hf Comfy-Org/z_image_turbo split_files/vae/ae.safetensors                              "$M/vae"              z_image_ae.safetensors
dl_hf Comfy-Org/z_image_turbo split_files/text_encoders/qwen_3_4b.safetensors             "$M/text_encoders"    qwen_3_4b.safetensors
dl_hf Comfy-Org/z_image_turbo split_files/diffusion_models/z_image_turbo_bf16.safetensors "$M/diffusion_models" z_image_turbo_bf16.safetensors
dl_civitai_curl 2279079 zimage_nsfw.safetensors
dl_civitai_ver  2812128 zimage_radiant_realism_v2.safetensors
dl_hf Bingsu/adetailer face_yolov8m.pt "$M/ultralytics/bbox" face_yolov8m.pt
dl_hf Kim2091/UltraSharp 4x-UltraSharp.pth "$M/upscale_models" 4x-UltraSharp.pth \
  || dl_hf lokCX/4x-Ultrasharp 4x-UltraSharp.pth "$M/upscale_models" 4x-UltraSharp.pth
rm -rf /workspace/hf_dl 2>/dev/null
echo "DL_DONE"
