#!/usr/bin/env bash
# Идемпотентная закачка моделей в ComfyUI/models с замером скорости по каждому файлу.
# Скоуп по умолчанию: смоук-набор (Big Lust SDXL + FLUX.2 fp8). Расширяемо флагами.
set -uo pipefail
source /workspace/env.sh
M=/workspace/ComfyUI/models
LOG=/workspace/dl_speed.log
mkdir -p "$M"/{checkpoints,vae,diffusion_models,text_encoders,controlnet,instantid,pulid,ipadapter,clip_vision,upscale_models}
: > "$LOG"

hr(){ printf '%s\n' "----------------------------------------------------------------"; }
# лог одной строки замера: имя | размер | время | скорость
record(){ # name file seconds
  local name="$1" f="$2" sec="$3"
  local bytes; bytes=$(stat -c%s "$f" 2>/dev/null || echo 0)
  local hsize; hsize=$(numfmt --to=iec "$bytes" 2>/dev/null || echo "$bytes")
  local mbps="n/a"
  if [ "$sec" -gt 0 ] && [ "$bytes" -gt 0 ]; then
    mbps=$(awk "BEGIN{printf \"%.1f\", $bytes/1048576/$sec}")
  fi
  printf "%-42s | %8s | %5ss | %8s MB/s\n" "$name" "$hsize" "$sec" "$mbps" | tee -a "$LOG"
}

# --- HF: скачать один файл в нужную папку ComfyUI ---
dl_hf(){ # repo  path_in_repo  dest_dir  dest_name  label
  local repo="$1" path="$2" dest="$3" name="$4" label="$5"
  local out="$dest/$name"
  if [ -s "$out" ]; then echo "[skip] $label уже есть ($(numfmt --to=iec $(stat -c%s "$out")))"; record "$label (cached)" "$out" 0; return 0; fi
  echo "[dl ] $label  <-  $repo/$path"
  local t0=$SECONDS
  hf download "$repo" "$path" --local-dir /workspace/hf_dl >/tmp/hf_$$.log 2>&1
  local rc=$?
  if [ $rc -ne 0 ]; then echo "  ❌ FAIL ($label): $(tail -1 /tmp/hf_$$.log)"; return 1; fi
  mv -f "/workspace/hf_dl/$path" "$out"
  record "$label" "$out" "$((SECONDS - t0))"
}

# --- Civitai: aria2c с токеном ---
dl_civitai(){ # url  dest_dir  dest_name  label
  local url="$1" dest="$2" name="$3" label="$4"
  local out="$dest/$name"
  if [ -s "$out" ]; then echo "[skip] $label уже есть ($(numfmt --to=iec $(stat -c%s "$out")))"; record "$label (cached)" "$out" 0; return 0; fi
  echo "[dl ] $label  (civitai)"
  local t0=$SECONDS
  # токен в query; БЕЗ Authorization-заголовка (иначе он улетает на R2 и ломает signed-URL)
  aria2c -x8 -s8 -c --summary-interval=10 --console-log-level=warn \
    -d "$dest" -o "$name" "${url}?token=${CIVITAI_TOKEN}" 2>&1 | tail -3
  if [ ! -s "$out" ]; then echo "  ❌ FAIL ($label)"; return 1; fi
  record "$label" "$out" "$((SECONDS - t0))"
}

echo "==== Phase 4 downloads $(date -u +%H:%M:%SZ) ====" | tee -a "$LOG"
hr | tee -a "$LOG"

# 1) Big Lust (SDXL) — первым, разблокирует SDXL-смоук
BL_URL=$(curl -s -H "Authorization: Bearer $CIVITAI_TOKEN" "https://civitai.com/api/v1/models/575395" | jq -r '.modelVersions[0].files[0].downloadUrl')
dl_civitai "$BL_URL" "$M/checkpoints" "bigLust_v16.safetensors" "Big Lust v1.6 (SDXL)"
hr | tee -a "$LOG"

# 2) FLUX.2 fp8 essentials
dl_hf "Comfy-Org/flux2-dev" "split_files/vae/flux2-vae.safetensors"                              "$M/vae"              "flux2-vae.safetensors"                 "FLUX.2 VAE"
dl_hf "Comfy-Org/flux2-dev" "split_files/text_encoders/mistral_3_small_flux2_fp8.safetensors"    "$M/text_encoders"   "mistral_3_small_flux2_fp8.safetensors" "FLUX.2 text-enc Mistral fp8"
dl_hf "Comfy-Org/flux2-dev" "split_files/diffusion_models/flux2_dev_fp8mixed.safetensors"        "$M/diffusion_models" "flux2_dev_fp8mixed.safetensors"       "FLUX.2 diffusion fp8"
hr | tee -a "$LOG"

echo "==== Phase 4 done $(date -u +%H:%M:%SZ) ====" | tee -a "$LOG"
echo; echo "=== ИТОГ по скорости ==="; column -t -s'|' "$LOG" 2>/dev/null || cat "$LOG"
rm -rf /workspace/hf_dl 2>/dev/null
