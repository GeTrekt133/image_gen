#!/usr/bin/env bash
# Идемпотентная закачка ФОТО-стека в ComfyUI/models с замером скорости по каждому файлу.
# Полный набор: image-базы (Z-Image bf16 / FLUX.2 Q8-GGUF / Qwen-2512 bf16 / SDXL) + Qwen-Edit-2511 +
# ControlNets (Qwen InstantX Union, FLUX.2 Fun, Z-Image Fun→model_patches) + консистентность
# (IP-Adapter, InstantID, PuLID, upscaler) + финиш-стек (face_yolov8m, 4x-UltraSharp, Skin-LoRA).
# Видео-ветка (Wan) — отдельно, здесь пропущена.
# Точность (ВЫВЕРЕНО НА ГЛАЗ, 2026-07-08): Z-Image bf16; Qwen — ТОЛЬКО 2512 bf16; FLUX.2 — Q8_0 GGUF
# + bf16 Mistral (fp8 «пластиковит» кожу!); Edit-2511 fp8mixed (bf16 41GB не влезает); SDXL fp16.
# ВНИМАНИЕ: всё сразу НЕ помещается на 200GB диск — либо SDXL-ветка, либо Edit; скрипт качает
# идемпотентно, отсутствующее доносится за минуты (Xet ~0.5 GB/s).
set -uo pipefail
source /workspace/env.sh
M=/workspace/ComfyUI/models
LOG=/workspace/dl_speed.log
mkdir -p "$M"/{checkpoints,vae,diffusion_models,text_encoders,controlnet,instantid,pulid,ipadapter,clip_vision,upscale_models,loras,model_patches,insightface/models}
: > "$LOG"

hr(){ printf '%s\n' "----------------------------------------------------------------"; }
record(){ local name="$1" f="$2" sec="$3"
  local bytes; bytes=$(stat -c%s "$f" 2>/dev/null || echo 0)
  local hsize; hsize=$(numfmt --to=iec "$bytes" 2>/dev/null || echo "$bytes")
  local mbps="n/a"
  if [ "$sec" -gt 0 ] && [ "$bytes" -gt 0 ]; then mbps=$(awk "BEGIN{printf \"%.1f\", $bytes/1048576/$sec}"); fi
  printf "%-46s | %8s | %5ss | %8s MB/s\n" "$name" "$hsize" "$sec" "$mbps" | tee -a "$LOG"
}

# HF: скачать один файл в нужную папку ComfyUI (Xet)
dl_hf(){ local repo="$1" path="$2" dest="$3" name="$4" label="$5"; local out="$dest/$name"
  if [ -s "$out" ]; then echo "[skip] $label ($(numfmt --to=iec $(stat -c%s "$out")))"; record "$label (cached)" "$out" 0; return 0; fi
  echo "[dl ] $label  <-  $repo/$path"; local t0=$SECONDS
  hf download "$repo" "$path" --local-dir /workspace/hf_dl >/tmp/hf_$$.log 2>&1
  if [ $? -ne 0 ]; then echo "  ❌ FAIL ($label): $(tail -2 /tmp/hf_$$.log)"; return 1; fi
  mkdir -p "$dest"; mv -f "/workspace/hf_dl/$path" "$out"; record "$label" "$out" "$((SECONDS - t0))"; }

# Civitai: aria2c, токен в query (БЕЗ Authorization — иначе 403 на signed R2 URL)
dl_civitai(){ local url="$1" dest="$2" name="$3" label="$4"; local out="$dest/$name"
  if [ -s "$out" ]; then echo "[skip] $label ($(numfmt --to=iec $(stat -c%s "$out")))"; record "$label (cached)" "$out" 0; return 0; fi
  echo "[dl ] $label  (civitai)"; local t0=$SECONDS
  aria2c -x8 -s8 -c --summary-interval=0 --console-log-level=warn \
    -d "$dest" -o "$name" "${url}?token=${CIVITAI_TOKEN}" 2>&1 | tail -2
  if [ ! -s "$out" ]; then echo "  ❌ FAIL ($label)"; return 1; fi
  record "$label" "$out" "$((SECONDS - t0))"; }

civitai_url(){ curl -s -H "Authorization: Bearer $CIVITAI_TOKEN" "https://civitai.com/api/v1/models/$1" | jq -r '.modelVersions[0].files[0].downloadUrl'; }

echo "==== Phase 4 downloads $(date -u +%H:%M:%SZ) ====" | tee -a "$LOG"; hr | tee -a "$LOG"

# ============ 1) SDXL-ветка (Civitai) ============
dl_civitai "$(civitai_url 575395)" "$M/checkpoints" "bigLust_v16.safetensors"   "Big Lust v1.6 (SDXL-NSFW)"
dl_civitai "$(civitai_url 133005)" "$M/checkpoints" "juggernautXL.safetensors"  "Juggernaut XL (SDXL realism)"
dl_civitai "$(civitai_url 139562)" "$M/checkpoints" "realvisxl_v5.safetensors"  "RealVisXL V5 (SDXL realism)"
hr | tee -a "$LOG"

# ============ 2) Z-Image Turbo (BF16 — flagship, fits) ============
dl_hf "Comfy-Org/z_image_turbo" "split_files/vae/ae.safetensors"                              "$M/vae"              "z_image_ae.safetensors"        "Z-Image VAE"
dl_hf "Comfy-Org/z_image_turbo" "split_files/text_encoders/qwen_3_4b.safetensors"             "$M/text_encoders"    "qwen_3_4b.safetensors"         "Z-Image text-enc qwen_3_4b (bf16)"
dl_hf "Comfy-Org/z_image_turbo" "split_files/diffusion_models/z_image_turbo_bf16.safetensors" "$M/diffusion_models" "z_image_turbo_bf16.safetensors" "Z-Image diffusion (bf16)"
hr | tee -a "$LOG"

# ============ 3) FLUX.2 Dev (Q8_0 GGUF ≈ bf16 + bf16 Mistral — проверено, fp8 глушит деталь) ============
# ВАЖНО: fp8mixed давал «пластиковый» вид; Q8 GGUF (city96) перцептивно = bf16 и на 30% меньше bf16.
# Требуется нода city96/ComfyUI-GGUF (loader UnetLoaderGGUF -> models/unet/).
dl_hf "Comfy-Org/flux2-dev" "split_files/vae/flux2-vae.safetensors"                            "$M/vae"              "flux2-vae.safetensors"                  "FLUX.2 VAE"
dl_hf "Comfy-Org/flux2-dev" "split_files/text_encoders/mistral_3_small_flux2_bf16.safetensors" "$M/text_encoders"   "mistral_3_small_flux2_bf16.safetensors" "FLUX.2 text-enc Mistral (bf16)"
dl_hf "city96/FLUX.2-dev-gguf" "flux2-dev-Q8_0.gguf"                                           "$M/unet"             "flux2-dev-Q8_0.gguf"                    "FLUX.2 diffusion Q8_0 GGUF"
hr | tee -a "$LOG"

# ============ 4) Qwen-Image-2512 (BF16, флагман) + Qwen-Image-Edit-2511 ============
# ВАЖНО: именно 2512 («Max» в API), не базовая qwen_image — базовая заметно мягче/хуже.
# Рендер: ModelSamplingAuraFlow shift 3.1, 50 шагов cfg 4, 1328x1328 натив + анти-«пластик» негатив (см. wf_qwen.json).
dl_hf "Comfy-Org/Qwen-Image_ComfyUI"      "split_files/diffusion_models/qwen_image_2512_bf16.safetensors"  "$M/diffusion_models" "qwen_image_2512_bf16.safetensors"  "Qwen-Image-2512 diffusion (bf16)"
dl_hf "Comfy-Org/Qwen-Image_ComfyUI"      "split_files/text_encoders/qwen_2.5_vl_7b.safetensors"           "$M/text_encoders"    "qwen_2.5_vl_7b.safetensors"        "Qwen 2.5-VL text-enc (bf16)"
dl_hf "Comfy-Org/Qwen-Image_ComfyUI"      "split_files/vae/qwen_image_vae.safetensors"                     "$M/vae"              "qwen_image_vae.safetensors"        "Qwen-Image VAE"
dl_hf "Comfy-Org/Qwen-Image-Edit_ComfyUI" "split_files/diffusion_models/qwen_image_edit_2511_fp8mixed.safetensors" "$M/diffusion_models" "qwen_image_edit_2511_fp8mixed.safetensors" "Qwen-Image-Edit-2511 (fp8mixed: эдиты/стиль-перенос, bf16=41GB не влезает)"
dl_hf "lightx2v/Qwen-Image-Lightning"     "Qwen-Image-Lightning-4steps-V1.0.safetensors"                   "$M/loras"            "Qwen-Image-Lightning-4steps-V1.0.safetensors" "Qwen Lightning 4-step LoRA"
hr | tee -a "$LOG"

# ============ 5) ControlNets по базам (все Union, поза через DWPose, strength 0.8-1.0) ============
# Qwen: нативный ControlNetLoader -> controlnet/
dl_hf "InstantX/Qwen-Image-ControlNet-Union" "diffusion_pytorch_model.safetensors" "$M/controlnet" "qwen_image_controlnet_union.safetensors" "Qwen ControlNet Union (InstantX)"
# FLUX.2: кастом-нода bryanmcguire/comfyui-flux2fun-controlnet
dl_hf "alibaba-pai/FLUX.2-dev-Fun-Controlnet-Union" "FLUX.2-dev-Fun-Controlnet-Union.safetensors" "$M/controlnet" "flux2_fun_controlnet_union.safetensors" "FLUX.2 Fun ControlNet Union (alibaba-pai)"
# Z-Image: ДРУГОЙ механизм — ModelPatchLoader -> model_patches/ (НЕ controlnet/!). Версия 2.1.
dl_hf "alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.1" "Z-Image-Turbo-Fun-Controlnet-Union-2.1.safetensors" "$M/model_patches" "zimage_fun_controlnet_union_2.1.safetensors" "Z-Image Fun ControlNet Union 2.1 (model_patches)"
hr | tee -a "$LOG"

# ============ 6) Консистентность: IP-Adapter / InstantID / PuLID / upscaler ============
dl_hf "h94/IP-Adapter" "sdxl_models/ip-adapter-plus-face_sdxl_vit-h.safetensors" "$M/ipadapter"  "ip-adapter-plus-face_sdxl_vit-h.safetensors" "IP-Adapter plus-face SDXL"
dl_hf "h94/IP-Adapter" "sdxl_models/ip-adapter-plus_sdxl_vit-h.safetensors"      "$M/ipadapter"  "ip-adapter-plus_sdxl_vit-h.safetensors"      "IP-Adapter plus SDXL"
dl_hf "h94/IP-Adapter" "models/image_encoder/model.safetensors"                  "$M/clip_vision" "CLIP-ViT-H-14-laion2B.safetensors"          "CLIP-vision ViT-H (IP-Adapter)"
dl_hf "InstantX/InstantID" "ip-adapter.bin"                                      "$M/instantid" "instantid_ip-adapter.bin"                "InstantID adapter"
dl_hf "InstantX/InstantID" "ControlNetModel/diffusion_pytorch_model.safetensors" "$M/controlnet" "instantid_controlnet.safetensors"       "InstantID ControlNet"
dl_hf "guozinan/PuLID" "pulid_flux_v0.9.1.safetensors" "$M/pulid" "pulid_flux_v0.9.1.safetensors" "PuLID-Flux v0.9.1"
dl_hf "Kim2091/UltraSharp" "4x-UltraSharp.pth" "$M/upscale_models" "4x-UltraSharp.pth" "4x-UltraSharp upscaler" \
  || dl_hf "lokCX/4x-Ultrasharp" "4x-UltraSharp.pth" "$M/upscale_models" "4x-UltraSharp.pth" "4x-UltraSharp upscaler (alt)"
hr | tee -a "$LOG"

# ============ 7) Финиш-стек (FaceDetailer + 2K + Skin-LoRA) — wf_finish_zimage.json ============
# Детектор лиц для FaceDetailer (нужна нода ltdrdata/ComfyUI-Impact-Subpack)
mkdir -p "$M/ultralytics/bbox"
dl_hf "Bingsu/adetailer" "face_yolov8m.pt" "$M/ultralytics/bbox" "face_yolov8m.pt" "face_yolov8m (FaceDetailer bbox)"
# Skin/Detail-LoRA под Z-Image Turbo (Civitai 2395852 «Radiant Realism Pro», версия 2812128).
# ГОТЧА: aria2c спотыкается на B2-редиректе версионных URL — качаем curl'ом.
SKIN_OUT="$M/loras/zimage_radiant_realism_v2.safetensors"
if [ -s "$SKIN_OUT" ]; then echo "[skip] Skin-LoRA Radiant Realism v2"; record "Skin-LoRA (cached)" "$SKIN_OUT" 0
else
  t0=$SECONDS
  curl -sL "https://civitai.com/api/download/models/2812128?token=${CIVITAI_TOKEN}" -o "$SKIN_OUT" --max-time 600
  [ -s "$SKIN_OUT" ] && record "Skin-LoRA Radiant Realism v2" "$SKIN_OUT" "$((SECONDS-t0))" || echo "  ❌ FAIL (Skin-LoRA)"
fi
hr | tee -a "$LOG"

echo "==== Phase 4 done $(date -u +%H:%M:%SZ) ====" | tee -a "$LOG"
echo; echo "=== ИТОГ по скорости ==="; column -t -s'|' "$LOG" 2>/dev/null || cat "$LOG"
rm -rf /workspace/hf_dl 2>/dev/null
echo "DOWNLOAD_DONE"
