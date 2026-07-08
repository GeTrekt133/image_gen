#!/usr/bin/env bash
# Загрузка моделей NSFW-пайплайна (unStable Revolution + finish + pose/depth + Qwen-Edit).
# Идемпотентно. См. PIPELINE_NSFW.md. Требует /workspace/secrets.env (HF_TOKEN, CIVITAI_TOKEN).
set -uo pipefail
source /workspace/env.sh
M=/workspace/ComfyUI/models
mkdir -p "$M"/{diffusion_models,text_encoders,vae,model_patches,loras,upscale_models,ultralytics/bbox}

dl_hf(){ local repo="$1" path="$2" dest="$3" name="$4"; local out="$dest/$name"
  [ -s "$out" ] && { echo "[skip] $name"; return 0; }
  echo "[dl ] $name <- $repo/$path"
  hf download "$repo" "$path" --local-dir /workspace/hf_dl >/tmp/hf_dl.log 2>&1 \
    && { mkdir -p "$dest"; mv -f "/workspace/hf_dl/$path" "$out"; echo "  ok $(numfmt --to=iec $(stat -c%s "$out"))"; } \
    || { echo "  ❌ FAIL $name"; tail -3 /tmp/hf_dl.log; }
}
# Civitai по versionId (уходит на B2 → только curl, БЕЗ Authorization-заголовка)
dl_civitai_ver(){ local vid="$1" dest="$2" name="$3"; local out="$dest/$name"
  [ -s "$out" ] && { echo "[skip] $name"; return 0; }
  echo "[dl ] $name (civitai ver $vid)"
  curl -sL "https://civitai.com/api/download/models/${vid}?token=${CIVITAI_TOKEN}" -o "$out" --max-time 3600 \
    && [ -s "$out" ] && echo "  ok $(numfmt --to=iec $(stat -c%s "$out"))" || echo "  ❌ FAIL $name"
}

echo "==== 1) unStable Revolution ZIT v3 fp16 (полный NSFW-чекпоинт, БАЗА) ===="
dl_civitai_ver 2852808 "$M/diffusion_models" unstable_revolution_zit_v3_fp16.safetensors

echo "==== 2) Z-Image текст-энкодер + VAE (нужны для unStable и финиша) ===="
dl_hf Comfy-Org/z_image_turbo split_files/text_encoders/qwen_3_4b.safetensors "$M/text_encoders" qwen_3_4b.safetensors
dl_hf Comfy-Org/z_image_turbo split_files/vae/ae.safetensors                  "$M/vae"           z_image_ae.safetensors

echo "==== 3) Финиш-стек (FaceDetailer bbox + upscaler) ===="
dl_hf Bingsu/adetailer face_yolov8m.pt "$M/ultralytics/bbox" face_yolov8m.pt
dl_hf Kim2091/UltraSharp 4x-UltraSharp.pth "$M/upscale_models" 4x-UltraSharp.pth \
  || dl_hf lokCX/4x-Ultrasharp 4x-UltraSharp.pth "$M/upscale_models" 4x-UltraSharp.pth

echo "==== 4) Pose/Depth control (Fun-CN Union 2.1 → model_patches!) ===="
dl_hf alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.1 Z-Image-Turbo-Fun-Controlnet-Union-2.1.safetensors \
      "$M/model_patches" zimage_fun_controlnet_union_2.1.safetensors
# DepthAnythingV2 (vitl) и DWPose догружаются автоматически нодой comfyui_controlnet_aux при первом запуске.

echo "==== 5) Qwen-Image-Edit-2511 (точечные NSFW-правки) ===="
dl_hf Comfy-Org/Qwen-Image-Edit_ComfyUI split_files/diffusion_models/qwen_image_edit_2511_fp8mixed.safetensors "$M/diffusion_models" qwen_image_edit_2511_fp8mixed.safetensors
dl_hf Comfy-Org/Qwen-Image_ComfyUI      split_files/text_encoders/qwen_2.5_vl_7b.safetensors                   "$M/text_encoders"    qwen_2.5_vl_7b.safetensors
dl_hf Comfy-Org/Qwen-Image_ComfyUI      split_files/vae/qwen_image_vae.safetensors                             "$M/vae"              qwen_image_vae.safetensors

# ==== (опц.) NSFW-LoRA для сравнения (обычно НЕ нужны — unStable самодостаточен) ====
# dl_civitai_ver 2565112 "$M/loras" zimage_nsfw.safetensors            # 2279079 Z-Image Turbo NSFW (rank-4)
# dl_civitai_ver 2502526 "$M/loras" zimage_nsfw_godpussy.safetensors   # 2222911 GodPussy (детейл вульвы)
# dl_civitai_ver 2486059 "$M/loras" zimage_nsfw_pussy.safetensors      # 2205140 z-image-pussy

rm -rf /workspace/hf_dl 2>/dev/null
echo "NSFW_PIPELINE_DL_DONE"
