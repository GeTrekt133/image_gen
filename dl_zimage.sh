source /workspace/env.sh
M=/workspace/ComfyUI/models; LOG=/workspace/dl_zimage_speed.log; : > "$LOG"
dl(){ repo="$1"; path="$2"; dest="$3"; name="$4"; out="$dest/$name"
  [ -s "$out" ] && { echo "[skip] $name"; return; }
  t0=$SECONDS; hf download "$repo" "$path" --local-dir /workspace/zdl >/dev/null 2>&1
  mv -f "/workspace/zdl/$path" "$out"; sec=$((SECONDS-t0))
  b=$(stat -c%s "$out"); mbps=$(awk "BEGIN{printf \"%.1f\", $b/1048576/($sec>0?$sec:1)}")
  printf "%-34s | %8s | %4ss | %7s MB/s\n" "$name" "$(numfmt --to=iec $b)" "$sec" "$mbps" | tee -a "$LOG"; }
dl Comfy-Org/z_image_turbo split_files/vae/ae.safetensors                          "$M/vae"              z_image_ae.safetensors
dl Comfy-Org/z_image_turbo split_files/text_encoders/qwen_3_4b.safetensors         "$M/text_encoders"   qwen_3_4b.safetensors
dl Comfy-Org/z_image_turbo split_files/diffusion_models/z_image_turbo_bf16.safetensors "$M/diffusion_models" z_image_turbo_bf16.safetensors
rm -rf /workspace/zdl; echo "==== Z-IMAGE DONE ====" | tee -a "$LOG"
