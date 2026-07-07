#!/usr/bin/env bash
# provision.sh — полная сборка стека с нуля (свежий под после Destroy).
# Воспроизводит Phase 1-3 из SETUP.md. Затем запусти startup.sh для моделей+сервера.
# Требует: /workspace/secrets.env с HF_TOKEN и CIVITAI_TOKEN. venv: /venv/main (torch cu13x).
set -uo pipefail
export DEBIAN_FRONTEND=noninteractive

echo "===== Phase 1: system + base ====="
apt-get update -qq
apt-get install -y -qq git git-lfs aria2 ffmpeg tmux jq build-essential
git lfs install --system
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || { echo "❌ нет GPU"; exit 1; }

# venv: используем существующий /venv/main (torch с CUDA под Blackwell уже собран).
# Если его нет на свежем образе — создать и поставить torch под свою CUDA отдельно.
if [ ! -x /venv/main/bin/python ]; then
  echo "⚠  /venv/main отсутствует — создаю чистый venv (torch поставь под свою CUDA вручную!)"
  python3 -m venv /venv/main
fi
source /venv/main/bin/activate
pip install -U huggingface_hub hf_transfer --root-user-action=ignore
[ -f /workspace/secrets.env ] && { set -a; source /workspace/secrets.env; set +a; huggingface-cli login --token "$HF_TOKEN"; } || echo "⚠  нет secrets.env"

echo "===== Phase 2: ComfyUI + custom nodes ====="
cd /workspace
[ -d ComfyUI/.git ] || git clone --depth 1 https://github.com/comfyanonymous/ComfyUI.git
cd ComfyUI && pip install -q -r requirements.txt --root-user-action=ignore
cd custom_nodes
for url in \
  https://github.com/ltdrdata/ComfyUI-Manager \
  https://github.com/cubiq/ComfyUI_InstantID \
  https://github.com/balazik/ComfyUI-PuLID-Flux \
  https://github.com/cubiq/ComfyUI_IPAdapter_plus \
  https://github.com/Fannovel16/comfyui_controlnet_aux \
  https://github.com/ltdrdata/ComfyUI-Impact-Pack \
  https://github.com/ssitu/ComfyUI_UltimateSDUpscale ; do
  d=$(basename "$url"); [ -d "$d/.git" ] || git clone --depth 1 --recurse-submodules "$url" "$d"
  [ -f "$d/requirements.txt" ] && pip install -q -r "$d/requirements.txt" --root-user-action=ignore
done

echo "===== Phase 3: trainers ====="
cd /workspace
[ -d ai-toolkit/.git ] || git clone --depth 1 --recurse-submodules https://github.com/ostris/ai-toolkit.git
[ -d kohya_ss/.git ]  || git clone --depth 1 --recurse-submodules https://github.com/bmaltais/kohya_ss.git
pip install -q -r ai-toolkit/requirements.txt --root-user-action=ignore || true

# Разрешение numpy-конфликта: ComfyUI/opencv хотят numpy>=2, ai-toolkit тянет scipy 1.12 (numpy<2).
# Фиксируем связку, рабочую для ComfyUI: numpy 2.4.6 + scipy>=1.13.
pip install -q "numpy==2.4.6" "scipy>=1.13" --root-user-action=ignore

echo "✅ provision завершён. Дальше:  bash /workspace/startup.sh"
