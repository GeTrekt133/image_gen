#!/usr/bin/env bash
# startup.sh — быстрый старт на УЖЕ собранном поде (после Vast Stop→Start).
# Идемпотентно докачивает модели и поднимает ComfyUI в tmux 'comfy'.
# Если под свежий (после Destroy) — сначала прогони provision.sh.
set -uo pipefail
cd /workspace
source /workspace/env.sh

echo "== [startup] докачка моделей (идемпотентно) =="
bash /workspace/download_models.sh

echo "== [startup] запуск ComfyUI в tmux 'comfy' =="
tmux kill-session -t comfy 2>/dev/null
tmux new -s comfy -d 'source /workspace/env.sh; cd /workspace/ComfyUI && python main.py --listen 0.0.0.0 --port 10100 > /workspace/comfy.log 2>&1'

echo -n "== [startup] жду :10100 "
for i in $(seq 1 60); do
  if curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:10100/ 2>/dev/null | grep -q 200; then
    echo "→ ✅ ComfyUI поднят (http://0.0.0.0:10100)"; exit 0
  fi
  echo -n "."; sleep 3
done
echo "→ ❌ не поднялся, смотри /workspace/comfy.log"; exit 1
