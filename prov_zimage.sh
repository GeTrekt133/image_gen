#!/usr/bin/env bash
# Тримленная сборка под Z-Image finish-прогон: ComfyUI reqs + Impact-Pack + Impact-Subpack + numpy-пин.
set -uo pipefail
source /workspace/env.sh
cd /workspace/ComfyUI
echo "== pip: ComfyUI requirements =="
pip install -q -r requirements.txt --root-user-action=ignore
cd custom_nodes
for url in \
  https://github.com/ltdrdata/ComfyUI-Impact-Pack \
  https://github.com/ltdrdata/ComfyUI-Impact-Subpack ; do
  d=$(basename "$url")
  [ -d "$d/.git" ] || git clone --depth 1 "$url" "$d"
  [ -f "$d/requirements.txt" ] && pip install -q -r "$d/requirements.txt" --root-user-action=ignore
done
# Blackwell пин (иначе ComfyUI не стартует)
pip install -q "numpy==2.4.6" "scipy>=1.13" --root-user-action=ignore
echo "PROV_DONE"
