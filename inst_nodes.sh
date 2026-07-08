#!/usr/bin/env bash
set -uo pipefail
source /workspace/env.sh
cd /workspace/ComfyUI/custom_nodes
clone(){ d=$(basename "$1"); [ -d "$d/.git" ] || git clone --depth 1 "$1" "$d"; \
  [ -f "$d/requirements.txt" ] && pip install -q -r "$d/requirements.txt" --root-user-action=ignore 2>&1 | tail -1; echo "  $d ok"; }
clone https://github.com/rgthree/rgthree-comfy
clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite
clone https://github.com/kijai/ComfyUI-KJNodes
clone https://github.com/evanspearman/ComfyMath
clone https://github.com/pythongosssss/ComfyUI-Custom-Scripts
clone https://github.com/city96/ComfyUI-GGUF
clone https://github.com/yolain/ComfyUI-Easy-Use
clone https://github.com/Smirnov75/ComfyUI-mxToolkit
# update ComfyUI-LTXVideo to latest (need FinalFrameSelector/LTX2SamplingPreviewOverride/ChunkFeedForward)
cd ComfyUI-LTXVideo && git fetch --depth 1 origin 2>/dev/null; git pull 2>&1 | tail -1; cd ..
# re-apply kornia pad shim if pull reverted it
python - <<'PY'
p="/workspace/ComfyUI/custom_nodes/ComfyUI-LTXVideo/pyramid_blending.py"
s=open(p).read()
if "pad = F.pad" not in s:
    s=s.replace("    is_powerof_two,\n    pad,\n)","    is_powerof_two,\n)\npad = __import__('torch').nn.functional.pad")
    open(p,"w").write(s); print("re-applied pad shim")
else: print("pad shim present")
PY
pip install -q "numpy==2.4.6" "scipy>=1.13" --root-user-action=ignore 2>&1 | tail -1
echo "NODES_INST_DONE"
