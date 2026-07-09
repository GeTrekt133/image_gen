#!/usr/bin/env bash
# Video-branch node install for LTX-2.3 motion-transfer (idempotent).
set -uo pipefail
source /workspace/env.sh
cd /workspace/ComfyUI
echo "== ComfyUI core requirements =="
pip install -q -r requirements.txt --root-user-action=ignore 2>&1 | tail -2

cd /workspace/ComfyUI/custom_nodes
clone(){ d=$(basename "$1"); [ -d "$d/.git" ] || git clone --depth 1 "$1" "$d" 2>&1 | tail -1;
  [ -f "$d/requirements.txt" ] && pip install -q -r "$d/requirements.txt" --root-user-action=ignore 2>&1 | tail -1; echo "  $d ok"; }
# core video-branch nodes
clone https://github.com/Lightricks/ComfyUI-LTXVideo
clone https://github.com/Fannovel16/comfyui_controlnet_aux
clone https://github.com/cubiq/ComfyUI_essentials
clone https://github.com/kijai/ComfyUI-Video-Depth-Anything
# UI-workflow support nodes (harmless, help object_info completeness)
clone https://github.com/rgthree/rgthree-comfy
clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite
clone https://github.com/kijai/ComfyUI-KJNodes
clone https://github.com/evanspearman/ComfyMath
clone https://github.com/city96/ComfyUI-GGUF

# --- kornia pad shim (LTXVideo pyramid_blending import breaks on kornia>=0.8) ---
python - <<'PY'
p="/workspace/ComfyUI/custom_nodes/ComfyUI-LTXVideo/pyramid_blending.py"
import os
if os.path.exists(p):
    s=open(p).read()
    if "pad = F.pad" not in s and "pad = __import__" not in s:
        s=s.replace("    is_powerof_two,\n    pad,\n)","    is_powerof_two,\n)\npad = __import__('torch').nn.functional.pad")
        open(p,"w").write(s); print("pad shim applied")
    else: print("pad shim present/na")
else: print("pyramid_blending.py not found (skip)")
PY

# --- PyAV rotation patch (LoadImage/LoadVideo crash on frame.rotation) ---
python - <<'PY'
import glob
for p in glob.glob("/workspace/ComfyUI/comfy_api/**/video_types.py",recursive=True):
    s=open(p).read()
    if "getattr(frame" not in s and "frame.rotation" in s:
        s=s.replace("frame.rotation","(getattr(frame,'rotation',0) or 0)")
        open(p,"w").write(s); print("rotation patch ->",p)
    else: print("rotation patch present/na ->",p)
PY

# Blackwell numpy/scipy pin
pip install -q "numpy==2.4.6" "scipy>=1.13" --root-user-action=ignore 2>&1 | tail -1
echo "NODES_INST_DONE"
