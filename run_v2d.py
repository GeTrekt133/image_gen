#!/usr/bin/env python3
"""v2d = v2c_hires graph + 'pro' stage-1 (distill 0, LTXVScheduler 30 steps, cfg 3,
euler_ancestral) while stage-2 keeps the official distilled tail via its own
LoRA chain. Files on disk untouched; wf_vace_v2.json patched in memory."""
import json, time, os, shutil, urllib.request, sys, copy
import numpy as np, cv2

HOST = "http://127.0.0.1:10100"
COMFY_OUT = "/workspace/ComfyUI/output"
OUT = "/workspace/gallery/sweep"
g = json.load(open("/workspace/wf_vace_v2.json"))

# --- hires (as v2c) ---
g["9000"] = {"class_type": "ResizeImageMaskNode", "inputs": {
    "input": ["5000", 0], "resize_type": "scale by multiplier",
    "resize_type.multiplier": 1.5, "scale_method": "lanczos"}}
g["7010"]["inputs"]["pixels"] = ["9000", 0]
g["5026"]["inputs"]["input"] = ["9000", 0]
g["5026"]["inputs"]["resize_type.shorter_size"] = 768
g["9001"] = {"class_type": "ResizeImageMaskNode", "inputs": {
    "input": ["7003", 0], "resize_type": "scale by multiplier",
    "resize_type.multiplier": 1.5, "scale_method": "lanczos"}}
g["7004"]["inputs"]["image"] = ["9001", 0]
g["8000"]["inputs"]["image_b"] = ["9000", 0]
g["8001"]["inputs"]["resize_type.multiplier"] = 1.5
g["8003"]["inputs"]["input"] = ["9001", 0]
g["8003"]["inputs"]["resize_type.multiplier"] = 1.5
g["8016"]["inputs"]["input"] = ["9000", 0]
g["8016"]["inputs"]["resize_type.multiplier"] = 1.5

# --- pro stage-1: no distill, 30-step scheduler, cfg 3, euler_ancestral ---
g["4922"]["inputs"]["strength_model"] = 0.0
g["7100"] = {"class_type": "LTXVScheduler", "inputs": {
    "steps": 30, "max_shift": 2.05, "base_shift": 0.95,
    "stretch": True, "terminal": 0.1}}
g["4829"]["inputs"]["sigmas"] = ["7100", 0]
del g["5025"]
g["4828"]["inputs"]["cfg"] = 3.0
g["4831"]["inputs"]["sampler_name"] = "euler_ancestral"

# --- stage-2 keeps distilled 0.5 via its own chain ---
g["9100"] = {"class_type": "LoraLoaderModelOnly", "inputs": {
    "model": ["3940", 0],
    "lora_name": "ltxv/ltx2/ltx-2.3-22b-distilled-lora-384-1.1.safetensors",
    "strength_model": 0.5}}
g["9101"] = {"class_type": "LTXICLoRALoaderModelOnly", "inputs": {
    "model": ["9100", 0],
    "lora_name": "ltxv/ltx2/ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors",
    "strength_model": 1.0}}
g["8012"]["inputs"]["model"] = ["9101", 0]

g["4852"]["inputs"]["filename_prefix"] = "v2d_s1"
g["8020"]["inputs"]["filename_prefix"] = "v2d_pro"

def post(gr):
    req = urllib.request.Request(HOST + "/prompt", data=json.dumps({"prompt": gr}).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=120))

def getj(p):
    return json.load(urllib.request.urlopen(HOST + p, timeout=60))

before = set(os.listdir(COMFY_OUT)); t0 = time.time()
try:
    pid = post(g)["prompt_id"]
except urllib.error.HTTPError as e:
    print("SUBMIT_FAIL", e.read().decode()[:1200]); sys.exit(1)
print("submitted", pid)
while time.time() - t0 < 3600:
    time.sleep(6)
    try:
        h = getj(f"/history/{pid}")
    except Exception:
        continue
    if pid in h:
        st = h[pid]["status"]
        if st.get("status_str") == "error":
            print("ERROR", [str(m)[:400] for m in st.get("messages", []) if "error" in str(m).lower()][:2]); sys.exit(2)
        if st.get("status_str") == "success" or st.get("completed"):
            break
new = [f for f in os.listdir(COMFY_OUT) if f.endswith(".mp4") and f not in before]
for f in sorted(new):
    if f.startswith("v2d_pro"):
        shutil.copy(os.path.join(COMFY_OUT, f), f"{OUT}/v2d_pro.mp4")
        print(f"OK {time.time()-t0:.0f}s -> {OUT}/v2d_pro.mp4")
print("V2D_DONE")
