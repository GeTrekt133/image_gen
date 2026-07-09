#!/usr/bin/env python3
"""v3 = the v2c_hires recipe (user-picked winner) on the NEW splice-free driving
clip (drive_new.mp4 / drive_new_mask.mp4, native 30 fps, frames 3-123 of
DZLY571hMqd). Same graph patches as v2c; only drive files + prompt change."""
import json, time, os, shutil, urllib.request, sys
import numpy as np, cv2

HOST = "http://127.0.0.1:10100"
COMFY_OUT = "/workspace/ComfyUI/output"
OUT = "/workspace/gallery/sweep"
g = json.load(open("/workspace/wf_vace_v2.json"))

# new drive
g["5001"]["inputs"]["file"] = "drive_new.mp4"
g["7002"]["inputs"]["file"] = "drive_new_mask.mp4"
g["2483"]["inputs"]["text"] = (
    "a beautiful young woman with long wavy brown hair wearing an oversized black "
    "t-shirt and leopard print shorts, in a minimalist beige room with soft natural "
    "daylight, photorealistic, natural skin texture, fine fabric detail, highly detailed")

# v2c hires patches (identical to the winning run)
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

g["4852"]["inputs"]["filename_prefix"] = "v3_s1"
g["8020"]["inputs"]["filename_prefix"] = "v3_newdrive"

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
got = None
for f in sorted(new):
    if f.startswith("v3_newdrive"):
        got = f"{OUT}/v3_newdrive.mp4"
        shutil.copy(os.path.join(COMFY_OUT, f), got)
        print(f"OK {time.time()-t0:.0f}s -> {got}")
if not got:
    print("NO_OUTPUT"); sys.exit(3)

def metrics(path):
    cap = cv2.VideoCapture(path); prev = None; ds = []
    while True:
        ok, f = cap.read()
        if not ok: break
        gr = cv2.cvtColor(cv2.resize(f, (512, 896)), cv2.COLOR_BGR2GRAY).astype(np.float32)
        if prev is not None: ds.append(np.abs(gr - prev).mean())
        prev = gr
    cap.release(); return float(np.mean(ds))
print(f"tNoise v3={metrics(got):.2f}  drive_new={metrics('/workspace/ComfyUI/input/drive_new.mp4'):.2f}")
print("V3_DONE")
