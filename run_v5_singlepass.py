#!/usr/bin/env python3
"""v5 experiment: the WHOLE reel (frames 3..243, 241 = 8*30+1, native 30 fps)
in ONE generation pass — hard cuts and outfit changes included. v2c recipe.
Purpose: see how the model handles cuts inside a single latent video."""
import json, time, os, shutil, urllib.request, sys, subprocess
import cv2, numpy as np
from PIL import Image

SRC = "/workspace/test_reels/DZLY571hMqd.mp4"
IN_DIR = "/workspace/ComfyUI/input"
COMFY_OUT = "/workspace/ComfyUI/output"
OUT = "/workspace/gallery/sweep"
TMP = "/tmp/claude-0/-workspace/a3dbfea7-d832-47e1-8bc4-d50747ea681e/scratchpad/reelsp"
os.makedirs(TMP, exist_ok=True)
HOST = "http://127.0.0.1:10100"
W, H = 512, 896
START, N = 3, 241
CUTS_LOCAL = [131, 168, 205]  # source cuts 134/171/208 minus START

def post(gr):
    req = urllib.request.Request(HOST + "/prompt", data=json.dumps({"prompt": gr}).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=120))

def getj(p):
    return json.load(urllib.request.urlopen(HOST + p, timeout=60))

# ---- frames ----
cap = cv2.VideoCapture(SRC)
frames = []; i = 0
while True:
    ok, f = cap.read()
    if not ok: break
    if START <= i < START + N:
        f = cv2.resize(f, (512, 910), interpolation=cv2.INTER_LANCZOS4)
        frames.append(f[(910 - H) // 2:(910 - H) // 2 + H])
    i += 1
cap.release()
assert len(frames) == N
print("frames:", N)

# ---- caption cleanup per inter-cut window ----
from simple_lama_inpainting import SimpleLama
lama = SimpleLama()
bounds = [0] + CUTS_LOCAL + [N]
for w0, w1 in zip(bounds[:-1], bounds[1:]):
    seg = frames[w0:w1]
    whiteish = np.zeros((H,), np.float64)
    for f in seg[::4]:
        hsv = cv2.cvtColor(f, cv2.COLOR_BGR2HSV)
        whiteish += ((hsv[:, :, 2] > 225) & (hsv[:, :, 1] < 40)).sum(axis=1)
    whiteish /= max(len(seg[::4]), 1)
    rows = np.where(whiteish > 25)[0]
    if not len(rows):
        print(f"window {w0}..{w1}: no caption"); continue
    b0, b1 = max(rows.min() - 12, 0), min(rows.max() + 12, H)
    print(f"window {w0}..{w1}: caption rows {b0}..{b1}")
    band = ((np.arange(H) >= b0) & (np.arange(H) < b1))[:, None] * np.ones((1, W))
    mask_img = Image.fromarray((band * 255).astype(np.uint8))
    for k in range(w0, w1):
        res = lama(Image.fromarray(cv2.cvtColor(frames[k], cv2.COLOR_BGR2RGB)), mask_img)
        frames[k] = cv2.cvtColor(np.array(res), cv2.COLOR_RGB2BGR)[:H, :W]

for k, f in enumerate(frames):
    cv2.imwrite(f"{TMP}/f_{k:05d}.png", f)
subprocess.run(["ffmpeg", "-v", "error", "-y", "-framerate", "30", "-i", f"{TMP}/f_%05d.png",
                "-c:v", "libx264", "-crf", "10", "-pix_fmt", "yuv420p",
                f"{IN_DIR}/drive_full.mp4"], check=True)
print("-> drive_full.mp4")

# ---- mask ----
from rembg import remove, new_session
sess = new_session("u2net_human_seg")
kernel = np.ones((21, 21), np.uint8)
vw = cv2.VideoWriter(f"{TMP}/mask.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 30, (W, H))
for f in frames:
    d = remove(Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)), session=sess)
    a = np.array(d)[:, :, 3]
    m = cv2.GaussianBlur(cv2.dilate((a > 30).astype(np.uint8) * 255, kernel, iterations=2), (9, 9), 0)
    vw.write(cv2.cvtColor(m, cv2.COLOR_GRAY2BGR))
vw.release()
subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", f"{TMP}/mask.mp4", "-c:v", "libx264",
                "-crf", "6", "-pix_fmt", "yuv420p", f"{IN_DIR}/drive_full_mask.mp4"], check=True)
print("-> drive_full_mask.mp4")

# ---- generation: v2c recipe, single pass ----
g = json.load(open("/workspace/wf_vace_v2.json"))
g["5001"]["inputs"]["file"] = "drive_full.mp4"
g["7002"]["inputs"]["file"] = "drive_full_mask.mp4"
g["2483"]["inputs"]["text"] = (
    "a beautiful young woman with long wavy brown hair wearing stylish fitted clothes, "
    "in a minimalist beige room with soft natural daylight, photorealistic, natural skin "
    "texture, fine fabric detail, highly detailed")
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
g["4852"]["inputs"]["filename_prefix"] = "v5_s1"
g["8020"]["inputs"]["filename_prefix"] = "v5_singlepass"

before = set(os.listdir(COMFY_OUT)); t0 = time.time()
try:
    pid = post(g)["prompt_id"]
except urllib.error.HTTPError as e:
    print("SUBMIT_FAIL", e.read().decode()[:1200]); sys.exit(1)
print("submitted", pid)
while time.time() - t0 < 3600:
    time.sleep(8)
    try:
        h = getj(f"/history/{pid}")
    except Exception:
        continue
    if pid in h:
        st = h[pid]["status"]
        if st.get("status_str") == "error":
            print("ERROR", [str(m)[:500] for m in st.get("messages", []) if "error" in str(m).lower()][:2]); sys.exit(2)
        if st.get("status_str") == "success" or st.get("completed"):
            break
got = None
for f in sorted(os.listdir(COMFY_OUT)):
    if f.endswith(".mp4") and f not in before and f.startswith("v5_singlepass"):
        got = os.path.join(COMFY_OUT, f)
if not got:
    print("NO_OUTPUT"); sys.exit(3)
print(f"gen OK {time.time()-t0:.0f}s")
# source audio for the same window
subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{START/30:.4f}", "-t", f"{N/30:.4f}",
                "-i", SRC, "-vn", "-c:a", "aac", "-b:a", "192k", f"{TMP}/aud.m4a"], check=True)
subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", got, "-i", f"{TMP}/aud.m4a",
                "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "copy", "-shortest",
                f"{OUT}/v5_singlepass.mp4"], check=True)
print(f"FULL -> {OUT}/v5_singlepass.mp4")
print("V5_DONE")
