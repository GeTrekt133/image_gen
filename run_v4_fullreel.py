#!/usr/bin/env python3
"""Full-reel generation from DZLY571hMqd.mp4: 4 splice-free segments (the reel's
outfit-change cuts stay as cuts), one persona, per-segment outfit prompt,
v2c recipe per segment, concat + per-segment source audio (perfect sync).
Output: gallery/sweep/v4_fullreel.mp4"""
import json, time, os, shutil, urllib.request, sys, copy, subprocess
import cv2, numpy as np
from PIL import Image

SRC = "/workspace/test_reels/DZLY571hMqd.mp4"
IN_DIR = "/workspace/ComfyUI/input"
COMFY_OUT = "/workspace/ComfyUI/output"
OUT = "/workspace/gallery/sweep"
TMP = "/tmp/claude-0/-workspace/a3dbfea7-d832-47e1-8bc4-d50747ea681e/scratchpad/reelfull"
os.makedirs(TMP, exist_ok=True)
HOST = "http://127.0.0.1:10100"
W, H = 512, 896
PERSONA = "a beautiful young woman with long wavy brown hair, {outfit}, in a minimalist beige room with soft natural daylight, photorealistic, natural skin texture, fine fabric detail, highly detailed"
SEGS = [  # (start_frame, n_frames(8k+1), outfit)
    (3,   129, "wearing an oversized black t-shirt and leopard print shorts"),
    (137,  33, "wearing a fitted white floral print midi dress with thin straps and matching floral arm sleeves"),
    (174,  33, "wearing a fitted black floral print midi dress with thin straps and matching floral arm sleeves"),
    (211,  33, "wearing a fitted white midi dress with red floral print and matching floral arm sleeves"),
]

def post(gr):
    req = urllib.request.Request(HOST + "/prompt", data=json.dumps({"prompt": gr}).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=120))

def getj(p):
    return json.load(urllib.request.urlopen(HOST + p, timeout=60))

# ---------- read all frames once ----------
cap = cv2.VideoCapture(SRC)
ALL = []
while True:
    ok, f = cap.read()
    if not ok: break
    f = cv2.resize(f, (512, 910), interpolation=cv2.INTER_LANCZOS4)
    ALL.append(f[(910 - H) // 2:(910 - H) // 2 + H])
cap.release()
print("source frames:", len(ALL))

from simple_lama_inpainting import SimpleLama
from rembg import remove, new_session
lama = SimpleLama()
sess = new_session("u2net_human_seg")
kernel = np.ones((21, 21), np.uint8)

def prep_segment(k, start, n):
    frames = [f.copy() for f in ALL[start:start + n]]
    # caption band per segment
    whiteish = np.zeros((H,), np.float64)
    for f in frames[::4]:
        hsv = cv2.cvtColor(f, cv2.COLOR_BGR2HSV)
        whiteish += ((hsv[:, :, 2] > 225) & (hsv[:, :, 1] < 40)).sum(axis=1)
    whiteish /= len(frames[::4])
    rows = np.where(whiteish > 25)[0]
    if len(rows):
        b0, b1 = max(rows.min() - 12, 0), min(rows.max() + 12, H)
        print(f"seg{k}: caption rows {b0}..{b1} -> LaMa")
        mask_img = Image.fromarray(np.where(np.arange(H)[:, None] * np.ones((1, W)) * 0 +
                                            ((np.arange(H) >= b0) & (np.arange(H) < b1))[:, None], 255, 0).astype(np.uint8))
        for i, f in enumerate(frames):
            res = lama(Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)), mask_img)
            frames[i] = cv2.cvtColor(np.array(res), cv2.COLOR_RGB2BGR)[:H, :W]
    else:
        print(f"seg{k}: no caption")
    # write drive
    for i, f in enumerate(frames):
        cv2.imwrite(f"{TMP}/s{k}_{i:05d}.png", f)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-framerate", "30", "-i", f"{TMP}/s{k}_%05d.png",
                    "-c:v", "libx264", "-crf", "10", "-pix_fmt", "yuv420p",
                    f"{IN_DIR}/drive_seg{k}.mp4"], check=True)
    # mask
    vw = cv2.VideoWriter(f"{TMP}/m{k}.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 30, (W, H))
    for f in frames:
        d = remove(Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)), session=sess)
        a = np.array(d)[:, :, 3]
        m = cv2.GaussianBlur(cv2.dilate((a > 30).astype(np.uint8) * 255, kernel, iterations=2), (9, 9), 0)
        vw.write(cv2.cvtColor(m, cv2.COLOR_GRAY2BGR))
    vw.release()
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", f"{TMP}/m{k}.mp4", "-c:v", "libx264",
                    "-crf", "6", "-pix_fmt", "yuv420p", f"{IN_DIR}/drive_seg{k}_mask.mp4"], check=True)
    # source audio slice for this segment
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{start/30:.4f}", "-t", f"{n/30:.4f}",
                    "-i", SRC, "-vn", "-c:a", "aac", "-b:a", "192k", f"{TMP}/aud{k}.m4a"], check=True)
    print(f"seg{k} prepped ({n} frames)")

def gen_segment(k, outfit):
    g = json.load(open("/workspace/wf_vace_v2.json"))
    g["5001"]["inputs"]["file"] = f"drive_seg{k}.mp4"
    g["7002"]["inputs"]["file"] = f"drive_seg{k}_mask.mp4"
    g["2483"]["inputs"]["text"] = PERSONA.format(outfit=outfit)
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
    g["4852"]["inputs"]["filename_prefix"] = f"v4s{k}_s1"
    g["8020"]["inputs"]["filename_prefix"] = f"v4_seg{k}"
    before = set(os.listdir(COMFY_OUT)); t0 = time.time()
    pid = post(g)["prompt_id"]
    print(f"seg{k} gen submitted {pid}")
    while time.time() - t0 < 3600:
        time.sleep(6)
        try:
            h = getj(f"/history/{pid}")
        except Exception:
            continue
        if pid in h:
            st = h[pid]["status"]
            if st.get("status_str") == "error":
                print(f"seg{k} ERROR", [str(m)[:300] for m in st.get("messages", []) if "error" in str(m).lower()][:2])
                return None
            if st.get("status_str") == "success" or st.get("completed"):
                break
    for f in sorted(os.listdir(COMFY_OUT)):
        if f.endswith(".mp4") and f not in before and f.startswith(f"v4_seg{k}"):
            print(f"seg{k} OK {time.time()-t0:.0f}s")
            return os.path.join(COMFY_OUT, f)
    print(f"seg{k} NO_OUTPUT"); return None

pieces = []
for k, (start, n, outfit) in enumerate(SEGS, 1):
    prep_segment(k, start, n)
for k, (start, n, outfit) in enumerate(SEGS, 1):
    v = gen_segment(k, outfit)
    if not v: sys.exit(2)
    av = f"{TMP}/seg{k}_av.mp4"   # replace generated audio with source slice
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", v, "-i", f"{TMP}/aud{k}.m4a",
                    "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "copy",
                    "-shortest", av], check=True)
    pieces.append(av)

with open(f"{TMP}/concat.txt", "w") as f:
    for p in pieces:
        f.write(f"file '{p}'\n")
subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0",
                "-i", f"{TMP}/concat.txt", "-c:v", "libx264", "-crf", "14",
                "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k",
                f"{OUT}/v4_fullreel.mp4"], check=True)
print(f"FULLREEL -> {OUT}/v4_fullreel.mp4")
print("V4_DONE")
