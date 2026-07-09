#!/usr/bin/env python3
"""Prepare the new driving clip from test_reels/DZLY571hMqd.mp4 (splice-free window).
1) trim frames 3..123 (121 frames, first-outfit segment, no cuts), native 30 fps,
   scale 720x1280 -> 512x910 -> center-crop 512x896
2) detect burned-in caption band (near-white text rows persistent over time),
   LaMa-inpaint it per frame -> ComfyUI/input/drive_new.mp4
3) rembg dancer silhouette mask -> ComfyUI/input/drive_new_mask.mp4
"""
import cv2, numpy as np, os, subprocess
from PIL import Image

SRC = "/workspace/test_reels/DZLY571hMqd.mp4"
IN_DIR = "/workspace/ComfyUI/input"
START, N = 3, 121
W, H = 512, 896

# ---- 1) read + trim + resize ----
cap = cv2.VideoCapture(SRC)
frames = []
i = 0
while True:
    ok, f = cap.read()
    if not ok: break
    if START <= i < START + N:
        f = cv2.resize(f, (512, 910), interpolation=cv2.INTER_LANCZOS4)
        y0 = (910 - H) // 2
        frames.append(f[y0:y0 + H])
    i += 1
cap.release()
assert len(frames) == N, f"got {len(frames)} frames"
print(f"trimmed {N} frames @512x896")

# ---- 2) caption band detection ----
whiteish = np.zeros((H,), np.float64)
for f in frames[::5]:
    hsv = cv2.cvtColor(f, cv2.COLOR_BGR2HSV)
    m = (hsv[:, :, 2] > 225) & (hsv[:, :, 1] < 40)
    whiteish += m.sum(axis=1)
whiteish /= len(frames[::5])
rows = np.where(whiteish > 25)[0]          # rows with persistent white text
if len(rows):
    band0, band1 = max(rows.min() - 12, 0), min(rows.max() + 12, H)
    print(f"caption band rows {band0}..{band1}")
else:
    band0 = band1 = None
    print("no caption band detected")

if band0 is not None:
    from simple_lama_inpainting import SimpleLama
    lama = SimpleLama()
    mask = np.zeros((H, W), np.uint8)
    mask[band0:band1, :] = 255
    mask_img = Image.fromarray(mask)
    for k, f in enumerate(frames):
        res = lama(Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)), mask_img)
        frames[k] = cv2.cvtColor(np.array(res), cv2.COLOR_RGB2BGR)[:H, :W]
    print("caption LaMa-cleaned on all frames")

# ---- write drive_new.mp4 (30 fps native) ----
tmp = "/tmp/claude-0/-workspace/a3dbfea7-d832-47e1-8bc4-d50747ea681e/scratchpad/drv"
os.makedirs(tmp, exist_ok=True)
for k, f in enumerate(frames):
    cv2.imwrite(f"{tmp}/d_{k:05d}.png", f)
dst = f"{IN_DIR}/drive_new.mp4"
subprocess.run(["ffmpeg", "-v", "error", "-y", "-framerate", "30",
                "-i", f"{tmp}/d_%05d.png", "-c:v", "libx264", "-crf", "10",
                "-pix_fmt", "yuv420p", dst], check=True)
print("->", dst)

# ---- 3) mask via rembg ----
from rembg import remove, new_session
sess = new_session("u2net_human_seg")
kernel = np.ones((21, 21), np.uint8)
vw = cv2.VideoWriter(f"{tmp}/mask_raw.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 30, (W, H))
for f in frames:
    rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
    d = remove(Image.fromarray(rgb), session=sess)
    a = np.array(d)[:, :, 3]
    m = (a > 30).astype(np.uint8) * 255
    m = cv2.dilate(m, kernel, iterations=2)
    m = cv2.GaussianBlur(m, (9, 9), 0)
    vw.write(cv2.cvtColor(m, cv2.COLOR_GRAY2BGR))
vw.release()
subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", f"{tmp}/mask_raw.mp4",
                "-c:v", "libx264", "-crf", "6", "-pix_fmt", "yuv420p",
                f"{IN_DIR}/drive_new_mask.mp4"], check=True)
print("->", f"{IN_DIR}/drive_new_mask.mp4")
print("PREP_DONE")
