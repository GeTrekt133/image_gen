#!/usr/bin/env python3
"""Post-fix for the visible mask halo in v6_outfits.mp4: the regenerated region
is slightly brighter than the real background. Per frame, fit an affine color
transform (per channel gain+offset) from the ring just INSIDE the mask boundary
(generated wall) to the ring just OUTSIDE (real wall), apply it to the whole
generated region with a feathered alpha. No regeneration needed.
Masks: drive_full_mask.mp4 (aligned with the v5/v6 timeline)."""
import cv2, numpy as np, subprocess, os

V6 = "/workspace/gallery/sweep/v6_outfits.mp4"
MASKV = "/workspace/ComfyUI/input/drive_full_mask.mp4"
OUT = "/workspace/gallery/sweep/v6_colorfix.mp4"
TMP = "/tmp/claude-0/-workspace/a3dbfea7-d832-47e1-8bc4-d50747ea681e/scratchpad/cfix"
os.makedirs(TMP, exist_ok=True)

capv = cv2.VideoCapture(V6)
capm = cv2.VideoCapture(MASKV)
tw, th = int(capv.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capv.get(cv2.CAP_PROP_FRAME_HEIGHT))
k_er = np.ones((25, 25), np.uint8)
k_di = np.ones((25, 25), np.uint8)
i = 0
gains = []
while True:
    okv, f = capv.read()
    okm, mf = capm.read()
    if not okv: break
    if not okm: mf = last_m
    last_m = mf
    m = cv2.resize(mf[:, :, 2], (tw, th), interpolation=cv2.INTER_LINEAR) > 127
    mu = m.astype(np.uint8) * 255
    inner_ring = (mu > 0) & (cv2.erode(mu, k_er) == 0)          # gen side of boundary
    outer_ring = (cv2.dilate(mu, k_di) > 0) & (mu == 0)          # real side
    ff = f.astype(np.float32)
    if inner_ring.sum() > 500 and outer_ring.sum() > 500:
        a = np.ones(3, np.float32); b = np.zeros(3, np.float32)
        for c in range(3):
            pi, po = ff[:, :, c][inner_ring], ff[:, :, c][outer_ring]
            si, so = max(pi.std(), 1e-3), po.std()
            g = float(np.clip(so / si, 0.85, 1.18))
            a[c] = g; b[c] = float(po.mean() - g * pi.mean())
        gains.append(a.mean())
        corr = np.clip(ff * a[None, None] + b[None, None], 0, 255)
        alpha = cv2.GaussianBlur(mu.astype(np.float32) / 255.0, (31, 31), 0)[:, :, None]
        f = (ff * (1 - alpha) + corr * alpha).astype(np.uint8)
    cv2.imwrite(f"{TMP}/c_{i:05d}.png", f)
    i += 1
capv.release(); capm.release()
print(f"frames {i}, mean gain {np.mean(gains):.3f} (<1 = darkened gen region)")
subprocess.run(["ffmpeg", "-v", "error", "-y", "-framerate", "30", "-i", f"{TMP}/c_%05d.png",
                "-i", V6, "-map", "0:v", "-map", "1:a", "-c:v", "libx264", "-crf", "14",
                "-pix_fmt", "yuv420p", "-c:a", "copy", "-shortest", OUT], check=True)
print("->", OUT)
print("CFIX_DONE")
