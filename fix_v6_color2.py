#!/usr/bin/env python3
"""v2 of the halo fix — LOCAL low-frequency wall correction (the mismatch is
spatially varying, a global affine cannot fix it; verified on frame 222).
Per frame:
  1. gen_region  = replica of the graph's blend mask (raw mask -> dilate 32 ->
     low-res(64) dilate 5 -> resize), soft alpha.
  2. persona     = rembg silhouette of the FINAL frame (generated girl), dilated.
  3. wall_fix    = gen_region minus persona  (only the generated wall band).
  4. plate       = real wall extrapolated under the gen region (TELEA inpaint
     at 1/8 scale = low-frequency continuation of the real background).
  5. cur_lowfreq = normalized masked blur of the current frame over wall_fix.
  6. F += (plate - cur_lowfreq) * alpha  on wall_fix, delta clipped to +-30.
Face/subject pixels are untouched. Output: gallery/sweep/v6_colorfix2.mp4"""
import cv2, numpy as np, subprocess, os
from PIL import Image
from rembg import remove, new_session

V6 = "/workspace/gallery/sweep/v6_outfits.mp4"
MASKV = "/workspace/ComfyUI/input/drive_full_mask.mp4"
OUT = "/workspace/gallery/sweep/v6_colorfix2.mp4"
TMP = "/tmp/claude-0/-workspace/a3dbfea7-d832-47e1-8bc4-d50747ea681e/scratchpad/cfix2"
os.makedirs(TMP, exist_ok=True)
sess = new_session("u2net_human_seg")

capv = cv2.VideoCapture(V6)
capm = cv2.VideoCapture(MASKV)
tw, th = int(capv.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capv.get(cv2.CAP_PROP_FRAME_HEIGHT))
sw, sh = tw // 8, th // 8          # working scale for low-frequency fields
i = 0
resid_before = []; resid_after = []
last_m = None
while True:
    okv, f = capv.read()
    okm, mf = capm.read()
    if not okv: break
    if okm: last_m = mf
    mu = (cv2.resize(last_m[:, :, 2], (tw, th)) > 127).astype(np.uint8) * 255

    # 1) blend-mask replica (soft)
    d32 = cv2.dilate(mu, np.ones((65, 65), np.uint8))
    small = cv2.resize(d32, (round(64 * tw / th), 64), interpolation=cv2.INTER_AREA)
    small = cv2.dilate(small, np.ones((11, 11), np.uint8))
    alpha = cv2.resize(small, (tw, th), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
    alpha = cv2.GaussianBlur(alpha, (61, 61), 0)
    gen_hard = alpha > 0.03

    # 2) persona silhouette from the FINAL frame
    small_f = cv2.resize(f, (tw // 3, th // 3))
    d = remove(Image.fromarray(cv2.cvtColor(small_f, cv2.COLOR_BGR2RGB)), session=sess)
    pm = (np.array(d)[:, :, 3] > 25).astype(np.uint8) * 255
    pm = cv2.dilate(pm, np.ones((13, 13), np.uint8))
    persona = cv2.resize(pm, (tw, th), interpolation=cv2.INTER_LINEAR) > 127

    wall_fix = gen_hard & ~persona

    # 4) real-wall plate: inpaint the whole gen+persona area from real pixels (1/8 scale)
    fs = cv2.resize(f, (sw, sh)).astype(np.float32)
    hole = cv2.resize(((gen_hard | persona).astype(np.uint8) * 255), (sw, sh)) > 40
    plate = cv2.inpaint(fs.astype(np.uint8), hole.astype(np.uint8) * 255, 7, cv2.INPAINT_TELEA).astype(np.float32)
    plate = cv2.GaussianBlur(plate, (15, 15), 0)

    # 5) current low-frequency of the generated wall (normalized masked blur)
    wsmall = (cv2.resize(wall_fix.astype(np.uint8) * 255, (sw, sh)) > 90).astype(np.float32)
    num = cv2.GaussianBlur(fs * wsmall[:, :, None], (31, 31), 0)
    den = cv2.GaussianBlur(wsmall, (31, 31), 0)[:, :, None] + 1e-4
    cur = num / den

    delta_s = np.clip(plate - cur, -30, 30)
    delta = cv2.resize(delta_s, (tw, th), interpolation=cv2.INTER_LINEAR)

    w = (alpha * wall_fix.astype(np.float32))[:, :, None]
    resid_before.append(float(np.abs(delta[wall_fix]).mean()) if wall_fix.sum() else 0.0)
    out = np.clip(f.astype(np.float32) + delta * w, 0, 255).astype(np.uint8)

    # residual after (recompute cur on corrected frame, cheap check every 30th)
    if i % 30 == 0 and wall_fix.sum():
        fs2 = cv2.resize(out, (sw, sh)).astype(np.float32)
        num2 = cv2.GaussianBlur(fs2 * wsmall[:, :, None], (31, 31), 0)
        cur2 = num2 / den
        resid_after.append(float(np.abs(np.clip(plate - cur2, -30, 30))[cv2.resize(wall_fix.astype(np.uint8)*255,(sw,sh))>90].mean()))
    cv2.imwrite(f"{TMP}/c_{i:05d}.png", out)
    i += 1
capv.release(); capm.release()
print(f"frames {i}; mean wall residual before={np.mean(resid_before):.2f}, after(sampled)={np.mean(resid_after):.2f}")
subprocess.run(["ffmpeg", "-v", "error", "-y", "-framerate", "30", "-i", f"{TMP}/c_%05d.png",
                "-i", V6, "-map", "0:v", "-map", "1:a", "-c:v", "libx264", "-crf", "14",
                "-pix_fmt", "yuv420p", "-c:a", "copy", "-shortest", OUT], check=True)
print("->", OUT)
print("CFIX2_DONE")
