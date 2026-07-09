#!/usr/bin/env python3
"""v6: outfit-swap pass over v5_singlepass. Segments 2-4 of the v5 output are
re-generated with per-segment outfit prompts, masking ONLY the clothing region
(person silhouette minus protected face ellipse) — the face keeps v5's exact
pixels, so identity is physically locked. Motion comes from v5 itself (DWPose
of the v5 frames drives the union IC-LoRA as usual).
Segments 2/3 are 37 frames -> padded with 4 frozen frames to 41 (8n+1), the pad
is discarded on reassembly. Output: gallery/sweep/v6_outfits.mp4"""
import json, time, os, shutil, urllib.request, sys, subprocess
import cv2, numpy as np
from PIL import Image

V5 = "/workspace/gallery/sweep/v5_singlepass.mp4"
IN_DIR = "/workspace/ComfyUI/input"
COMFY_OUT = "/workspace/ComfyUI/output"
OUT = "/workspace/gallery/sweep"
TMP = "/tmp/claude-0/-workspace/a3dbfea7-d832-47e1-8bc4-d50747ea681e/scratchpad/v6"
os.makedirs(TMP, exist_ok=True)
HOST = "http://127.0.0.1:10100"
W, H = 512, 896
PERSONA = ("a beautiful young woman with long wavy brown hair, {outfit}, in a minimalist "
           "beige room with soft natural daylight, photorealistic, natural skin texture, "
           "fine fabric detail, highly detailed")
SEGS = [  # (start_in_v5, n_real, n_gen(8k+1), outfit)
    (131, 37, 41, "wearing a fitted white floral print midi dress with thin straps"),
    (168, 37, 41, "wearing a fitted black floral print midi dress with thin straps"),
    (205, 33, 33, "wearing a fitted white midi dress with bold red floral print"),
]

def post(gr):
    req = urllib.request.Request(HOST + "/prompt", data=json.dumps({"prompt": gr}).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=120))

def getj(p):
    return json.load(urllib.request.urlopen(HOST + p, timeout=60))

# ---- load all v5 frames (full res 1152x2016) ----
cap = cv2.VideoCapture(V5)
V5F = []
while True:
    ok, f = cap.read()
    if not ok: break
    V5F.append(f)
cap.release()
print("v5 frames:", len(V5F))

from rembg import remove, new_session
sess = new_session("u2net_human_seg")
kernel = np.ones((21, 21), np.uint8)

def prep_segment(k, start, n_real, n_gen):
    frames = [cv2.resize(V5F[start + min(i, n_real - 1)], (W, H),
                         interpolation=cv2.INTER_AREA) for i in range(n_gen)]
    for i, f in enumerate(frames):
        cv2.imwrite(f"{TMP}/s{k}_{i:05d}.png", f)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-framerate", "30", "-i", f"{TMP}/s{k}_%05d.png",
                    "-c:v", "libx264", "-crf", "10", "-pix_fmt", "yuv420p",
                    f"{IN_DIR}/v6drive{k}.mp4"], check=True)
    vw = cv2.VideoWriter(f"{TMP}/m{k}.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 30, (W, H))
    for f in frames:
        d = remove(Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)), session=sess)
        a = np.array(d)[:, :, 3]
        m = cv2.dilate((a > 30).astype(np.uint8) * 255, kernel, iterations=2)
        ys, xs = np.where(m > 127)
        if len(ys) > 50:  # protect face: ellipse around head (top of silhouette)
            ytop = ys.min()
            band = (ys >= ytop) & (ys < ytop + 110)
            cx, cy = int(xs[band].mean()), ytop + 62
            cv2.ellipse(m, (cx, cy), (58, 78), 0, 0, 360, 0, -1)
        m = cv2.GaussianBlur(m, (9, 9), 0)
        vw.write(cv2.cvtColor(m, cv2.COLOR_GRAY2BGR))
    vw.release()
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", f"{TMP}/m{k}.mp4", "-c:v", "libx264",
                    "-crf", "6", "-pix_fmt", "yuv420p", f"{IN_DIR}/v6mask{k}.mp4"], check=True)
    print(f"seg{k} prepped ({n_gen} frames, face-protected mask)")

def gen_segment(k, outfit):
    g = json.load(open("/workspace/wf_vace_v2.json"))
    g["5001"]["inputs"]["file"] = f"v6drive{k}.mp4"
    g["7002"]["inputs"]["file"] = f"v6mask{k}.mp4"
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
    g["4852"]["inputs"]["filename_prefix"] = f"v6s{k}_s1"
    g["8020"]["inputs"]["filename_prefix"] = f"v6_seg{k}"
    before = set(os.listdir(COMFY_OUT)); t0 = time.time()
    pid = post(g)["prompt_id"]
    print(f"seg{k} submitted {pid}")
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
        if f.endswith(".mp4") and f not in before and f.startswith(f"v6_seg{k}"):
            print(f"seg{k} OK {time.time()-t0:.0f}s")
            return os.path.join(COMFY_OUT, f)
    return None

for k, (start, n_real, n_gen, outfit) in enumerate(SEGS, 2):
    prep_segment(k, start, n_real, n_gen)

final = [f.copy() for f in V5F]
for k, (start, n_real, n_gen, outfit) in enumerate(SEGS, 2):
    v = gen_segment(k, outfit)
    if not v: sys.exit(2)
    cap = cv2.VideoCapture(v); gen = []
    while True:
        ok, f = cap.read()
        if not ok: break
        gen.append(f)
    cap.release()
    th, tw = V5F[0].shape[:2]
    for i in range(min(n_real, len(gen))):
        final[start + i] = cv2.resize(gen[i], (tw, th), interpolation=cv2.INTER_LANCZOS4)
    print(f"seg{k} spliced ({min(n_real, len(gen))} frames)")

for i, f in enumerate(final):
    cv2.imwrite(f"{TMP}/out_{i:05d}.png", f)
subprocess.run(["ffmpeg", "-v", "error", "-y", "-framerate", "30", "-i", f"{TMP}/out_%05d.png",
                "-i", V5, "-map", "0:v", "-map", "1:a", "-c:v", "libx264", "-crf", "14",
                "-pix_fmt", "yuv420p", "-c:a", "copy", "-shortest",
                f"{OUT}/v6_outfits.mp4"], check=True)
print(f"FINAL -> {OUT}/v6_outfits.mp4")
print("V6_DONE")
