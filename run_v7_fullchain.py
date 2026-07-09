#!/usr/bin/env python3
"""v7: full chain with the halo fixed IN-GRAPH (post-hoc patching rejected).
Fixes vs v5/v6 graphs:
  - ColorMatch (KJNodes, mkl) on the decoded gen frames BEFORE each Laplacian
    blend, reference = the real frames of that stage -> gen exposure/color is
    pulled to the real footage, so the blend has nothing to hide.
  - Laplacian mask_low_res_dilation 5 -> 2 (the ~150px feather band shrinks).
  - No lighting words in the prompt (was "soft natural daylight" in a dim room
    -> the very source of the brightness mismatch).
Phase A: single-pass over drive_full (241f) -> v7_base.mp4
Phase B: outfit swap segments 2-4 over v7_base  -> v7_outfits.mp4"""
import json, time, os, shutil, urllib.request, sys, subprocess
import cv2, numpy as np
from PIL import Image

IN_DIR = "/workspace/ComfyUI/input"
COMFY_OUT = "/workspace/ComfyUI/output"
OUT = "/workspace/gallery/sweep"
SRC = "/workspace/test_reels/DZLY571hMqd.mp4"
TMP = "/tmp/claude-0/-workspace/a3dbfea7-d832-47e1-8bc4-d50747ea681e/scratchpad/v7"
os.makedirs(TMP, exist_ok=True)
HOST = "http://127.0.0.1:10100"
W, H = 512, 896
PERSONA = ("a beautiful young woman with long wavy brown hair, {outfit}, in a minimalist "
           "beige room, photorealistic, natural skin texture, fine fabric detail, highly detailed")

def post(gr):
    req = urllib.request.Request(HOST + "/prompt", data=json.dumps({"prompt": gr}).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=120))

def getj(p):
    return json.load(urllib.request.urlopen(HOST + p, timeout=60))

def wait_get(pid, prefix, before, timeout=3600):
    t0 = time.time()
    while time.time() - t0 < timeout:
        time.sleep(6)
        try:
            h = getj(f"/history/{pid}")
        except Exception:
            continue
        if pid in h:
            st = h[pid]["status"]
            if st.get("status_str") == "error":
                print(prefix, "ERROR", [str(m)[:300] for m in st.get("messages", []) if "error" in str(m).lower()][:2])
                return None
            if st.get("status_str") == "success" or st.get("completed"):
                break
    for f in sorted(os.listdir(COMFY_OUT)):
        if f.endswith(".mp4") and f not in before and f.startswith(prefix):
            print(f"{prefix} OK {time.time()-t0:.0f}s")
            return os.path.join(COMFY_OUT, f)
    print(prefix, "NO_OUTPUT"); return None

def build_graph(drive, maskv, prompt, prefix):
    g = json.load(open("/workspace/wf_vace_v2.json"))
    g["5001"]["inputs"]["file"] = drive
    g["7002"]["inputs"]["file"] = maskv
    g["2483"]["inputs"]["text"] = prompt
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
    g["8001"]["inputs"]["resize_type.multiplier"] = 1.5
    g["8003"]["inputs"]["input"] = ["9001", 0]
    g["8003"]["inputs"]["resize_type.multiplier"] = 1.5
    g["8016"]["inputs"]["input"] = ["9000", 0]
    g["8016"]["inputs"]["resize_type.multiplier"] = 1.5
    # --- halo fixes ---
    g["8100"] = {"class_type": "ColorMatch", "inputs": {
        "image_ref": ["9000", 0], "image_target": ["5065", 0],
        "method": "mkl", "strength": 1.0, "multithread": True}}
    g["8000"]["inputs"]["image_a"] = ["8100", 0]
    g["8000"]["inputs"]["image_b"] = ["9000", 0]
    g["8000"]["inputs"]["mask_low_res_dilation"] = 2
    g["8101"] = {"class_type": "ColorMatch", "inputs": {
        "image_ref": ["8016", 0], "image_target": ["8015", 0],
        "method": "mkl", "strength": 1.0, "multithread": True}}
    g["8017"]["inputs"]["image_a"] = ["8101", 0]
    g["8017"]["inputs"]["mask_low_res_dilation"] = 2
    g["4852"]["inputs"]["filename_prefix"] = f"{prefix}_s1"
    g["8020"]["inputs"]["filename_prefix"] = prefix
    return g

# ================= Phase A: single-pass base =================
before = set(os.listdir(COMFY_OUT))
gA = build_graph("drive_full.mp4", "drive_full_mask.mp4",
                 PERSONA.format(outfit="wearing stylish fitted clothes"), "v7_base")
pid = post(gA)["prompt_id"]
print("v7_base submitted", pid)
base_raw = wait_get(pid, "v7_base", before)
if not base_raw: sys.exit(2)
subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{3/30:.4f}", "-t", f"{241/30:.4f}",
                "-i", SRC, "-vn", "-c:a", "aac", "-b:a", "192k", f"{TMP}/audA.m4a"], check=True)
subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", base_raw, "-i", f"{TMP}/audA.m4a",
                "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "copy", "-shortest",
                f"{OUT}/v7_base.mp4"], check=True)
print("->", f"{OUT}/v7_base.mp4")

# ================= Phase B: outfit swap over v7_base =================
from rembg import remove, new_session
sess = new_session("u2net_human_seg")
kernel = np.ones((21, 21), np.uint8)
cap = cv2.VideoCapture(f"{OUT}/v7_base.mp4")
BASE = []
while True:
    ok, f = cap.read()
    if not ok: break
    BASE.append(f)
cap.release()
print("base frames:", len(BASE))
SEGS = [
    (131, 37, 41, "wearing a fitted white floral print midi dress with thin straps"),
    (168, 37, 41, "wearing a fitted black floral print midi dress with thin straps"),
    (205, 33, 33, "wearing a fitted white midi dress with bold red floral print"),
]
final = [f.copy() for f in BASE]
for k, (start, n_real, n_gen, outfit) in enumerate(SEGS, 2):
    frames = [cv2.resize(BASE[start + min(i, n_real - 1)], (W, H), interpolation=cv2.INTER_AREA)
              for i in range(n_gen)]
    for i, f in enumerate(frames):
        cv2.imwrite(f"{TMP}/s{k}_{i:05d}.png", f)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-framerate", "30", "-i", f"{TMP}/s{k}_%05d.png",
                    "-c:v", "libx264", "-crf", "10", "-pix_fmt", "yuv420p",
                    f"{IN_DIR}/v7drive{k}.mp4"], check=True)
    vw = cv2.VideoWriter(f"{TMP}/m{k}.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 30, (W, H))
    for f in frames:
        d = remove(Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)), session=sess)
        a = np.array(d)[:, :, 3]
        m = cv2.dilate((a > 30).astype(np.uint8) * 255, kernel, iterations=2)
        ys, xs = np.where(m > 127)
        if len(ys) > 50:
            ytop = ys.min()
            band = (ys >= ytop) & (ys < ytop + 110)
            cv2.ellipse(m, (int(xs[band].mean()), ytop + 62), (58, 78), 0, 0, 360, 0, -1)
        vw.write(cv2.cvtColor(cv2.GaussianBlur(m, (9, 9), 0), cv2.COLOR_GRAY2BGR))
    vw.release()
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", f"{TMP}/m{k}.mp4", "-c:v", "libx264",
                    "-crf", "6", "-pix_fmt", "yuv420p", f"{IN_DIR}/v7mask{k}.mp4"], check=True)
    before = set(os.listdir(COMFY_OUT))
    gk = build_graph(f"v7drive{k}.mp4", f"v7mask{k}.mp4", PERSONA.format(outfit=outfit), f"v7_seg{k}")
    pid = post(gk)["prompt_id"]
    print(f"v7_seg{k} submitted {pid}")
    v = wait_get(pid, f"v7_seg{k}", before)
    if not v: sys.exit(3)
    capg = cv2.VideoCapture(v); gen = []
    while True:
        ok, f = capg.read()
        if not ok: break
        gen.append(f)
    capg.release()
    th_, tw_ = BASE[0].shape[:2]
    for i in range(min(n_real, len(gen))):
        final[start + i] = cv2.resize(gen[i], (tw_, th_), interpolation=cv2.INTER_LANCZOS4)
    print(f"seg{k} spliced")

for i, f in enumerate(final):
    cv2.imwrite(f"{TMP}/out_{i:05d}.png", f)
subprocess.run(["ffmpeg", "-v", "error", "-y", "-framerate", "30", "-i", f"{TMP}/out_%05d.png",
                "-i", f"{OUT}/v7_base.mp4", "-map", "0:v", "-map", "1:a", "-c:v", "libx264",
                "-crf", "14", "-pix_fmt", "yuv420p", "-c:a", "copy", "-shortest",
                f"{OUT}/v7_outfits.mp4"], check=True)
print(f"FINAL -> {OUT}/v7_outfits.mp4")
print("V7_DONE")
