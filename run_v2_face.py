#!/usr/bin/env python3
"""Face-drift A/B on top of wf_vace_v2.json (patched in-memory, files untouched).
v2b_noface : DWPose detect_face=disable -> face not driven by jittery per-frame
             face keypoints; model animates the face itself (temporal attention).
v2c_hires  : stage-1 at 1.5x (768x1344), stage-2 x1.5 -> 1152x2016 final.
             More pixels on the face in stage-1 = less per-frame hallucination.
             (also keeps detect_face disabled if it wins? no - single variable:
              v2c keeps detect_face ENABLED, only resolution changes)
"""
import json, time, os, shutil, urllib.request, sys, copy
import numpy as np, cv2

HOST = "http://127.0.0.1:10100"
COMFY_OUT = "/workspace/ComfyUI/output"
OUT = "/workspace/gallery/sweep"
os.makedirs(OUT, exist_ok=True)
BASE = json.load(open("/workspace/wf_vace_v2.json"))

def post(gr):
    req = urllib.request.Request(HOST + "/prompt", data=json.dumps({"prompt": gr}).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=120))

def getj(p):
    return json.load(urllib.request.urlopen(HOST + p, timeout=60))

def build_noface():
    g = copy.deepcopy(BASE)
    g["4991"]["inputs"]["detect_face"] = "disable"
    g["4852"]["inputs"]["filename_prefix"] = "v2b_s1"
    g["8020"]["inputs"]["filename_prefix"] = "v2b_noface"
    return g

def build_hires():
    g = copy.deepcopy(BASE)
    # upscale driving frames x1.5 -> 768x1344, everything derives from it
    g["9000"] = {"class_type": "ResizeImageMaskNode", "inputs": {
        "input": ["5000", 0], "resize_type": "scale by multiplier",
        "resize_type.multiplier": 1.5, "scale_method": "lanczos"}}
    g["7010"]["inputs"]["pixels"] = ["9000", 0]
    g["5026"]["inputs"]["input"] = ["9000", 0]
    g["5026"]["inputs"]["resize_type.shorter_size"] = 768
    # mask x1.5
    g["9001"] = {"class_type": "ResizeImageMaskNode", "inputs": {
        "input": ["7003", 0], "resize_type": "scale by multiplier",
        "resize_type.multiplier": 1.5, "scale_method": "lanczos"}}
    g["7004"]["inputs"]["image"] = ["9001", 0]
    # stage-1 blend against upscaled real frames
    g["8000"]["inputs"]["image_b"] = ["9000", 0]
    # stage-2: x1.5 on top of 768p -> 1152x2016
    g["8001"]["inputs"]["resize_type.multiplier"] = 1.5
    g["8003"]["inputs"]["input"] = ["9001", 0]
    g["8003"]["inputs"]["resize_type.multiplier"] = 1.5
    g["8016"]["inputs"]["input"] = ["9000", 0]
    g["8016"]["inputs"]["resize_type.multiplier"] = 1.5
    g["4852"]["inputs"]["filename_prefix"] = "v2c_s1"
    g["8020"]["inputs"]["filename_prefix"] = "v2c_hires"
    return g

def run(tag, g, timeout=3600):
    before = set(os.listdir(COMFY_OUT))
    t0 = time.time()
    try:
        pid = post(g)["prompt_id"]
    except urllib.error.HTTPError as e:
        print(tag, "SUBMIT_FAIL", e.read().decode()[:800]); return None
    print(f"{tag} submitted {pid}")
    while time.time() - t0 < timeout:
        time.sleep(6)
        try:
            h = getj(f"/history/{pid}")
        except Exception:
            continue
        if pid in h:
            st = h[pid]["status"]
            if st.get("status_str") == "error":
                msgs = [str(m)[:300] for m in st.get("messages", []) if "error" in str(m).lower()]
                print(tag, "RUN_ERR", msgs[:1]); return None
            if st.get("status_str") == "success" or st.get("completed"):
                break
    new = [f for f in os.listdir(COMFY_OUT) if f.endswith(".mp4") and f not in before]
    got = None
    for f in sorted(new):
        if f.startswith(("v2b_noface", "v2c_hires")):
            got = f"{OUT}/{f.split('_0000')[0]}.mp4"
            shutil.copy(os.path.join(COMFY_OUT, f), got)
    print(f"{tag} OK {time.time()-t0:.0f}s -> {got}")
    return got

def metrics(path):
    cap = cv2.VideoCapture(path); prev = None; diffs = []; sharp = []
    while True:
        ok, f = cap.read()
        if not ok: break
        f = cv2.resize(f, (512, 896))
        gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32)
        if prev is not None: diffs.append(np.abs(gray - prev).mean())
        prev = gray
        sharp.append(cv2.Laplacian(gray[300:750, 150:380], cv2.CV_32F).var())
    cap.release()
    return float(np.mean(diffs)), float(np.mean(sharp))

results = []
for tag, g in [("v2b_noface", build_noface()), ("v2c_hires", build_hires())]:
    p = run(tag, g)
    if p:
        tn, sh = metrics(p)
        results.append((tag, tn, sh))
        print(f"{tag}: tNoise={tn:.2f} sharp={sh:.0f}")
print("\n=== SUMMARY vs v2_stage2 tNoise=8.60 (real footage=8.30) ===")
for t, tn, sh in results: print(f"  {t:12} tNoise={tn:.2f} sharp={sh:.0f}")
print("FACE_AB_DONE")
