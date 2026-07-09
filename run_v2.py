#!/usr/bin/env python3
"""V2 fork of the VACE motion-control pipeline (audit fixes, 2026-07-09).
Fixes vs wf_vace_graft.json: aligned guide/latent geometry (512x896 everywhere),
clamp_min=0 (real bg kept), euler_ancestral_cfg_pp, distilled 0.5, guide 1.0,
style-only prompt, official negative, stage-2 x2 upscale + refine tail
(0.725/0.4219/0) + Laplacian pyramid blend of real bg at both stages.
Does not modify any existing workflow/script.
"""
import json, time, os, shutil, urllib.request, sys
import numpy as np, cv2

HOST = "http://127.0.0.1:10100"
COMFY_OUT = "/workspace/ComfyUI/output"
OUT = "/workspace/gallery/sweep"
os.makedirs(OUT, exist_ok=True)

g = json.load(open("/workspace/wf_vace_v2.json"))

def post(gr):
    req = urllib.request.Request(HOST + "/prompt", data=json.dumps({"prompt": gr}).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=120))

def getj(p):
    return json.load(urllib.request.urlopen(HOST + p, timeout=60))

def metrics(path):
    """Frame-to-frame |delta| (temporal noise, lower=steadier) + Laplacian var
    (detail, higher=sharper), computed on frames normalized to 512x896."""
    cap = cv2.VideoCapture(path); prev = None; diffs = []; sharp = []
    bg_diffs = []
    while True:
        ok, f = cap.read()
        if not ok: break
        f = cv2.resize(f, (512, 896))
        gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32)
        if prev is not None:
            diffs.append(np.abs(gray - prev).mean())
            bg_diffs.append(np.abs(gray[:, 400:] - prev[:, 400:]).mean())  # right bg strip
        prev = gray
        sharp.append(cv2.Laplacian(gray[300:750, 150:380], cv2.CV_32F).var())
    cap.release()
    return float(np.mean(diffs)), float(np.mean(bg_diffs)), float(np.mean(sharp))

before = set(os.listdir(COMFY_OUT)) if os.path.isdir(COMFY_OUT) else set()
t0 = time.time()
try:
    pid = post(g)["prompt_id"]
except urllib.error.HTTPError as e:
    print("SUBMIT_FAIL", e.read().decode()[:1200]); sys.exit(1)
except Exception as e:
    print("SUBMIT_FAIL", str(e)[:400]); sys.exit(1)
print(f"submitted {pid}")

err = None
while time.time() - t0 < 3600:
    time.sleep(6)
    try:
        h = getj(f"/history/{pid}")
    except Exception:
        continue
    if pid in h:
        st = h[pid]["status"]
        if st.get("status_str") == "error":
            msgs = [m for m in st.get("messages", []) if "error" in str(m).lower()]
            err = msgs[:2]; break
        if st.get("status_str") == "success" or st.get("completed"):
            break
    sys.stdout.write("."); sys.stdout.flush()

if err:
    print("\nERROR:", json.dumps(err)[:2000]); sys.exit(2)

new = [f for f in os.listdir(COMFY_OUT) if f.endswith(".mp4") and f not in before]
if not new:
    print("\nNO_OUTPUT_MP4 (check comfy.log)"); sys.exit(3)
print(f"\ndone in {time.time()-t0:.0f}s")

copied = []
for f in sorted(new):
    for tag in ("v2_stage1", "v2_stage2"):
        if f.startswith(tag):
            dst = f"{OUT}/{tag}.mp4"
            shutil.copy(os.path.join(COMFY_OUT, f), dst)
            copied.append(dst)
            print("saved", dst, f"(src {f})")

print("\n=== METRICS (normalized 512x896; tNoise/bgNoise lower=steadier, sharp higher=more detail) ===")
refs = ["/workspace/gallery/sweep/VP_cur_clamp0.5.mp4", "/workspace/ComfyUI/input/drive_clean.mp4"]
for p in refs + copied:
    if os.path.isfile(p):
        tn, bn, sh = metrics(p)
        print(f"  {os.path.basename(p):24} tNoise={tn:6.2f}  bgNoise={bn:6.2f}  sharp={sh:7.0f}")
print("V2_DONE")
