#!/usr/bin/env python3
"""v2e = v2d (pro stage-1, hires, two-stage, real bg) + temporally SMOOTHED DWPose.
Steps: 1) export DWPose keypoints via ComfyUI (768x1344, same geometry as v2d)
       2) Savitzky-Golay smoothing per keypoint track (face 9, hands 7, body 5)
       3) re-render skeleton video with the same controlnet_aux renderer
       4) run the v2d graph with the smoothed pose video as the union-control guide.
Existing files untouched."""
import json, time, os, shutil, urllib.request, sys, copy, subprocess
import numpy as np

HOST = "http://127.0.0.1:10100"
COMFY_OUT = "/workspace/ComfyUI/output"
COMFY_IN = "/workspace/ComfyUI/input"
OUT = "/workspace/gallery/sweep"
SCRATCH = "/tmp/claude-0/-workspace/a3dbfea7-d832-47e1-8bc4-d50747ea681e/scratchpad/pose_frames"

def post(gr):
    req = urllib.request.Request(HOST + "/prompt", data=json.dumps({"prompt": gr}).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=120))

def getj(p):
    return json.load(urllib.request.urlopen(HOST + p, timeout=60))

def wait(pid, timeout=3600, tag=""):
    t0 = time.time()
    while time.time() - t0 < timeout:
        time.sleep(5)
        try:
            h = getj(f"/history/{pid}")
        except Exception:
            continue
        if pid in h:
            st = h[pid]["status"]
            if st.get("status_str") == "error":
                print(tag, "RUN_ERR", [str(m)[:300] for m in st.get("messages", []) if "error" in str(m).lower()][:2])
                return False
            if st.get("status_str") == "success" or st.get("completed"):
                return True
    print(tag, "TIMEOUT"); return False

# ---------- 1) export keypoints at v2d geometry (768x1344) ----------
kps_graph = {
 "1": {"class_type": "LoadVideo", "inputs": {"file": "drive_clean.mp4"}},
 "2": {"class_type": "GetVideoComponents", "inputs": {"video": ["1", 0]}},
 "3": {"class_type": "ResizeImageMaskNode", "inputs": {"input": ["2", 0],
       "resize_type": "scale shorter dimension", "resize_type.shorter_size": 768,
       "scale_method": "lanczos"}},
 "4": {"class_type": "DWPreprocessor", "inputs": {"image": ["3", 0],
       "detect_hand": "enable", "detect_body": "enable", "detect_face": "enable",
       "resolution": 512, "bbox_detector": "yolox_l.torchscript.pt",
       "pose_estimator": "dw-ll_ucoco_384_bs5.torchscript.pt",
       "scale_stick_for_xinsr_cn": "disable"}},
 "5": {"class_type": "SavePoseKpsAsJsonFile", "inputs": {"pose_kps": ["4", 1],
       "filename_prefix": "posekps"}},
}
before = set(os.listdir(COMFY_OUT))
pid = post(kps_graph)["prompt_id"]
print("kps export submitted", pid)
if not wait(pid, 1800, "kps"): sys.exit(1)
jsons = sorted(f for f in os.listdir(COMFY_OUT) if f.startswith("posekps") and f.endswith(".json") and f not in before)
if not jsons: print("NO_KPS_JSON"); sys.exit(1)
KPS = json.load(open(os.path.join(COMFY_OUT, jsons[-1])))
print("frames in kps:", len(KPS))

# ---------- 2) smooth ----------
from scipy.signal import savgol_filter

def tracks(get_flat, n_pts):
    """(F, n_pts, 3) array from per-frame flat lists; NaN when missing."""
    F = len(KPS)
    arr = np.full((F, n_pts, 3), np.nan, dtype=np.float64)
    for i, fr in enumerate(KPS):
        ppl = fr.get("people") or []
        if not ppl: continue
        flat = get_flat(ppl[0])
        if not flat: continue
        a = np.asarray(flat, dtype=np.float64).reshape(-1, 3)[:n_pts]
        arr[i, :a.shape[0]] = a
    return arr

def smooth(arr, window, conf_thr=0.15):
    F, N, _ = arr.shape
    out = arr.copy()
    t = np.arange(F)
    for k in range(N):
        x, y, c = arr[:, k, 0], arr[:, k, 1], arr[:, k, 2]
        good = np.isfinite(x) & np.isfinite(c) & (c > conf_thr)
        if good.sum() < window:  # too sparse to smooth
            continue
        for j, series in ((0, x), (1, y)):
            filled = np.interp(t, t[good], series[good])   # bridge dropouts
            w = min(window if window % 2 == 1 else window + 1, len(t) - (1 - len(t) % 2))
            sm = savgol_filter(filled, w, 2)
            out[:, k, j] = np.where(good, sm, filled)      # smoothed everywhere
    return out

parts = {
    "pose_keypoints_2d": (18, 5),
    "face_keypoints_2d": (70, 9),
    "hand_left_keypoints_2d": (21, 7),
    "hand_right_keypoints_2d": (21, 7),
}
smoothed = {}
for key, (n, w) in parts.items():
    smoothed[key] = smooth(tracks(lambda p, k=key: p.get(k), n), w)

for i, fr in enumerate(KPS):
    ppl = fr.get("people") or []
    if not ppl: continue
    for key, (n, _) in parts.items():
        if not ppl[0].get(key): continue
        a = smoothed[key][i]
        orig = np.asarray(ppl[0][key], dtype=np.float64).reshape(-1, 3)
        m = min(len(orig), n)
        merged = orig.copy()
        fin = np.isfinite(a[:m]).all(axis=1)
        merged[:m][fin] = a[:m][fin]
        ppl[0][key] = [float(v) for v in merged.reshape(-1)]
print("smoothing done")

# ---------- 3) re-render skeleton ----------
sys.path.insert(0, "/workspace/ComfyUI/custom_nodes/comfyui_controlnet_aux/src")
os.environ.setdefault("AUX_TEMP_DIR", "/tmp")
from custom_controlnet_aux.dwpose import decode_json_as_poses, draw_poses
import cv2
os.makedirs(SCRATCH, exist_ok=True)
H = W = None
for i, fr in enumerate(KPS):
    poses, _, h, w = decode_json_as_poses(fr)
    H, W = h, w
    img = draw_poses(poses, h, w, True, True, True)
    cv2.imwrite(f"{SCRATCH}/f_{i:05d}.png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
print(f"rendered {len(KPS)} frames at {W}x{H}")
dst = os.path.join(COMFY_IN, "drive_pose_smooth.mp4")
subprocess.run(["ffmpeg", "-v", "error", "-y", "-framerate", "24",
                "-i", f"{SCRATCH}/f_%05d.png", "-c:v", "libx264",
                "-crf", "10", "-pix_fmt", "yuv420p", dst], check=True)
print("pose video ->", dst)

# ---------- 4) v2d graph with smoothed pose ----------
g = json.load(open("/workspace/wf_vace_v2.json"))
# hires
g["9000"] = {"class_type": "ResizeImageMaskNode", "inputs": {
    "input": ["5000", 0], "resize_type": "scale by multiplier",
    "resize_type.multiplier": 1.5, "scale_method": "lanczos"}}
g["7010"]["inputs"]["pixels"] = ["9000", 0]
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
# pro stage-1
g["4922"]["inputs"]["strength_model"] = 0.0
g["7100"] = {"class_type": "LTXVScheduler", "inputs": {
    "steps": 30, "max_shift": 2.05, "base_shift": 0.95, "stretch": True, "terminal": 0.1}}
g["4829"]["inputs"]["sigmas"] = ["7100", 0]
del g["5025"]
g["4828"]["inputs"]["cfg"] = 3.0
g["4831"]["inputs"]["sampler_name"] = "euler_ancestral"
# stage-2 distilled chain
g["9100"] = {"class_type": "LoraLoaderModelOnly", "inputs": {
    "model": ["3940", 0],
    "lora_name": "ltxv/ltx2/ltx-2.3-22b-distilled-lora-384-1.1.safetensors",
    "strength_model": 0.5}}
g["9101"] = {"class_type": "LTXICLoRALoaderModelOnly", "inputs": {
    "model": ["9100", 0],
    "lora_name": "ltxv/ltx2/ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors",
    "strength_model": 1.0}}
g["8012"]["inputs"]["model"] = ["9101", 0]
# smoothed pose replaces DWPose chain
g["5027"] = {"class_type": "LoadVideo", "inputs": {"file": "drive_pose_smooth.mp4"}}
g["5030"] = {"class_type": "GetVideoComponents", "inputs": {"video": ["5027", 0]}}
g["5028"]["inputs"]["input"] = ["5030", 0]
del g["4991"]; del g["5026"]
g["4852"]["inputs"]["filename_prefix"] = "v2e_s1"
g["8020"]["inputs"]["filename_prefix"] = "v2e_smooth"

before = set(os.listdir(COMFY_OUT)); t0 = time.time()
try:
    pid = post(g)["prompt_id"]
except urllib.error.HTTPError as e:
    print("SUBMIT_FAIL", e.read().decode()[:1200]); sys.exit(1)
print("v2e submitted", pid)
if not wait(pid, 3600, "v2e"): sys.exit(2)
for f in sorted(os.listdir(COMFY_OUT)):
    if f.endswith(".mp4") and f not in before and f.startswith("v2e_smooth"):
        shutil.copy(os.path.join(COMFY_OUT, f), f"{OUT}/v2e_smooth.mp4")
        print(f"OK {time.time()-t0:.0f}s -> {OUT}/v2e_smooth.mp4")
# also drop the smoothed pose render into the gallery for inspection
shutil.copy(dst, f"{OUT}/dwpose_smooth.mp4")
print("V2E_DONE")
