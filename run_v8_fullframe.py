#!/usr/bin/env python3
"""v8: NO partial generation — full-frame pose-transfer over the whole reel.
No noise masks, no Laplacian blends -> the halo cannot exist by construction.
Background is fully generated but anchored: i2v anchor = frame 0 of
v5_singlepass (persona already in the real room, pose-aligned with the driving
video's first frame), LTXVPreprocess(18), strength 0.7 (official V2V numbers).
Recipe fixes applied: union 1.0, guide 1.0, distilled 0.5,
euler_ancestral_cfg_pp, official negative, style-only prompt, native 30 fps,
plus an unmasked stage-2 (x1.5 -> 1152x2016, tail 0.725/0.4219/0, euler_cfg_pp).
Base graph: wf_ltx_dance_notile.json (has the i2v anchor path)."""
import json, time, os, shutil, urllib.request, sys, subprocess
import cv2

IN_DIR = "/workspace/ComfyUI/input"
COMFY_OUT = "/workspace/ComfyUI/output"
OUT = "/workspace/gallery/sweep"
SRC = "/workspace/test_reels/DZLY571hMqd.mp4"
TMP = "/tmp/claude-0/-workspace/a3dbfea7-d832-47e1-8bc4-d50747ea681e/scratchpad"
HOST = "http://127.0.0.1:10100"

# anchor = v5 frame 0 (persona in the room, first-frame pose)
cap = cv2.VideoCapture(f"{OUT}/v5_singlepass.mp4")
ok, f0 = cap.read(); cap.release()
assert ok
cv2.imwrite(f"{IN_DIR}/v8_anchor.png", f0)

g = json.load(open("/workspace/wf_ltx_dance_notile.json"))
g["5001"]["inputs"]["file"] = "drive_full.mp4"
g["2004"]["inputs"]["image"] = "v8_anchor.png"
g["2483"]["inputs"]["text"] = (
    "a beautiful young woman with long wavy brown hair wearing a fitted white cutout "
    "crop top and white shorts, in a minimalist beige room, photorealistic, natural "
    "skin texture, fine fabric detail, highly detailed")
g["2612"]["inputs"]["text"] = "pc game, console game, video game, cartoon, childish, ugly"
g["5026"]["inputs"]["resize_type.shorter_size"] = 768
g["4991"]["inputs"]["resolution"] = 768  # DWPose output sets the latent size in this graph
g["4831"]["inputs"]["sampler_name"] = "euler_ancestral_cfg_pp"
g["4922"]["inputs"]["strength_model"] = 0.5
g["5011"]["inputs"]["strength_model"] = 1.0
g["5012"]["inputs"]["strength"] = 1.0
g["3159"]["inputs"]["strength"] = 0.7
# official V2V preprocesses the anchor image
g["5107"] = {"class_type": "LTXVPreprocess", "inputs": {"image": ["5035", 0], "img_compression": 18}}
g["3159"]["inputs"]["image"] = ["5107", 0]
g["4852"]["inputs"]["filename_prefix"] = "v8_s1"

# ---- unmasked stage-2: x1.5 upscale + refine tail ----
g["8200"] = {"class_type": "ResizeImageMaskNode", "inputs": {
    "input": ["5065", 0], "resize_type": "scale by multiplier",
    "resize_type.multiplier": 1.5, "scale_method": "lanczos"}}
g["8201"] = {"class_type": "VAEEncodeTiled", "inputs": {
    "pixels": ["8200", 0], "vae": ["3940", 2],
    "tile_size": 512, "overlap": 64, "temporal_size": 500, "temporal_overlap": 64}}
g["8202"] = {"class_type": "LTXVConcatAVLatent", "inputs": {
    "video_latent": ["8201", 0], "audio_latent": ["4845", 1]}}
g["8203"] = {"class_type": "RandomNoise", "inputs": {"noise_seed": 43}}
g["8204"] = {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler_cfg_pp"}}
g["8205"] = {"class_type": "ManualSigmas", "inputs": {"sigmas": "0.7250, 0.4219, 0.0"}}
g["8206"] = {"class_type": "CFGGuider", "inputs": {
    "model": ["5011", 0], "positive": ["5013", 0], "negative": ["5013", 1], "cfg": 1}}
g["8207"] = {"class_type": "SamplerCustomAdvanced", "inputs": {
    "noise": ["8203", 0], "guider": ["8206", 0], "sampler": ["8204", 0],
    "sigmas": ["8205", 0], "latent_image": ["8202", 0]}}
g["8208"] = {"class_type": "LTXVSeparateAVLatent", "inputs": {"av_latent": ["8207", 0]}}
g["8209"] = {"class_type": "LTXVTiledVAEDecode", "inputs": {
    "vae": ["3940", 2], "latents": ["8208", 0], "horizontal_tiles": 2, "vertical_tiles": 2,
    "overlap": 6, "last_frame_fix": False, "working_device": "auto", "working_dtype": "auto"}}
g["8210"] = {"class_type": "LTXVAudioVAEDecode", "inputs": {
    "samples": ["8208", 1], "audio_vae": ["4010", 0]}}
g["8211"] = {"class_type": "CreateVideo", "inputs": {
    "images": ["8209", 0], "fps": ["5000", 2], "audio": ["8210", 0]}}
g["8212"] = {"class_type": "SaveVideo", "inputs": {
    "video": ["8211", 0], "filename_prefix": "v8_full", "format": "mp4", "codec": "h264"}}

def post(gr):
    req = urllib.request.Request(HOST + "/prompt", data=json.dumps({"prompt": gr}).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=120))

def getj(p):
    return json.load(urllib.request.urlopen(HOST + p, timeout=60))

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
    if f.endswith(".mp4") and f not in before and f.startswith("v8_full"):
        got = os.path.join(COMFY_OUT, f)
if not got:
    print("NO_OUTPUT"); sys.exit(3)
print(f"gen OK {time.time()-t0:.0f}s")
subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{3/30:.4f}", "-t", f"{241/30:.4f}",
                "-i", SRC, "-vn", "-c:a", "aac", "-b:a", "192k", f"{TMP}/v8aud.m4a"], check=True)
subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", got, "-i", f"{TMP}/v8aud.m4a",
                "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "copy", "-shortest",
                f"{OUT}/v8_fullframe.mp4"], check=True)
print(f"-> {OUT}/v8_fullframe.mp4")
print("V8_DONE")
