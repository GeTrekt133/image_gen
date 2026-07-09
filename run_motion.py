#!/usr/bin/env python3
"""Run LTX-2.3 motion-transfer from wf_ltx_dance.json with parametric anchor/driving/prompt.
Usage: run_motion.py --anchor persona_ref.png --drive drive_dzp.mp4 --tag baseline \
       [--prompt "..."] [--union 1.0] [--guide 0.75] [--seed 42]
Anchor/drive are filenames inside ComfyUI/input.
"""
import json, time, os, shutil, urllib.request, copy, argparse, sys
HOST="http://127.0.0.1:10100"; COMFY_OUT="/workspace/ComfyUI/output"; OUT="/workspace/gallery/motion_out"
os.makedirs(OUT, exist_ok=True)
ap=argparse.ArgumentParser()
ap.add_argument("--anchor", required=True)
ap.add_argument("--drive", required=True)
ap.add_argument("--tag", required=True)
ap.add_argument("--prompt", default="photorealistic video of a beautiful young woman with wavy brown hair in a blue striped shirt dress dancing, moving to music, pink and purple ambient bedroom lighting, cinematic, highly detailed, smooth natural motion, sharp focus")
ap.add_argument("--union", type=float, default=1.0)
ap.add_argument("--guide", type=float, default=0.75)
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--distilled", type=float, default=0.35)  # ablation: 0.35 > 0.5 on both steadiness+detail
ap.add_argument("--workflow", default="/workspace/wf_ltx_dance_pruned.json")
a=ap.parse_args()

BASE=json.load(open(a.workflow))
def post(g): return json.load(urllib.request.urlopen(urllib.request.Request(HOST+"/prompt",data=json.dumps({"prompt":g}).encode(),headers={"Content-Type":"application/json"}),timeout=120))
def getj(p): return json.load(urllib.request.urlopen(HOST+p,timeout=60))

g=copy.deepcopy(BASE)
g["5001"]["inputs"]["file"]=a.drive          # LoadVideo (driving)
g["2004"]["inputs"]["image"]=a.anchor        # LoadImage (persona/composite anchor)
g["2483"]["inputs"]["text"]=a.prompt         # positive
g["5011"]["inputs"]["strength_model"]=a.union
g["5012"]["inputs"]["strength"]=a.guide
g["4922"]["inputs"]["strength_model"]=a.distilled
g["4832"]["inputs"]["noise_seed"]=a.seed
before=set(os.listdir(COMFY_OUT)) if os.path.isdir(COMFY_OUT) else set()
t0=time.time()
try: pid=post(g)["prompt_id"]
except Exception as e:
    print("SUBMIT_FAIL", str(e)[:400]); sys.exit(1)
print(f"submitted {pid} tag={a.tag} union={a.union} guide={a.guide} seed={a.seed}")
err=None
while time.time()-t0<1800:
    time.sleep(6)
    try: h=getj(f"/history/{pid}")
    except Exception: continue
    if pid in h:
        st=h[pid]["status"]
        if st.get("status_str")=="error":
            msgs=[m for m in st.get("messages",[]) if 'error' in str(m).lower()]
            err=msgs[:2]; break
        if st.get("status_str")=="success" or st.get("completed"): break
    sys.stdout.write("."); sys.stdout.flush()
if err:
    print("\nERROR:", json.dumps(err)[:1500]); sys.exit(2)
new=[f for f in os.listdir(COMFY_OUT) if f.endswith(".mp4") and f not in before]
new=sorted(new, key=lambda f:os.path.getmtime(os.path.join(COMFY_OUT,f)))
if not new:
    print("\nNO_OUTPUT_MP4 (check comfy.log)"); sys.exit(3)
dst=f"{OUT}/{a.tag}.mp4"
shutil.copy(os.path.join(COMFY_OUT,new[-1]), dst)
print(f"\nOK {time.time()-t0:.0f}s -> {dst}  (src {new[-1]})")
