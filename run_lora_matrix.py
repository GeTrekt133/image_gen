#!/usr/bin/env python3
"""Ablate LoRA strengths (distilled / union / guide) on the bg-from-video setup.
Same anchor/prompt/seed; measures background temporal noise (wobble) + detail sharpness.
"""
import json,time,os,shutil,urllib.request,copy,subprocess
import numpy as np, cv2
HOST="http://127.0.0.1:10100"; COMFY_OUT="/workspace/ComfyUI/output"; OUT="/workspace/gallery/lora_matrix"
os.makedirs(OUT,exist_ok=True)
BASE=json.load(open("/workspace/wf_ltx_dance_notile.json"))
PROMPT=("photorealistic video of a beautiful young woman with wavy brown hair in a blue and white striped "
        "shirt dress dancing in a dim bedroom, pink and purple LED ambient wall lighting, dark furniture, "
        "moody indoor scene, cinematic, highly detailed, smooth natural motion, sharp focus")
def post(g): return json.load(urllib.request.urlopen(urllib.request.Request(HOST+"/prompt",data=json.dumps({"prompt":g}).encode(),headers={"Content-Type":"application/json"}),timeout=120))
def getj(p): return json.load(urllib.request.urlopen(HOST+p,timeout=60))

# (tag, distilled, union, guide)
CONFIGS=[
 ("base_d0.5_u1.0_g0.75", 0.5, 1.0, 0.75),
 ("guide0.5",             0.5, 1.0, 0.50),
 ("union0.7",             0.5, 0.7, 0.75),
 ("distill0.8",           0.8, 1.0, 0.75),
 ("distill0.35",          0.35,1.0, 0.75),
 ("smooth_d0.7_u0.7_g0.55",0.7,0.7, 0.55),
]
def metrics(path):
    cap=cv2.VideoCapture(path); prev=None; diffs=[]; sharp=[]
    while True:
        ok,f=cap.read()
        if not ok: break
        g=cv2.cvtColor(f,cv2.COLOR_BGR2GRAY).astype(np.float32)
        roi=np.concatenate([g[0:110,:].ravel(), g[110:430,0:70].ravel(), g[110:430,442:512].ravel()])
        if prev is not None: diffs.append(np.abs(roi-prev).mean())
        prev=roi
        sharp.append(cv2.Laplacian(g[300:750,150:380],cv2.CV_32F).var())  # dress/body region
    cap.release(); return np.mean(diffs), np.mean(sharp)

rows=[]
for tag,d,u,gd in CONFIGS:
    g=copy.deepcopy(BASE)
    g["5001"]["inputs"]["file"]="drive_dzp.mp4"
    g["2004"]["inputs"]["image"]="anchor_bgvideo.png"
    g["2483"]["inputs"]["text"]=PROMPT
    g["4922"]["inputs"]["strength_model"]=d
    g["5011"]["inputs"]["strength_model"]=u
    g["5012"]["inputs"]["strength"]=gd
    g["4832"]["inputs"]["noise_seed"]=42
    before=set(os.listdir(COMFY_OUT))
    t0=time.time(); pid=post(g)["prompt_id"]
    while time.time()-t0<1200:
        time.sleep(5)
        h=getj(f"/history/{pid}")
        if pid in h and (h[pid]["status"].get("completed") or h[pid]["status"].get("status_str") in ("success","error")):
            break
    new=[f for f in os.listdir(COMFY_OUT) if f.endswith(".mp4") and f not in before]
    if not new: print(f"{tag}: NO OUTPUT"); continue
    src=sorted(new,key=lambda f:os.path.getmtime(os.path.join(COMFY_OUT,f)))[-1]
    dst=f"{OUT}/{tag}.mp4"; shutil.copy(os.path.join(COMFY_OUT,src),dst)
    noise,sharp=metrics(dst)
    dt=time.time()-t0
    rows.append((tag,d,u,gd,noise,sharp,dt))
    print(f"{tag:26} d={d} u={u} g={gd}  bgNoise={noise:.3f}  sharp={sharp:6.0f}  {dt:.0f}s")

print("\n=== SUMMARY (lower bgNoise=steadier ; higher sharp=more detail) ===")
print(f"{'config':26} {'bgNoise':>8} {'sharp':>7}")
for r in rows: print(f"{r[0]:26} {r[4]:8.3f} {r[5]:7.0f}")
json.dump([{'tag':r[0],'distilled':r[1],'union':r[2],'guide':r[3],'bgNoise':r[4],'sharp':r[5],'sec':r[6]} for r in rows],
          open(f"{OUT}/results.json","w"),indent=1)
