#!/usr/bin/env python3
"""Hyperparameter sweep for bg-from-video motion transfer, MULTIPLE SEEDS per set
(stability check). Saves clearly-named clips + builds a grouped HTML gallery.
"""
import json,time,os,shutil,urllib.request,copy
import numpy as np, cv2
HOST="http://127.0.0.1:10100"; COMFY_OUT="/workspace/ComfyUI/output"
OUT="/workspace/gallery/sweep"; os.makedirs(OUT,exist_ok=True)
BASE=json.load(open("/workspace/wf_ltx_dance_notile.json"))
PROMPT=("photorealistic video of a beautiful young woman with wavy brown hair in a blue and white striped "
        "shirt dress dancing in a dim bedroom, pink and purple LED ambient wall lighting, dark furniture, "
        "moody indoor scene, cinematic, highly detailed, smooth natural motion, sharp focus")
SEEDS=[42,7,2026]
# id, human label, params
SETS=[
 ("A","distilled 0.35 · euler_ancestral  (резкий край)",      dict(distilled=0.35,union=1.0,guide=0.75,sampler="euler_ancestral")),
 ("B","distilled 0.50 · euler_ancestral  (старая база)",      dict(distilled=0.50,union=1.0,guide=0.75,sampler="euler_ancestral")),
 ("C","distilled 0.80 · euler_ancestral  (ТВОЙ ВЫБОР)",       dict(distilled=0.80,union=1.0,guide=0.75,sampler="euler_ancestral")),
 ("D","distilled 1.00 · euler_ancestral  (макс. гладкость)",  dict(distilled=1.00,union=1.0,guide=0.75,sampler="euler_ancestral")),
 ("E","distilled 0.80 · euler  (неанцестральный, глаже)",     dict(distilled=0.80,union=1.0,guide=0.75,sampler="euler")),
 ("F","distilled 0.65 · dpmpp_2m  (др. сэмплер)",             dict(distilled=0.65,union=1.0,guide=0.75,sampler="dpmpp_2m")),
 ("G","distilled 0.90 · euler_ancestral  (между C и D)",      dict(distilled=0.90,union=1.0,guide=0.75,sampler="euler_ancestral")),
]
def post(g): return json.load(urllib.request.urlopen(urllib.request.Request(HOST+"/prompt",data=json.dumps({"prompt":g}).encode(),headers={"Content-Type":"application/json"}),timeout=120))
def getj(p): return json.load(urllib.request.urlopen(HOST+p,timeout=60))
def metrics(path):
    cap=cv2.VideoCapture(path); prev=None; diffs=[]; sharp=[]
    while True:
        ok,f=cap.read()
        if not ok: break
        g=cv2.cvtColor(f,cv2.COLOR_BGR2GRAY).astype(np.float32)
        roi=np.concatenate([g[0:110,:].ravel(), g[110:430,0:70].ravel(), g[110:430,442:512].ravel()])
        if prev is not None: diffs.append(np.abs(roi-prev).mean())
        prev=roi
        sharp.append(cv2.Laplacian(g[300:750,150:380],cv2.CV_32F).var())
    cap.release(); return float(np.mean(diffs)), float(np.mean(sharp))

results=[]
for sid,label,p in SETS:
    for seed in SEEDS:
        g=copy.deepcopy(BASE)
        g["5001"]["inputs"]["file"]="drive_dzp.mp4"
        g["2004"]["inputs"]["image"]="anchor_bgvideo.png"
        g["2483"]["inputs"]["text"]=PROMPT
        g["4922"]["inputs"]["strength_model"]=p["distilled"]
        g["5011"]["inputs"]["strength_model"]=p["union"]
        g["5012"]["inputs"]["strength"]=p["guide"]
        g["4831"]["inputs"]["sampler_name"]=p["sampler"]
        g["4832"]["inputs"]["noise_seed"]=seed
        name=f"{sid}_dist{p['distilled']:.2f}_{p['sampler']}_seed{seed}"
        dst=f"{OUT}/{name}.mp4"
        if os.path.exists(dst):                       # idempotent: skip already-rendered
            n,s=metrics(dst); results.append(dict(set=sid,label=label,seed=seed,file=name+".mp4",bgNoise=n,sharp=s,sec=0,**p))
            print(f"{name:44} [skip, exists]"); continue
        g["4852"]["inputs"]["filename_prefix"]=name   # <-- readable output filename
        before=set(os.listdir(COMFY_OUT)); t0=time.time()
        try: pid=post(g)["prompt_id"]
        except Exception as e: print(f"{name} SUBMIT_FAIL {str(e)[:100]}"); continue
        ok=False
        while time.time()-t0<1200:
            time.sleep(4); h=getj(f"/history/{pid}")
            if pid in h:
                st=h[pid]["status"]
                if st.get("status_str")=="error": print(f"{name} ERROR"); break
                if st.get("completed") or st.get("status_str")=="success": ok=True; break
        if not ok: continue
        new=[f for f in os.listdir(COMFY_OUT) if f.endswith(".mp4") and f not in before]
        if not new: print(f"{name} NO_OUTPUT"); continue
        src=sorted(new,key=lambda f:os.path.getmtime(os.path.join(COMFY_OUT,f)))[-1]
        shutil.copy(os.path.join(COMFY_OUT,src),dst)
        n,s=metrics(dst); dt=time.time()-t0
        results.append(dict(set=sid,label=label,seed=seed,file=name+".mp4",bgNoise=n,sharp=s,sec=dt,**p))
        print(f"{name:44} bgNoise={n:.3f} sharp={s:6.0f} {dt:.0f}s")
json.dump(results,open(f"{OUT}/results.json","w"),indent=1)
print("SWEEP_DONE", len(results),"clips")
