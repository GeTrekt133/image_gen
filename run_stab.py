#!/usr/bin/env python3
"""A/B stabilizers on the VACE distilled+8step real-bg graph: STG, STG+StatNorm, AdaIN-to-bg."""
import json,urllib.request,time,os,copy,shutil
import numpy as np, cv2
HOST="http://127.0.0.1:10100"; COMFY_OUT="/workspace/ComfyUI/output"; OUT="/workspace/gallery/sweep"
G=json.load(open("/workspace/wf_vace_graft.json")); G["5001"]["inputs"]["file"]="drive_clean.mp4"
G["4922"]["inputs"]["strength_model"]=0.9   # distilled kept
PROMPT=("a beautiful young woman with long wavy brown hair wearing a light blue and white vertical pinstripe "
        "shirt dress, dancing, pink and purple ambient room light, photorealistic, highly detailed, natural skin")
for n in ["2483"]: G[n]["inputs"]["text"]=PROMPT
def post(g): return json.load(urllib.request.urlopen(urllib.request.Request(HOST+"/prompt",data=json.dumps({"prompt":g}).encode(),headers={"Content-Type":"application/json"}),timeout=120))
def getj(p): return json.load(urllib.request.urlopen(HOST+p,timeout=60))
def subj_noise(path):
    cap=cv2.VideoCapture(path); prev=None; ds=[]
    while True:
        ok,f=cap.read()
        if not ok: break
        g=cv2.cvtColor(cv2.resize(f,(512,896)),cv2.COLOR_BGR2GRAY).astype(np.float32)[120:820,150:380]
        if prev is not None: ds.append(np.abs(g-prev).mean())
        prev=g
    cap.release(); return float(np.mean(ds))

def build(kind):
    g=copy.deepcopy(G)
    if kind=="stg":
        g["7202"]={"class_type":"LTXVApplySTG","inputs":{"model":["5011",0],"block_indices":"14, 19"}}
        g["4828"]={"class_type":"STGGuider","inputs":{"model":["7202",0],"positive":["5012",0],"negative":["5012",1],"cfg":1.0,"stg":1.0,"rescale":0.7}}
    elif kind=="stg_statnorm":
        g["7200"]={"class_type":"LTXVPerStepStatNormPatcher","inputs":{"model":["5011",0],"factors":"0.9, 0.75, 0.0","target_mean":0.0,"target_std":1.0,"percentile":95.0,"clip_outliers":False}}
        g["7202"]={"class_type":"LTXVApplySTG","inputs":{"model":["7200",0],"block_indices":"14, 19"}}
        g["4828"]={"class_type":"STGGuider","inputs":{"model":["7202",0],"positive":["5012",0],"negative":["5012",1],"cfg":1.0,"stg":1.0,"rescale":0.7}}
    elif kind=="adain_bg":
        # normalize latent toward the real encoded background each step
        g["7201"]={"class_type":"LTXVPerStepAdainPatcher","inputs":{"model":["5011",0],"factors":"0.9, 0.75, 0.0","reference":["7010",0],"per_frame":False}}
        g["4828"]["inputs"]["model"]=["7201",0]
    return g

CONFIGS=["stg","stg_statnorm"]
rows=[]
for kind in CONFIGS:
    g=copy.deepcopy(G) if kind=="baseline" else build(kind)
    g["4832"]["inputs"]["noise_seed"]=42
    g["4852"]["inputs"]["filename_prefix"]=f"STAB_{kind}"
    before=set(os.listdir(COMFY_OUT)); t0=time.time()
    try: pid=post(g)["prompt_id"]
    except urllib.error.HTTPError as e: print(kind,"SUBMIT_ERR",json.loads(e.read()).get("error",{}).get("message")[:150]); continue
    ok=False
    while time.time()-t0<900:
        time.sleep(4); h=getj(f"/history/{pid}")
        if pid in h and (h[pid]["status"].get("completed") or h[pid]["status"].get("status_str") in ("success","error")):
            if h[pid]["status"].get("status_str")=="error": print(kind,"ERR",[str(m)[:180] for m in h[pid]["status"].get("messages",[]) if 'error' in str(m).lower()][:1])
            else: ok=True
            break
    if not ok: continue
    new=[f for f in os.listdir(COMFY_OUT) if f.endswith(".mp4") and f not in before]
    src=sorted(new,key=lambda f:os.path.getmtime(os.path.join(COMFY_OUT,f)))[-1]
    shutil.copy(os.path.join(COMFY_OUT,src),f"{OUT}/STAB_{kind}.mp4")
    sn=subj_noise(f"{OUT}/STAB_{kind}.mp4"); dt=time.time()-t0
    rows.append((kind,sn,dt)); print(f"STAB_{kind:14} subjNoise={sn:.2f}  {dt:.0f}s")
print("\n=== SUMMARY (subjNoise lower = steadier subject) ===")
for k,sn,dt in rows: print(f"  {k:14} {sn:.2f}  ({dt:.0f}s)")
print("STAB_DONE")
