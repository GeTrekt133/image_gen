#!/usr/bin/env python3
"""Ablate 'less distilled + more steps' on the VACE real-bg graph.
Replaces the fixed 8-step ManualSigmas with LTXVScheduler(steps). Measures time + peak VRAM.
"""
import json,urllib.request,time,os,subprocess,copy
import numpy as np, cv2
HOST="http://127.0.0.1:10100"; COMFY_OUT="/workspace/ComfyUI/output"
OUT="/workspace/gallery/sweep"; os.makedirs(OUT,exist_ok=True)
G=json.load(open("/workspace/wf_vace_graft.json"))
G["5001"]["inputs"]["file"]="drive_clean.mp4"
PROMPT=("a beautiful young woman with long wavy brown hair wearing a light blue and white vertical pinstripe "
        "shirt dress, dancing, pink and purple ambient room light, photorealistic, highly detailed, natural skin")
# (tag, distilled_strength, steps)
CONFIGS=[
 ("PRO_sched30_distill0.0", 0.0, 30),   # user's idea: no distillation, many steps
 ("PRO_sched30_distill0.3", 0.3, 30),
 ("PRO_sched40_distill0.2", 0.2, 40),
 ("PRO_sched20_distill0.5", 0.5, 20),
]
def post(g): return json.load(urllib.request.urlopen(urllib.request.Request(HOST+"/prompt",data=json.dumps({"prompt":g}).encode(),headers={"Content-Type":"application/json"}),timeout=120))
def getj(p): return json.load(urllib.request.urlopen(HOST+p,timeout=60))
def metrics(path):
    cap=cv2.VideoCapture(path); prev=None; diffs=[]; sharp=[]
    while True:
        ok,f=cap.read()
        if not ok: break
        g=cv2.cvtColor(f,cv2.COLOR_BGR2GRAY).astype(np.float32)
        if prev is not None: diffs.append(np.abs(g-prev).mean())
        prev=g
        sharp.append(cv2.Laplacian(g[300:750,150:380],cv2.CV_32F).var())
    cap.release(); return float(np.mean(diffs)),float(np.mean(sharp))
def vram_now():
    try: return int(subprocess.check_output(["nvidia-smi","--query-gpu=memory.used","--format=csv,noheader,nounits"]).decode().split("\n")[0])
    except: return 0
rows=[]
for tag,dist,steps in CONFIGS:
    g=copy.deepcopy(G)
    g["4922"]["inputs"]["strength_model"]=dist
    # replace ManualSigmas(5025) -> LTXVScheduler(steps)
    g["7100"]={"class_type":"LTXVScheduler","inputs":{"steps":steps,"max_shift":2.05,"base_shift":0.95,"stretch":True,"terminal":0.1}}
    g["4829"]["inputs"]["sigmas"]=["7100",0]
    if "5025" in g: del g["5025"]
    g["4832"]["inputs"]["noise_seed"]=42
    g["2483"]["inputs"]["text"]=PROMPT
    g["4852"]["inputs"]["filename_prefix"]=tag
    before=set(os.listdir(COMFY_OUT)); t0=time.time(); peak=0
    try: pid=post(g)["prompt_id"]
    except urllib.error.HTTPError as e:
        print(f"{tag} SUBMIT_ERR",json.loads(e.read()).get("error",{}).get("message")); continue
    ok=False
    while time.time()-t0<1800:
        peak=max(peak,vram_now()); time.sleep(3)
        h=getj(f"/history/{pid}")
        if pid in h:
            st=h[pid]["status"]
            if st.get("status_str")=="error":
                print(f"{tag} RUN_ERR",[str(m)[:200] for m in st.get("messages",[]) if 'error' in str(m).lower()][:1]); break
            if st.get("completed") or st.get("status_str")=="success": ok=True; break
    if not ok: continue
    new=[f for f in os.listdir(COMFY_OUT) if f.endswith(".mp4") and f not in before]
    if not new: print(f"{tag} NO_OUTPUT"); continue
    src=sorted(new,key=lambda f:os.path.getmtime(os.path.join(COMFY_OUT,f)))[-1]
    dst=f"{OUT}/{tag}.mp4"; import shutil; shutil.copy(os.path.join(COMFY_OUT,src),dst)
    tn,sh=metrics(dst); dt=time.time()-t0
    rows.append((tag,dist,steps,dt,peak,tn,sh))
    print(f"{tag:26} d={dist} steps={steps}  {dt:5.0f}s  VRAM={peak/1024:.1f}GB  tNoise={tn:.2f} sharp={sh:.0f}")
print("\n=== SUMMARY ===")
print(f"{'config':26}{'sec':>6}{'VRAM':>8}{'tNoise':>8}{'sharp':>8}")
for r in rows: print(f"{r[0]:26}{r[3]:6.0f}{r[4]/1024:7.1f}G{r[5]:8.2f}{r[6]:8.0f}")
