#!/usr/bin/env python3
"""Explicit-набор через wf_zimage_finish_nsfw.json (Snapshot v5 в финише).
Один сабмит = 4 стадии. Выход -> /workspace/gallery/explicit/. Синтетическая 18+ персона (CLAUDE.md)."""
import json, time, os, shutil, subprocess, threading, urllib.request

HOST="http://127.0.0.1:10100"
COMFY_OUT="/workspace/ComfyUI/output"
OUT="/workspace/gallery/explicit"; os.makedirs(OUT,exist_ok=True)
WF="/workspace/wf_zimage_finish_nsfw.json"
STAGES={"52":"A_raw","62":"B_nsfw","72":"C_facedetailer","86":"D_final"}

PROMPTS=[
 ("doggy","photorealistic explicit nude photo of a beautiful 24 year old woman on all fours on a bed, doggy style pose viewed from behind, arched back, bare ass raised, exposed pussy visible between thighs, natural skin texture with realistic detail, soft bedroom daylight, 85mm f1.8, highly detailed, sharp focus",1216,832,601),
 ("spread","explicit close-up photorealistic photo of a beautiful 24 year old woman lying on her back with legs spread wide, detailed vulva and pussy fully visible, smooth natural skin, soft warm light, shallow depth of field, highly detailed, sharp focus",1024,1024,602),
 ("pussyplay","photorealistic explicit photo of a beautiful 24 year old woman masturbating on a bed, one hand between her spread legs, fingers touching her pussy, aroused parted-lip expression, natural glowing skin, soft window light, 85mm, highly detailed",896,1152,603),
 ("dildo","photorealistic explicit photo of a beautiful 24 year old woman using a dildo sex toy, holding it against her pussy between spread legs, reclined on bed, natural skin texture, soft bedroom light, 85mm f1.8, highly detailed, sharp focus",896,1152,604),
]

def gpu_mib():
    try: return int(subprocess.check_output(["nvidia-smi","--query-gpu=memory.used","--format=csv,noheader,nounits"]).decode().split("\n")[0])
    except: return 0
class Peak(threading.Thread):
    def __init__(s): super().__init__(daemon=True); s.stop=False; s.peak=0
    def run(s):
        while not s.stop: s.peak=max(s.peak,gpu_mib()); time.sleep(0.4)
def post(g): return json.load(urllib.request.urlopen(urllib.request.Request(HOST+"/prompt",data=json.dumps({"prompt":g}).encode(),headers={"Content-Type":"application/json"}),timeout=60))
def getj(p): return json.load(urllib.request.urlopen(HOST+p,timeout=60))

def run_one(key,text,w,h,seed):
    g=json.load(open(WF))
    g["30"]["inputs"]["text"]=text
    g["40"]["inputs"]["width"]=w; g["40"]["inputs"]["height"]=h
    g["50"]["inputs"]["seed"]=seed; g["60"]["inputs"]["seed"]=seed
    g["71"]["inputs"]["seed"]=seed+1000; g["84"]["inputs"]["seed"]=seed+2000
    pk=Peak(); pk.start(); t0=time.time()
    try: r=post(g)
    except Exception as e: pk.stop=True; return {"key":key,"status":"POST_FAIL","err":str(e)[:400]}
    pid=r.get("prompt_id"); status="TIMEOUT"; outs={}
    while time.time()-t0<1800:
        time.sleep(2); h_=getj(f"/history/{pid}")
        if pid in h_:
            st=h_[pid]["status"]
            if st.get("status_str")=="error": pk.stop=True; return {"key":key,"status":"ERROR","detail":str(st.get("messages",[]))[-800:]}
            if st.get("status_str")=="success" or st.get("completed"): outs=h_[pid].get("outputs",{}); status="OK"; break
    pk.stop=True; dt=time.time()-t0; saved={}
    for nid,stage in STAGES.items():
        imgs=outs.get(nid,{}).get("images",[])
        if imgs:
            im=imgs[0]; src=os.path.join(COMFY_OUT,im.get("subfolder",""),im["filename"])
            dst=os.path.join(OUT,f"{key}__{stage}.png")
            if os.path.exists(src): shutil.copy(src,dst); saved[stage]=os.path.basename(dst)
    return {"key":key,"status":status,"seconds":round(dt,1),"peak_vram_gb":round(pk.peak/1024,1),"w":w,"h":h,"seed":seed,"prompt":text,"images":saved}

def main():
    results=[]
    for key,text,w,h,seed in PROMPTS:
        print(f"\n=== {key} ({w}x{h}, seed {seed}) ===",flush=True)
        res=run_one(key,text,w,h,seed)
        print(json.dumps({k:v for k,v in res.items() if k!='prompt'},ensure_ascii=False),flush=True)
        results.append(res); json.dump(results,open(f"{OUT}/results.json","w"),ensure_ascii=False,indent=2)
    print(f"\n=== ГОТОВО: {sum(1 for r in results if r['status']=='OK')}/{len(results)} OK ===")

if __name__=="__main__": main()
