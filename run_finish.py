#!/usr/bin/env python3
"""Прогон wf_zimage_finish_nsfw.json по набору промптов.
Один сабмит = 4 стадии (A raw / B +NSFW / C +FaceDetailer / D +2K+skin).
Собирает выходы по node-id, копирует в /workspace/gallery/finish_nsfw/, пишет results.json."""
import json, time, os, shutil, subprocess, threading, urllib.request

HOST="http://127.0.0.1:10100"
COMFY_OUT="/workspace/ComfyUI/output"
OUT="/workspace/gallery/finish_nsfw"
os.makedirs(OUT,exist_ok=True)
WF="/workspace/wf_zimage_finish_nsfw.json"

# node-id -> стадия
STAGES={"52":"A_raw","62":"B_nsfw","72":"C_facedetailer","86":"D_final"}

# (key, prompt, w, h, seed) — синтетическая 18+ персона, NSFW в скоупе (CLAUDE.md)
PROMPTS=[
 ("boudoir","photorealistic nude boudoir photo of a beautiful 24 year old woman lying on white silk sheets, natural skin texture with realistic pores, soft window daylight, 85mm f1.4, shallow depth of field, elegant, tasteful, highly detailed, sharp focus",896,1152,555),
 ("lingerie","full body photo of a beautiful 24 year old woman in black lace lingerie, standing by a window, natural glowing skin, soft morning light, photorealistic, highly detailed, 85mm lens, full length shot",832,1216,556),
 ("topless","photorealistic topless portrait of a beautiful 24 year old woman, natural bare skin with realistic texture, relaxed confident pose, soft studio light, 85mm f1.8, shallow depth of field, highly detailed",896,1152,557),
]

def gpu_mib():
    try: return int(subprocess.check_output(["nvidia-smi","--query-gpu=memory.used","--format=csv,noheader,nounits"]).decode().split("\n")[0])
    except: return 0
class Peak(threading.Thread):
    def __init__(s): super().__init__(daemon=True); s.stop=False; s.peak=0
    def run(s):
        while not s.stop: s.peak=max(s.peak,gpu_mib()); time.sleep(0.4)
def post(g):
    return json.load(urllib.request.urlopen(urllib.request.Request(HOST+"/prompt",data=json.dumps({"prompt":g}).encode(),headers={"Content-Type":"application/json"}),timeout=60))
def getj(p): return json.load(urllib.request.urlopen(HOST+p,timeout=60))

def run_one(key,text,w,h,seed):
    g=json.load(open(WF))
    g["30"]["inputs"]["text"]=text
    g["40"]["inputs"]["width"]=w; g["40"]["inputs"]["height"]=h
    # разные, но детерминированные сиды по стадиям
    g["50"]["inputs"]["seed"]=seed
    g["60"]["inputs"]["seed"]=seed
    g["71"]["inputs"]["seed"]=seed+1000
    g["84"]["inputs"]["seed"]=seed+2000
    pk=Peak(); pk.start(); t0=time.time()
    try: r=post(g)
    except Exception as e: pk.stop=True; return {"key":key,"status":"POST_FAIL","err":str(e)[:400]}
    pid=r.get("prompt_id"); status="TIMEOUT"; outs={}
    while time.time()-t0<1800:
        time.sleep(2); h_=getj(f"/history/{pid}")
        if pid in h_:
            st=h_[pid]["status"]
            if st.get("status_str")=="error":
                pk.stop=True; return {"key":key,"status":"ERROR","detail":str(st.get("messages",[]))[-800:]}
            if st.get("status_str")=="success" or st.get("completed"):
                outs=h_[pid].get("outputs",{}); status="OK"; break
    pk.stop=True; dt=time.time()-t0
    saved={}
    for nid,stage in STAGES.items():
        imgs=outs.get(nid,{}).get("images",[])
        if imgs:
            im=imgs[0]; src=os.path.join(COMFY_OUT,im.get("subfolder",""),im["filename"])
            dst=os.path.join(OUT,f"{key}__{stage}.png")
            if os.path.exists(src): shutil.copy(src,dst); saved[stage]=os.path.basename(dst)
    return {"key":key,"status":status,"seconds":round(dt,1),"peak_vram_gb":round(pk.peak/1024,1),
            "w":w,"h":h,"seed":seed,"prompt":text,"images":saved}

def main():
    results=[]
    for key,text,w,h,seed in PROMPTS:
        print(f"\n=== {key} ({w}x{h}, seed {seed}) ===",flush=True)
        res=run_one(key,text,w,h,seed)
        print(json.dumps({k:v for k,v in res.items() if k!='prompt'},ensure_ascii=False),flush=True)
        results.append(res)
        json.dump(results,open("/workspace/gallery/finish_nsfw/results.json","w"),ensure_ascii=False,indent=2)
    ok=sum(1 for r in results if r["status"]=="OK")
    print(f"\n=== ГОТОВО: {ok}/{len(results)} OK ===")

if __name__=="__main__": main()
