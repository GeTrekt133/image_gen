#!/usr/bin/env python3
"""Матрица 5 промптов × N баз с ПРАВИЛЬНЫМИ настройками каждой модели.
Группировка по модели (загрузка весов один раз). Сохраняет в /workspace/gallery/matrix/.
Результат -> /workspace/matrix_results.json
"""
import json, time, os, shutil, subprocess, threading, urllib.request

HOST="http://127.0.0.1:10100"
COMFY_OUT="/workspace/ComfyUI/output"
OUT="/workspace/gallery/matrix"
os.makedirs(OUT,exist_ok=True)

# 5 промптов: портрет + fitness (pushup top+легинсы) + street + beach + evening
PROMPTS=[
 ("portrait","instagram-style portrait photo of a beautiful 24 year old woman, influencer aesthetic, natural glowing skin with realistic pores, soft window daylight, shot on 85mm f1.4, shallow depth of field, elegant casual outfit, looking at camera, photorealistic, highly detailed",1024,1024,900),
 ("fitness","full body photo of a fit 24 year old woman at a modern gym, wearing a push-up sports bra top and tight high-waist seamless leggings, athletic toned figure, standing confident pose, bright daylight, photorealistic, highly detailed, full length shot",832,1216,901),
 ("street","full body street-style fashion photo of a young woman, cropped tank top and high-waist skinny jeans, urban city background, golden hour, candid walking pose, photorealistic, sharp focus, full length",832,1216,902),
 ("beach","full body photo of a young woman at the beach, wearing a swimsuit, sunny day, ocean and sand background, relaxed pose, photorealistic, natural skin, full length shot",832,1216,903),
 ("evening","photo of a young woman in an elegant fitted evening cocktail dress, upscale restaurant interior, glamour lighting, confident pose, photorealistic, highly detailed",896,1152,904),
]

# базы: (label, wf, positive_node, prefix, has_negative_node)
MODELS=[
 ("Z-Image Turbo","wf_zimage.json","4","zimage",None),
 ("FLUX.2 Dev","wf_flux2.json","4","flux2",None),
 ("Qwen-Image","wf_qwen.json","4","qwen","5"),
 ("Juggernaut XL","wf_sdxl_juggernaut.json","6","juggernaut","7"),
 ("RealVisXL V5","wf_sdxl_realvis.json","6","realvis","7"),
 ("Big Lust v1.6","wf_sdxl_biglust.json","6","biglust","7"),
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
def free():
    try:
        urllib.request.urlopen(urllib.request.Request(HOST+"/free",data=json.dumps({"unload_models":True,"free_memory":True}).encode(),headers={"Content-Type":"application/json"}),timeout=60).read()
        time.sleep(2)
    except Exception as e: print("free warn",e)

def set_wh(g,w,h):
    for n in g.values():
        ins=n.get("inputs",{})
        if "width" in ins and "height" in ins: ins["width"]=w; ins["height"]=h

def run_one(g,label,tag):
    pk=Peak(); pk.start(); t0=time.time()
    try: r=post(g)
    except Exception as e: pk.stop=True; return {"status":"POST_FAIL","err":str(e)[:200]}
    pid=r.get("prompt_id"); img=None; status="TIMEOUT"
    while time.time()-t0<1800:
        time.sleep(2); h=getj(f"/history/{pid}")
        if pid in h:
            st=h[pid]["status"]
            if st.get("status_str")=="success" or st.get("completed"):
                for _,o in h[pid].get("outputs",{}).items():
                    for im in o.get("images",[]): img=im; break
                status="OK"; break
            if st.get("status_str")=="error":
                pk.stop=True; return {"status":"ERROR","detail":str(st.get("messages",[]))[-400:]}
    pk.stop=True; dt=time.time()-t0; saved=None
    if img:
        src=os.path.join(COMFY_OUT,img.get("subfolder",""),img["filename"])
        dst=os.path.join(OUT,f"{tag}.png")
        if os.path.exists(src): shutil.copy(src,dst); saved=dst
    return {"status":status,"seconds":round(dt,1),"peak_vram_gb":round(pk.peak/1024,1),"image":saved}

def main():
    results=[]
    for label,wf,pos,pfx,neg in MODELS:
        print(f"\n########## {label} ##########",flush=True)
        free()
        for pkey,ptext,w,h,seed in PROMPTS:
            g=json.load(open(wf))
            g[pos]["inputs"]["text"]=ptext
            set_wh(g,w,h)
            for n in g.values():
                ins=n.get("inputs",{})
                if "seed" in ins: ins["seed"]=seed
                if "noise_seed" in ins: ins["noise_seed"]=seed
            tag=f"{pfx}__{pkey}"
            res=run_one(g,label,tag)
            res.update({"model":label,"prompt":pkey,"tag":tag})
            print(f"  {pkey:<9} {res['status']:<7} {res.get('seconds','-')}s",flush=True)
            results.append(res)
        json.dump(results,open("/workspace/matrix_results.json","w"),ensure_ascii=False,indent=2)
    ok=sum(1 for r in results if r["status"]=="OK")
    print(f"\n=== ГОТОВО: {ok}/{len(results)} OK ===")

if __name__=="__main__": main()
