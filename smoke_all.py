#!/usr/bin/env python3
"""Единый смоук всех фото-баз одним и тем же insta-influencer промптом.
Замеряет секунды, it/s (шаги/сек) и пиковую VRAM (nvidia-smi), собирает выходные файлы.
Результат -> /workspace/smoke_results.json + копии картинок в /workspace/gallery/base/.
Usage: python smoke_all.py [shared_seed]
"""
import json, sys, time, os, shutil, subprocess, threading, urllib.request

HOST = "http://127.0.0.1:10100"
COMFY_OUT = "/workspace/ComfyUI/output"
GAL = "/workspace/gallery/base"
os.makedirs(GAL, exist_ok=True)

# Общий insta/AI-influencer промпт для честного сравнения
SHARED_POS = ("instagram-style portrait photo of a beautiful 24 year old woman, influencer aesthetic, "
              "natural glowing skin with realistic pores, soft window daylight, shot on 85mm f1.4, "
              "shallow depth of field, elegant casual outfit, relaxed pose looking at camera, "
              "photorealistic, highly detailed, sharp focus")
SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 777

# base: (label, wf_file, positive_node, seed_node, steps_field_node, prefix)
BASES = [
    ("Z-Image Turbo (bf16)",  "wf_zimage.json",          "4", "8", "test_zimage"),
    ("FLUX.2 Dev (fp8mixed)", "wf_flux2.json",           "4", "8", "smoke_flux2"),
    ("Qwen-Image (bf16)",     "wf_qwen.json",            "4", "7", "smoke_qwen"),
    ("SDXL Juggernaut XL",    "wf_sdxl_juggernaut.json", "6", "3", "smoke_sdxl_juggernaut"),
    ("SDXL RealVisXL V5",     "wf_sdxl_realvis.json",    "6", "3", "smoke_sdxl_realvis"),
    ("SDXL Big Lust v1.6",    "wf_sdxl_biglust.json",    "6", "3", "smoke_sdxl_biglust"),
]

def gpu_mib():
    try:
        o = subprocess.check_output(["nvidia-smi","--query-gpu=memory.used","--format=csv,noheader,nounits"])
        return int(o.decode().split("\n")[0])
    except Exception:
        return 0

class Peak(threading.Thread):
    def __init__(self): super().__init__(daemon=True); self.stop=False; self.peak=0
    def run(self):
        while not self.stop:
            self.peak=max(self.peak, gpu_mib()); time.sleep(0.4)

def post(graph):
    data=json.dumps({"prompt":graph}).encode()
    req=urllib.request.Request(HOST+"/prompt",data=data,headers={"Content-Type":"application/json"})
    return json.load(urllib.request.urlopen(req,timeout=60))

def getj(p):
    return json.load(urllib.request.urlopen(HOST+p,timeout=60))

def free_models():
    try:
        data=json.dumps({"unload_models":True,"free_memory":True}).encode()
        req=urllib.request.Request(HOST+"/free",data=data,headers={"Content-Type":"application/json"})
        urllib.request.urlopen(req,timeout=60).read()
        time.sleep(3)
    except Exception as e:
        print("  free warn:",e)

def run_base(label, wf, pos, seednode, prefix, seed):
    free_models()  # выгрузить прошлую модель -> чистый замер пиковой VRAM
    g=json.load(open(wf))
    g[pos]["inputs"]["text"]=SHARED_POS
    # seed/steps — node-id-агностично (KSampler 'seed' | RandomNoise 'noise_seed')
    steps=0
    for nid,n in g.items():
        ins=n.get("inputs",{})
        if "seed" in ins: ins["seed"]=seed
        if "noise_seed" in ins: ins["noise_seed"]=seed
        if isinstance(ins.get("steps"),(int,float)): steps=max(steps,int(ins["steps"]))
    pk=Peak(); pk.start(); t0=time.time()
    try:
        r=post(g)
    except Exception as e:
        pk.stop=True
        return {"label":label,"status":"POST_FAIL","error":str(e)[:300]}
    pid=r.get("prompt_id")
    img=None; status="TIMEOUT"
    while time.time()-t0 < 1200:
        time.sleep(2)
        h=getj(f"/history/{pid}")
        if pid in h:
            st=h[pid].get("status",{})
            if st.get("status_str")=="success" or st.get("completed"):
                for _,out in h[pid].get("outputs",{}).items():
                    for im in out.get("images",[]):
                        img=im; break
                status="OK"; break
            if st.get("status_str")=="error":
                status="ERROR"
                msgs=st.get("messages",[])
                pk.stop=True
                return {"label":label,"status":"ERROR","detail":str(msgs)[-500:],"seconds":round(time.time()-t0,1)}
    pk.stop=True
    dt=time.time()-t0
    saved=None
    if img:
        src=os.path.join(COMFY_OUT, img.get("subfolder",""), img["filename"])
        dst=os.path.join(GAL, f"{prefix}.png")
        if os.path.exists(src): shutil.copy(src,dst); saved=dst
    its = round(steps/dt,3) if dt>0 and steps else None
    return {"label":label,"status":status,"seconds":round(dt,1),"steps":steps,
            "it_per_s":its,"peak_vram_mib":pk.peak,"peak_vram_gb":round(pk.peak/1024,1),
            "image":saved,"wf":wf,"seed":seed}

def main():
    results=[]
    for label,wf,pos,seednode,prefix in BASES:
        print(f"\n=== SMOKE: {label} ({wf}) ===",flush=True)
        res=run_base(label,wf,pos,seednode,prefix,SEED)
        print(json.dumps(res,ensure_ascii=False),flush=True)
        results.append(res)
    json.dump(results,open("/workspace/smoke_results.json","w"),ensure_ascii=False,indent=2)
    print("\n=== ИТОГ ===")
    print(f"{'База':<26}{'статус':<9}{'сек':>7}{'it/s':>8}{'VRAM ГБ':>9}")
    for r in results:
        print(f"{r['label']:<26}{r['status']:<9}{str(r.get('seconds','-')):>7}{str(r.get('it_per_s','-')):>8}{str(r.get('peak_vram_gb','-')):>9}")

if __name__=="__main__":
    main()
