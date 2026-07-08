#!/usr/bin/env python3
"""Сравнение NSFW-LoRA вариантов на explicit-кадрах (base t2i, изолирован эффект LoRA).
6 вариантов × N промптов, одинаковый сид. -> /workspace/gallery/lora_compare/"""
import json, time, os, shutil, urllib.request

HOST="http://127.0.0.1:10100"; COMFY_OUT="/workspace/ComfyUI/output"
OUT="/workspace/gallery/lora_compare"; os.makedirs(OUT,exist_ok=True)
WF="/workspace/wf_nsfw_var.json"

VARIANTS=[  # (tag, lora_file или None=raw, strength)
 ("raw",        None,                              None),
 ("nsfw23k",    "zimage_nsfw.safetensors",         1.0),
 ("godpussy",   "zimage_nsfw_godpussy.safetensors",1.0),
 ("pussy",      "zimage_nsfw_pussy.safetensors",   1.0),
 ("eason_v31",  "zimage_nsfw_eason_v31.safetensors",1.0),
 ("nsfw_v2",    "zimage_nsfw_v2.safetensors",      1.0),
]
PROMPTS=[
 ("boudoir","photorealistic nude boudoir photo of a beautiful 24 year old woman lying on white silk sheets, natural skin texture with realistic pores, soft window daylight, 85mm f1.4, shallow depth of field, elegant, tasteful, highly detailed, sharp focus",896,1152,555),
 ("lingerie","full body photo of a beautiful 24 year old woman in black lace lingerie, standing by a window, natural glowing skin, soft morning light, photorealistic, highly detailed, 85mm lens, full length shot",832,1216,556),
 ("topless","photorealistic topless portrait of a beautiful 24 year old woman, natural bare skin with realistic texture, relaxed confident pose, soft studio light, 85mm f1.8, shallow depth of field, highly detailed",896,1152,557),
 ("doggy","photorealistic explicit nude photo of a beautiful 24 year old woman on all fours on a bed, doggy style pose viewed from behind, arched back, bare ass raised, exposed pussy visible between thighs, natural skin texture with realistic detail, soft bedroom daylight, 85mm f1.8, highly detailed, sharp focus",1216,832,601),
 ("spread","explicit close-up photorealistic photo of a beautiful 24 year old woman lying on her back with legs spread wide, detailed vulva and pussy fully visible, smooth natural skin, soft warm light, shallow depth of field, highly detailed, sharp focus",1024,1024,602),
 ("pussyplay","photorealistic explicit photo of a beautiful 24 year old woman masturbating on a bed, one hand between her spread legs, fingers touching her pussy, aroused parted-lip expression, natural glowing skin, soft window light, 85mm, highly detailed",896,1152,603),
 ("dildo","photorealistic explicit photo of a beautiful 24 year old woman using a dildo sex toy, holding it against her pussy between spread legs, reclined on bed, natural skin texture, soft bedroom light, 85mm f1.8, highly detailed, sharp focus",896,1152,604),
]

def post(g): return json.load(urllib.request.urlopen(urllib.request.Request(HOST+"/prompt",data=json.dumps({"prompt":g}).encode(),headers={"Content-Type":"application/json"}),timeout=60))
def getj(p): return json.load(urllib.request.urlopen(HOST+p,timeout=60))

def build(text,w,h,seed,lora,strength):
    g=json.load(open(WF))
    g["30"]["inputs"]["text"]=text
    g["40"]["inputs"]["width"]=w; g["40"]["inputs"]["height"]=h
    g["50"]["inputs"]["seed"]=seed
    if lora is None:
        g["20"]["inputs"]["model"]=["1",0]; g.pop("10",None)   # без LoRA
    else:
        g["10"]["inputs"]["lora_name"]=lora; g["10"]["inputs"]["strength_model"]=strength
    return g

def run(g):
    t0=time.time(); r=post(g); pid=r.get("prompt_id"); img=None
    while time.time()-t0<600:
        time.sleep(1.5); h=getj(f"/history/{pid}")
        if pid in h:
            st=h[pid]["status"]
            if st.get("status_str")=="error": return None,"ERROR",round(time.time()-t0,1),str(st.get("messages",[]))[-300:]
            if st.get("status_str")=="success" or st.get("completed"):
                for _,o in h[pid].get("outputs",{}).items():
                    for im in o.get("images",[]): img=im; break
                return img,"OK",round(time.time()-t0,1),None
    return None,"TIMEOUT",round(time.time()-t0,1),None

def main():
    res=[]
    for pkey,text,w,h,seed in PROMPTS:
        for tag,lora,strn in VARIANTS:
            g=build(text,w,h,seed,lora,strn)
            img,status,sec,err=run(g)
            saved=None
            if img:
                src=os.path.join(COMFY_OUT,img.get("subfolder",""),img["filename"])
                dst=os.path.join(OUT,f"{pkey}__{tag}.png")
                if os.path.exists(src): shutil.copy(src,dst); saved=os.path.basename(dst)
            print(f"{pkey:8s} {tag:10s} {status:7s} {sec}s {err or ''}",flush=True)
            res.append({"prompt":pkey,"variant":tag,"lora":lora,"status":status,"seconds":sec,"image":saved,"seed":seed,"w":w,"h":h})
        json.dump(res,open(f"{OUT}/results.json","w"),ensure_ascii=False,indent=2)
    print(f"\n=== ГОТОВО: {sum(1 for r in res if r['status']=='OK')}/{len(res)} OK ===")

if __name__=="__main__": main()
