#!/usr/bin/env python3
"""Прогон unStable Revolution ZIT (full NSFW checkpoint) по всем 7 NSFW-промптам.
base t2i, те же сиды/разрешения, что в lora_compare. Добавляет variant='unstable' в тот же results.json."""
import json, time, os, shutil, urllib.request

HOST="http://127.0.0.1:10100"; COMFY_OUT="/workspace/ComfyUI/output"
OUT="/workspace/gallery/lora_compare"; os.makedirs(OUT,exist_ok=True)
WF="/workspace/wf_unstable.json"
RES=f"{OUT}/results.json"

# те же промпты/сиды/wh, что в run_lora_compare.py
PROMPTS=[
 ("boudoir","photorealistic nude boudoir photo of a beautiful 24 year old woman lying on white silk sheets, natural skin texture with realistic pores, soft window daylight, 85mm f1.4, shallow depth of field, elegant, tasteful, highly detailed, sharp focus",896,1152,555),
 ("lingerie","full body photo of a beautiful 24 year old woman in black lace lingerie, standing by a window, natural glowing skin, soft morning light, photorealistic, highly detailed, 85mm lens, full length shot",832,1216,556),
 ("topless","photorealistic topless portrait of a beautiful 24 year old woman, natural bare skin with realistic texture, relaxed confident pose, soft studio light, 85mm f1.8, shallow depth of field, highly detailed",896,1152,557),
 ("doggy","photorealistic explicit nude photo of a beautiful 24 year old woman on all fours on a bed, doggy style pose viewed from behind, arched back, bare ass raised, exposed pussy visible between thighs, natural skin texture with realistic detail, soft bedroom daylight, 85mm f1.8, highly detailed, sharp focus",1216,832,601),
 ("spread","explicit close-up photorealistic photo of a beautiful 24 year old woman lying on her back with legs spread wide, detailed vulva and pussy fully visible, smooth natural skin, soft warm light, shallow depth of field, highly detailed, sharp focus",1024,1024,602),
 ("pussyplay","photorealistic explicit photo of a beautiful 24 year old woman masturbating on a bed, one hand between her spread legs, fingers touching her pussy, aroused parted-lip expression, natural glowing skin, soft window light, 85mm, highly detailed",896,1152,603),
 ("dildo","photorealistic explicit photo of a beautiful 24 year old woman using a dildo sex toy, holding it against her pussy between spread legs, reclined on bed, natural skin texture, soft bedroom light, 85mm f1.8, highly detailed, sharp focus",896,1152,604),
]
TAG="unstable"

def post(g): return json.load(urllib.request.urlopen(urllib.request.Request(HOST+"/prompt",data=json.dumps({"prompt":g}).encode(),headers={"Content-Type":"application/json"}),timeout=60))
def getj(p): return json.load(urllib.request.urlopen(HOST+p,timeout=60))

def run(text,w,h,seed):
    g=json.load(open(WF))
    g["30"]["inputs"]["text"]=text; g["40"]["inputs"]["width"]=w; g["40"]["inputs"]["height"]=h
    g["50"]["inputs"]["seed"]=seed
    t0=time.time(); r=post(g); pid=r.get("prompt_id"); img=None
    while time.time()-t0<600:
        time.sleep(1.5); h_=getj(f"/history/{pid}")
        if pid in h_:
            st=h_[pid]["status"]
            if st.get("status_str")=="error": return None,"ERROR",round(time.time()-t0,1),str(st.get("messages",[]))[-300:]
            if st.get("status_str")=="success" or st.get("completed"):
                for _,o in h_[pid].get("outputs",{}).items():
                    for im in o.get("images",[]): img=im; break
                return img,"OK",round(time.time()-t0,1),None
    return None,"TIMEOUT",round(time.time()-t0,1),None

def main():
    res=json.load(open(RES)) if os.path.exists(RES) else []
    res=[r for r in res if r.get("variant")!=TAG]  # чистим прошлый unstable
    for pkey,text,w,h,seed in PROMPTS:
        img,status,sec,err=run(text,w,h,seed); saved=None
        if img:
            src=os.path.join(COMFY_OUT,img.get("subfolder",""),img["filename"])
            dst=os.path.join(OUT,f"{pkey}__{TAG}.png")
            if os.path.exists(src): shutil.copy(src,dst); saved=os.path.basename(dst)
        print(f"{pkey:10s} {TAG:9s} {status:7s} {sec}s {err or ''}",flush=True)
        res.append({"prompt":pkey,"variant":TAG,"lora":"unStable Revolution ZIT v3 fp16 (checkpoint)","status":status,"seconds":sec,"image":saved,"seed":seed,"w":w,"h":h})
        json.dump(res,open(RES,"w"),ensure_ascii=False,indent=2)
    print(f"\n=== unStable ГОТОВО: {sum(1 for r in res if r.get('variant')==TAG and r['status']=='OK')}/{len(PROMPTS)} OK ===")

if __name__=="__main__": main()
