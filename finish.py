#!/usr/bin/env python3
"""Универсальный финиш-блок: 2K -> skin-refine dn0.35 -> FaceDetailer.
Работает на ЛЮБОЙ картинке из любого пайплайна (t2i / pose / depth / qwen-edit / внешняя).
Usage: python finish.py <in1.png> [in2.png ...]   ->  сохраняет <name>_finished.png рядом."""
import sys,json,time,os,shutil,urllib.request,copy
HOST="http://127.0.0.1:10100"; COMFY_OUT="/workspace/ComfyUI/output"; INP="/workspace/ComfyUI/input"
base=json.load(open("/workspace/wf_finish_block.json"))
def post(x): return json.load(urllib.request.urlopen(urllib.request.Request(HOST+"/prompt",data=json.dumps({"prompt":x}).encode(),headers={"Content-Type":"application/json"}),timeout=60))
def getj(p): return json.load(urllib.request.urlopen(HOST+p,timeout=60))
from PIL import Image

def finish(path):
    name=os.path.splitext(os.path.basename(path))[0]
    tmpname=f"finish_in_{name}.png"; shutil.copy(path, os.path.join(INP,tmpname))
    g=copy.deepcopy(base); g["5"]["inputs"]["image"]=tmpname
    t0=time.time(); r=post(g); pid=r["prompt_id"]; outs={}
    while time.time()-t0<600:
        time.sleep(2); h=getj(f"/history/{pid}")
        if pid in h:
            st=h[pid]["status"]
            if st.get("status_str")=="error": print(name,"ERROR",str(st.get("messages"))[:250]); return None
            if st.get("status_str")=="success" or st.get("completed"): outs=h[pid]["outputs"]; break
    im=outs.get("62",{}).get("images",[])
    if not im: print(name,"no output"); return None
    src=os.path.join(COMFY_OUT,im[0].get("subfolder",""),im[0]["filename"])
    out=os.path.join(os.path.dirname(path), name+"_finished.png"); shutil.copy(src,out)
    print(f"{name:28s} -> {os.path.basename(out)}  {Image.open(out).size}  {time.time()-t0:.1f}s")
    return out

if __name__=="__main__":
    for p in sys.argv[1:]: finish(p)
    print("DONE")
