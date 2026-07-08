#!/usr/bin/env python3
"""Дебаг-матрица LTX motion-control: прогоняет конфиги, автоматически меряет
яркость кадров (детект canny-коллапса) и складывает mid-кадры + результат.
-> /workspace/gallery/ltx_matrix/"""
import json,time,os,shutil,subprocess,urllib.request,copy
import numpy as np
from PIL import Image

HOST="http://127.0.0.1:10100"; COMFY_OUT="/workspace/ComfyUI/output"
OUT="/workspace/gallery/ltx_matrix"; os.makedirs(OUT,exist_ok=True)
BASE=json.load(open("/workspace/wf_ltx_dance.json"))

def post(g): return json.load(urllib.request.urlopen(urllib.request.Request(HOST+"/prompt",data=json.dumps({"prompt":g}).encode(),headers={"Content-Type":"application/json"}),timeout=120))
def getj(p): return json.load(urllib.request.urlopen(HOST+p,timeout=60))

def run_cfg(tag, mutate):
    g=copy.deepcopy(BASE); mutate(g)
    t0=time.time()
    try: pid=post(g)["prompt_id"]
    except Exception as e:
        print(f"{tag:28s} SUBMIT_FAIL {str(e)[:120]}"); return
    while time.time()-t0<1200:
        time.sleep(8); h=getj(f"/history/{pid}")
        if pid in h:
            st=h[pid]["status"]
            if st.get("status_str")=="error":
                msg=[str(m)[:160] for m in st.get("messages",[]) if 'error' in str(m).lower()]
                print(f"{tag:28s} ERROR {msg[:1]}"); return
            if st.get("status_str")=="success" or st.get("completed"): break
    # grab newest mp4
    mp4=sorted([f for f in os.listdir(COMFY_OUT) if f.startswith("ltx_dance") and f.endswith(".mp4")],
               key=lambda f: os.path.getmtime(os.path.join(COMFY_OUT,f)))[-1]
    dst=f"{OUT}/{tag}.mp4"; shutil.copy(os.path.join(COMFY_OUT,mp4),dst)
    # brightness across frames -> collapse detector
    bright=[]
    for f in [0,15,40,70,100]:
        fp=f"{OUT}/{tag}_f{f}.png"
        subprocess.run(["ffmpeg","-y","-i",dst,"-vf",f"select='eq(n,{f})'","-vframes","1",fp],capture_output=True)
        if os.path.exists(fp):
            bright.append(round(float(np.asarray(Image.open(fp).convert("L"),dtype=np.float32).mean()),1))
    verdict="COLLAPSE" if len(bright)>1 and min(bright[1:])<30 else "OK-VIDEO"
    print(f"{tag:28s} {verdict}  brightness={bright}  {time.time()-t0:.0f}s  -> {os.path.basename(dst)}")

# ---- конфиги ----
def fix_common(g):
    g["5012"]["inputs"]["latent_downscale_factor"]=["5011",1]   # ГЛАВНЫЙ ФИКС: линк из лоадера (ref0.5)
    g["5019"]["inputs"]["value"]=False                          # персона-якорь ON
    g["4831"]["inputs"]["sampler_name"]="euler_ancestral"
    g["5001"]["inputs"]["file"]="dance_drive.mp4"
    g["2483"]["inputs"]["text"]="photorealistic video of a beautiful 24 year old woman with long brown hair dancing, natural skin, cinematic, highly detailed"

def cfg_author(g):   # как у автора: union 1.0, guide 0.75
    fix_common(g); g["5011"]["inputs"]["strength_model"]=1.0; g["5012"]["inputs"]["strength"]=0.75
def cfg_mid(g):      # union 0.5, guide 0.75
    fix_common(g); g["5011"]["inputs"]["strength_model"]=0.5; g["5012"]["inputs"]["strength"]=0.75
def cfg_control_off(g):  # санити: контрол выключен (ожидаем OK-VIDEO)
    fix_common(g); g["5011"]["inputs"]["strength_model"]=0.0; g["5012"]["inputs"]["strength"]=0.0

for tag,fn in [("A_df-link_author_u1.0_g0.75",cfg_author),
               ("B_df-link_u0.5_g0.75",cfg_mid),
               ("C_sanity_control_off",cfg_control_off)]:
    run_cfg(tag,fn)
print("MATRIX_DONE")
