#!/usr/bin/env python3
"""Real-background compositing:
  plate = driving video with the original dancer removed per-frame (LaMa)  -> real room, live light
  fg    = our generated persona clip, matted per-frame (rembg)
  out   = persona over the REAL room.
"""
import cv2, numpy as np, os, sys, argparse
from PIL import Image
from rembg import remove, new_session
from simple_lama_inpainting import SimpleLama
ap=argparse.ArgumentParser()
ap.add_argument("--persona", default="/workspace/gallery/sweep/G_dist0.90_euler_ancestral_seed7.mp4")
ap.add_argument("--drive",   default="/workspace/ComfyUI/input/drive_dzp.mp4")
ap.add_argument("--mask",    default="/workspace/ComfyUI/input/drive_mask.mp4")
ap.add_argument("--tag",     default="persona_realbg")
a=ap.parse_args()
W,H=512,896
OUT=f"/workspace/gallery/composite/{a.tag}.mp4"; os.makedirs(os.path.dirname(OUT),exist_ok=True)
lama=SimpleLama(); seg=new_session("isnet-general-use")

def read_all(path):
    cap=cv2.VideoCapture(path); fs=[]
    while True:
        ok,f=cap.read()
        if not ok: break
        fs.append(cv2.cvtColor(cv2.resize(f,(W,H)),cv2.COLOR_BGR2RGB))
    cap.release(); return fs
drive=read_all(a.drive); mask=read_all(a.mask); persona=read_all(a.persona)
N=min(len(drive),len(mask),len(persona))
print(f"frames: drive={len(drive)} mask={len(mask)} persona={len(persona)} -> {N}")
fps=cv2.VideoCapture(a.drive).get(cv2.CAP_PROP_FPS) or 24
vw=cv2.VideoWriter(OUT,cv2.VideoWriter_fourcc(*"mp4v"),fps,(W,H),True)
for i in range(N):
    # 1) real room plate: remove original dancer (generous) + burned-in caption band
    m=cv2.cvtColor(mask[i],cv2.COLOR_RGB2GRAY)
    m=(m>60).astype(np.uint8)*255
    m=cv2.dilate(m, np.ones((25,25),np.uint8), iterations=2)   # cover shadow/edges -> no ghost
    m[int(H*0.42):int(H*0.66), :]=255                          # caption strip -> also inpainted out
    plate=np.array(lama(Image.fromarray(drive[i]), Image.fromarray(m)).resize((W,H)).convert("RGB"))
    # 2) matte persona (alpha-matting for clean edges, erode to kill halo fringe)
    d=remove(Image.fromarray(persona[i]), session=seg, alpha_matting=True,
             alpha_matting_foreground_threshold=240, alpha_matting_background_threshold=15,
             alpha_matting_erode_size=8)
    pa=np.array(d); a=pa[:,:,3].astype(np.uint8)
    a=cv2.erode(a, np.ones((3,3),np.uint8), iterations=1)       # remove bright edge halo
    a=cv2.GaussianBlur(a,(3,3),0)
    alpha=(a.astype(np.float32)/255.0)[...,None]
    fg=pa[:,:,:3].astype(np.float32)
    # 3) composite
    comp=(fg*alpha + plate.astype(np.float32)*(1-alpha)).clip(0,255).astype(np.uint8)
    vw.write(cv2.cvtColor(comp,cv2.COLOR_RGB2BGR))
    if i%30==0: print(f"  {i}/{N}")
vw.release()
# re-encode h264 for browser + clear name in comfy output
os.system(f'ffmpeg -y -i "{OUT}" -c:v libx264 -pix_fmt yuv420p -crf 18 "/workspace/ComfyUI/output/{a.tag}.mp4" >/dev/null 2>&1')
print("done ->",OUT," and ComfyUI/output/"+a.tag+".mp4")
