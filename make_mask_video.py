#!/usr/bin/env python3
"""Per-frame silhouette mask of the driving dancer -> mask video for LTX inpaint.
White = regenerate (persona region), black = keep real background pixels.
Output: ComfyUI/input/drive_mask.mp4  (512x896, matches drive_dzp.mp4)
"""
import cv2, numpy as np, os
from rembg import remove, new_session
from PIL import Image
DRIVE="/workspace/ComfyUI/input/drive_dzp.mp4"
OUT="/workspace/ComfyUI/input/drive_mask.mp4"
W,H=512,896
sess=new_session("u2net_human_seg")
cap=cv2.VideoCapture(DRIVE)
fps=cap.get(cv2.CAP_PROP_FPS) or 24
fourcc=cv2.VideoWriter_fourcc(*"mp4v")
vw=cv2.VideoWriter(OUT,fourcc,fps,(W,H),isColor=True)
n=0
kernel=np.ones((21,21),np.uint8)
while True:
    ok,f=cap.read()
    if not ok: break
    rgb=cv2.cvtColor(cv2.resize(f,(W,H)),cv2.COLOR_BGR2RGB)
    d=remove(Image.fromarray(rgb),session=sess)
    a=np.array(d)[:,:,3]
    m=(a>30).astype(np.uint8)*255
    m=cv2.dilate(m,kernel,iterations=2)          # generous margin so persona fits
    m=cv2.GaussianBlur(m,(9,9),0)
    vw.write(cv2.cvtColor(m,cv2.COLOR_GRAY2BGR))
    n+=1
cap.release(); vw.release()
print(f"mask video written {OUT}  frames={n}")
