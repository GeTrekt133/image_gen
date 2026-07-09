#!/usr/bin/env python3
"""Build a composite i2v anchor = persona (from images.jpg) placed into the driving
video's real room background (dancer + caption inpainted out).
Output: /workspace/ComfyUI/input/anchor_bgvideo.png  (512x896)
"""
import numpy as np, cv2, os
from PIL import Image
from rembg import remove, new_session
from simple_lama_inpainting import SimpleLama

W,H = 512,896
DRIVE = "/workspace/ComfyUI/input/drive_dzp.mp4"
PERSONA = "/workspace/images.jpg"
work = "/workspace/vid_work"; os.makedirs(work, exist_ok=True)

# --- 1. grab a driving frame (already 512x896) as the room source ---
cap = cv2.VideoCapture(DRIVE)
cap.set(cv2.CAP_PROP_POS_FRAMES, 8)   # a settled frame
ok, frame = cap.read(); cap.release()
assert ok, "no frame"
room_bgr = cv2.resize(frame, (W,H))
room_rgb = cv2.cvtColor(room_bgr, cv2.COLOR_BGR2RGB)
Image.fromarray(room_rgb).save(f"{work}/room_raw.png")

sess = new_session("u2net_human_seg")

# --- 2. mask the blonde dancer in the room frame -> dilate -> inpaint her out ---
d = remove(Image.fromarray(room_rgb), session=sess)          # RGBA, alpha=person
alpha = np.array(d)[:,:,3]
mask = (alpha > 30).astype(np.uint8)*255
mask = cv2.dilate(mask, np.ones((25,25),np.uint8), iterations=2)  # cover hair/edges
# also mask the burned-in caption band (roughly vertical-center, full width)
cap_mask = np.zeros((H,W), np.uint8)
cap_mask[int(H*0.42):int(H*0.66), :] = 255
full_mask = np.clip(mask + cap_mask, 0, 255).astype(np.uint8)
Image.fromarray(full_mask).save(f"{work}/room_mask.png")

lama = SimpleLama()
room_clean = lama(Image.fromarray(room_rgb), Image.fromarray(full_mask))  # RGB PIL
room_clean = room_clean.resize((W,H)).convert("RGB")
room_clean.save(f"{work}/room_clean.png")

# --- 3. cut the persona out of images.jpg ---
p = Image.open(PERSONA).convert("RGB")
p_cut = remove(p, session=new_session("u2net"))               # RGBA (general seg keeps dress)
p_cut.save(f"{work}/persona_cut.png")

# --- 4. scale persona to full-body dance framing & composite into clean room ---
# target: persona height ~ 0.92*H, horizontally centered, feet near bottom
pa = np.array(p_cut); pa_alpha = pa[:,:,3]
ys,xs = np.where(pa_alpha>10)
y0,y1,x0,x1 = ys.min(),ys.max(),xs.min(),xs.max()
p_crop = p_cut.crop((x0,y0,x1+1,y1+1))
target_h = int(H*0.92)
scale = target_h / p_crop.height
target_w = int(p_crop.width*scale)
p_scaled = p_crop.resize((target_w, target_h), Image.LANCZOS)

# --- 4b. mild color-harmonization: cast persona toward the room's ambient light ---
room_avg = np.array(room_clean).reshape(-1,3).mean(0)      # ambient color of the room
ps = np.array(p_scaled).astype(np.float32)
rgb, a = ps[:,:,:3], ps[:,:,3:4]
tint = 0.22                                                 # blend toward ambient
rgb = rgb*(1-tint) + room_avg[None,None,:]*tint
rgb *= 0.90                                                 # match dimmer room exposure
ps[:,:,:3] = np.clip(rgb,0,255)
p_scaled = Image.fromarray(ps.astype(np.uint8),"RGBA")

canvas = room_clean.convert("RGBA")
px = (W - target_w)//2
py = H - target_h - int(H*0.02)     # feet slightly above bottom edge
canvas.alpha_composite(p_scaled, (px, py))
out = canvas.convert("RGB")
out.save("/workspace/ComfyUI/input/anchor_bgvideo.png")
out.save(f"{work}/anchor_bgvideo.png")
print("composite saved ->", "/workspace/ComfyUI/input/anchor_bgvideo.png", out.size)
