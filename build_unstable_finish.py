#!/usr/bin/env python3
"""ABCD HTML для unStable + finish (gallery/unstable_finish)."""
import json, os, base64, io
from PIL import Image, ImageChops
GAL="/workspace/gallery/unstable_finish"; OUTDIR="/workspace/gallery/artifact"; os.makedirs(OUTDIR,exist_ok=True)
OUT=f"{OUTDIR}/unstable_finish.html"
results=json.load(open(f"{GAL}/results.json"))
STAGES=[("A_unstable","A · unStable raw","полный NSFW-чекпоинт, без LoRA"),
        ("B_facedetailer","B · +FaceDetailer","face_yolov8m, инпейнт лица (denoise 0.45)"),
        ("C_final","C · +2K refine","4x-UltraSharp→×2 → img2img 0.28 (мягкий, анти-пластик)")]
def uri(p,maxw=760,q=87):
    im=Image.open(p).convert("RGB")
    if im.width>maxw: im=im.resize((maxw,int(im.height*maxw/im.width)),Image.LANCZOS)
    b=io.BytesIO(); im.save(b,"JPEG",quality=q); return "data:image/jpeg;base64,"+base64.b64encode(b.getvalue()).decode()
def dims(p): w,h=Image.open(p).size; return f"{w}×{h}"
def cell(p,badge,desc):
    if not os.path.exists(p): return '<div class="cell empty">нет файла</div>'
    return f'<figure class="cell"><div class="badge">{badge}</div><img src="{uri(p)}"><figcaption>{desc}<span class="dim">{dims(p)}</span></figcaption></figure>'
def fd_noop(k):
    b=f"{GAL}/{k}__A_unstable.png"; c=f"{GAL}/{k}__B_facedetailer.png"
    if not(os.path.exists(b) and os.path.exists(c)): return False
    B=Image.open(b).convert("RGB"); C=Image.open(c).convert("RGB")
    return B.size==C.size and ImageChops.difference(B,C).getbbox() is None
runs=[]
for r in results:
    k=r["key"]; cells=[cell(f"{GAL}/{k}__{sid}.png",t,d) for sid,t,d in STAGES]
    note='<p class="note">⚠ FaceDetailer no-op: лицо вне кадра/мелкое → A=B.</p>' if fd_noop(k) else ""
    runs.append(f'<section class="run"><h2>{k} <span class="meta">{r["w"]}×{r["h"]} · seed {r["seed"]} · {r["seconds"]}s · пик {r["peak_vram_gb"]} GB</span></h2><div class="grid">{"".join(cells)}</div>{note}</section>')
html=f'''<div class="wrap">
<header><h1>unStable Revolution + finish</h1>
<p class="sub">База — полный NSFW-чекпоинт <b>unStable Revolution ZIT v3 fp16</b> (2193942). Финиш: FaceDetailer → 2K refine (denoise 0.28). <b>Realistic LoRA убрана</b> (конфликтовала с весами чекпоинта → галлюцинации). Синтетическая 18+ персона. RTX PRO 6000 96GB · 2026-07-08</p>
<div class="chips"><span>unStable Revolution ZIT v3 fp16 · 8 шагов cfg1 res_multistep</span><span>FaceDetailer face_yolov8m (bbox_thr 0.4)</span><span>4x-UltraSharp → 2K · refine denoise 0.28</span></div></header>
{"".join(runs)}
<footer>PNG: <code>/workspace/gallery/unstable_finish/&lt;key&gt;__&lt;stage&gt;.png</code> · workflow: <code>/workspace/wf_unstable_finish.json</code></footer></div>'''
css='''*{box-sizing:border-box}body{margin:0}
.wrap{font-family:-apple-system,Segoe UI,Roboto,sans-serif;background:#0e0f13;color:#e7e7ea;max-width:1400px;margin:0 auto;padding:28px 22px 60px}
h1{font-size:25px;margin:0 0 6px}.sub{color:#9aa0aa;margin:0 0 14px;font-size:14px}
.chips{display:flex;flex-wrap:wrap;gap:8px;margin-bottom:20px}.chips span{background:#1b1d24;border:1px solid #2a2d37;color:#c3c8d2;font-size:12px;padding:4px 10px;border-radius:20px}
.run{margin-bottom:30px}.run h2{font-size:18px;margin:0 0 10px;text-transform:capitalize}.run .meta{font-size:12.5px;color:#8b909b;font-weight:400;text-transform:none}
.grid{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}@media(max-width:900px){.grid{grid-template-columns:repeat(2,1fr)}}
.cell{margin:0;background:#15171d;border:1px solid #23262f;border-radius:10px;overflow:hidden;position:relative}.cell img{width:100%;display:block}
.badge{position:absolute;top:8px;left:8px;background:rgba(10,11,15,.82);color:#dfe3ea;font-size:11.5px;font-weight:600;padding:3px 9px;border-radius:6px}
figcaption{padding:8px 10px;font-size:11.5px;color:#9aa0aa;line-height:1.4}.dim{display:block;color:#5f636d;margin-top:3px}
.cell.empty{display:flex;align-items:center;justify-content:center;color:#5f636d;min-height:200px}
.note{margin:10px 0 0;background:#1d1a12;border:1px solid #43391d;color:#d8c48c;font-size:12.5px;padding:9px 12px;border-radius:8px}
footer{margin-top:26px;color:#6b6f79;font-size:12px}code{color:#9fd0ff}'''
open(OUT,"w").write(f"<!doctype html><html><head><meta charset='utf-8'><title>unStable + finish ABCD</title><style>{css}</style></head><body>{html}</body></html>")
print("wrote",OUT,f"{os.path.getsize(OUT)//1024} KB")
