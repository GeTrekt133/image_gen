#!/usr/bin/env python3
"""HTML-сетка сравнения NSFW-LoRA вариантов (строки=промпты, колонки=LoRA)."""
import json, os, base64, io
from PIL import Image
GAL="/workspace/gallery/lora_compare"; OUTDIR="/workspace/gallery/artifact"; os.makedirs(OUTDIR,exist_ok=True)
OUT=f"{OUTDIR}/lora_compare.html"
res=json.load(open(f"{GAL}/results.json"))
VARORDER=["raw","nsfw23k","godpussy","pussy","eason_v31","nsfw_v2","unstable"]
VLABEL={"raw":"raw (без LoRA)","nsfw23k":"NSFW 23K · 2279079","godpussy":"GodPussy · 2222911","pussy":"z-image-pussy · 2205140","eason_v31":"Eason V3.1 · 2359268","nsfw_v2":"NSFW V2 · 2299623","unstable":"unStable Revolution ⭐ (ckpt · 2193942)"}
prompts=[]
for r in res:
    if r["prompt"] not in prompts: prompts.append(r["prompt"])
idx={(r["prompt"],r["variant"]):r for r in res}
def uri(p,maxw=520,q=87):
    im=Image.open(p).convert("RGB")
    if im.width>maxw: im=im.resize((maxw,int(im.height*maxw/im.width)),Image.LANCZOS)
    b=io.BytesIO(); im.save(b,"JPEG",quality=q); return "data:image/jpeg;base64,"+base64.b64encode(b.getvalue()).decode()

secs=[]
for pk in prompts:
    cells=[]
    for v in VARORDER:
        r=idx.get((pk,v));
        p=os.path.join(GAL,f"{pk}__{v}.png")
        if r and os.path.exists(p):
            cells.append(f'<figure class="cell"><div class="badge">{VLABEL[v]}</div><img src="{uri(p)}"></figure>')
        else:
            cells.append('<div class="cell empty">—</div>')
    secs.append(f'<section><h2>{pk}</h2><div class="grid">{"".join(cells)}</div></section>')

html=f'''<div class="wrap">
<header><h1>NSFW-LoRA · сравнение вариантов</h1>
<p class="sub">Base t2i (Z-Image Turbo, 8 шагов cfg1, тот же сид) — изолирован эффект NSFW-LoRA @1.0. Синтетическая 18+ персона.</p>
<p class="sub">Census Civitai: ~847 NSFW Z-Image-Turbo LoRA (XXX 348 · X-nude 165 · R-soft 334). Здесь — топ фотореалистичных по загрузкам.</p></header>
{"".join(secs)}
<footer>PNG: <code>/workspace/gallery/lora_compare/</code> · census: <code>/workspace/loras_research.json</code></footer></div>'''
css='''*{box-sizing:border-box}body{margin:0}
.wrap{font-family:-apple-system,Segoe UI,Roboto,sans-serif;background:#0e0f13;color:#e7e7ea;max-width:1500px;margin:0 auto;padding:26px 20px 60px}
h1{font-size:24px;margin:0 0 6px}.sub{color:#9aa0aa;margin:0 0 6px;font-size:13px}
section{margin:22px 0}h2{font-size:18px;text-transform:capitalize;margin:0 0 10px}
.grid{display:grid;grid-template-columns:repeat(7,1fr);gap:9px}@media(max-width:1100px){.grid{grid-template-columns:repeat(3,1fr)}}
.cell{position:relative;background:#15171d;border:1px solid #23262f;border-radius:9px;overflow:hidden;margin:0}.cell img{width:100%;display:block}
.badge{position:absolute;top:6px;left:6px;right:6px;background:rgba(10,11,15,.8);color:#dfe3ea;font-size:10.5px;font-weight:600;padding:3px 7px;border-radius:5px;text-align:center}
.cell.empty{min-height:180px;display:flex;align-items:center;justify-content:center;color:#5f636d}
footer{margin-top:24px;color:#6b6f79;font-size:12px}code{color:#9fd0ff}'''
open(OUT,"w").write(f"<!doctype html><html><head><meta charset='utf-8'><title>NSFW-LoRA compare</title><style>{css}</style></head><body>{html}</body></html>")
print("wrote",OUT,f"{os.path.getsize(OUT)//1024} KB")
