#!/usr/bin/env python3
"""Build a grouped HTML gallery from the sweep, backfilling cached seed=42 clips."""
import json,os,shutil
import numpy as np, cv2
OUT="/workspace/gallery/sweep"
res=json.load(open(f"{OUT}/results.json"))
def metrics(path):
    cap=cv2.VideoCapture(path); prev=None; diffs=[]; sharp=[]
    while True:
        ok,f=cap.read()
        if not ok: break
        g=cv2.cvtColor(f,cv2.COLOR_BGR2GRAY).astype(np.float32)
        roi=np.concatenate([g[0:110,:].ravel(), g[110:430,0:70].ravel(), g[110:430,442:512].ravel()])
        if prev is not None: diffs.append(np.abs(roi-prev).mean())
        prev=roi
        sharp.append(cv2.Laplacian(g[300:750,150:380],cv2.CV_32F).var())
    cap.release(); return float(np.mean(diffs)), float(np.mean(sharp))
# backfill cached seed=42 clips (identical params already rendered earlier)
BACKFILL={
 "A_dist0.35_euler_ancestral_seed42.mp4":("/workspace/gallery/lora_matrix/distill0.35.mp4","A","distilled 0.35 · euler_ancestral  (резкий край)",0.35,"euler_ancestral"),
 "B_dist0.50_euler_ancestral_seed42.mp4":("/workspace/gallery/motion_out/bgvideo_notile.mp4","B","distilled 0.50 · euler_ancestral  (старая база)",0.50,"euler_ancestral"),
 "C_dist0.80_euler_ancestral_seed42.mp4":("/workspace/gallery/lora_matrix/distill0.8.mp4","C","distilled 0.80 · euler_ancestral  (ТВОЙ ВЫБОР)",0.80,"euler_ancestral"),
}
have={r["file"] for r in res}
for fn,(src,sid,label,dist,samp) in BACKFILL.items():
    if fn not in have and os.path.exists(src):
        shutil.copy(src,f"{OUT}/{fn}"); n,s=metrics(f"{OUT}/{fn}")
        res.append(dict(set=sid,label=label,seed=42,file=fn,bgNoise=n,sharp=s,distilled=dist,union=1.0,guide=0.75,sampler=samp))
# group
sets={}
for r in res: sets.setdefault(r["set"],{"label":r["label"],"rows":[]})["rows"].append(r)
for s in sets.values(): s["rows"].sort(key=lambda r:r["seed"])
html=['<meta charset=utf8><title>LTX sweep — стабильность по сидам</title>',
'<style>body{background:#111;color:#eee;font-family:system-ui,sans-serif;margin:0;padding:16px}'
'h1{font-size:18px;font-weight:600}h2{font-size:15px;margin:22px 0 6px;color:#9ad}'
'.stab{color:#8c8;font-size:12px;font-weight:400}.row{display:flex;gap:10px;flex-wrap:wrap}'
'.cell{background:#1b1b1e;border-radius:8px;padding:6px}video{width:220px;border-radius:6px;display:block}'
'.cap{font-size:11px;color:#aaa;margin-top:4px;text-align:center}</style>',
'<h1>LTX-2.3 · перебор гиперпараметров · 3 сида на набор (оценка стабильности)</h1>',
'<div style="font-size:12px;color:#888">bgNoise = дрожь фона (меньше=стабильнее) · sharp = высокочастотная деталь (больше=резче/зернистее)</div>']
for sid in sorted(sets):
    s=sets[sid]; rows=s["rows"]
    ns=[r["bgNoise"] for r in rows]; ss=[r["sharp"] for r in rows]
    stab=f"bgNoise {np.mean(ns):.2f}±{np.std(ns):.2f} · sharp {np.mean(ss):.0f}±{np.std(ss):.0f}  (±=разброс по сидам)"
    html.append(f'<h2>[{sid}] {s["label"]} &nbsp;<span class=stab>{stab}</span></h2>')
    html.append('<div class=row>')
    for r in rows:
        html.append(f'<div class=cell><video src="{r["file"]}" autoplay muted loop playsinline></video>'
                    f'<div class=cap>seed {r["seed"]} · bgNoise {r["bgNoise"]:.2f} · sharp {r["sharp"]:.0f}</div></div>')
    html.append('</div>')
open(f"{OUT}/index.html","w").write("\n".join(html))
print("gallery written:",f"{OUT}/index.html","sets:",len(sets),"clips:",len(res))
