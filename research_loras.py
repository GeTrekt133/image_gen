#!/usr/bin/env python3
"""Полная перепись Z-Image LoRA на Civitai (UA + курсор-пагинация).
Дедуп по id, классификация NSFW/база. -> /workspace/loras_research.json"""
import os, json, time, urllib.request, urllib.parse

TOK=os.environ["CIVITAI_TOKEN"]
UA="Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"
def api(params):
    url="https://civitai.com/api/v1/models?"+urllib.parse.urlencode(params)
    req=urllib.request.Request(url,headers={"Authorization":f"Bearer {TOK}","User-Agent":UA})
    for _ in range(4):
        try:
            with urllib.request.urlopen(req,timeout=60) as r: return json.load(r)
        except Exception as e: last=e; time.sleep(3)
    print("  api fail:",last); return {"items":[],"metadata":{}}

def crawl(query,maxpages=10):
    got=[]; cursor=None
    for _ in range(maxpages):
        p={"limit":100,"types":"LORA","query":query,"sort":"Most Downloaded","nsfw":"true"}
        if cursor: p["cursor"]=cursor
        d=api(p); items=d.get("items",[]); got+=items
        cursor=d.get("metadata",{}).get("nextCursor")
        if not items or not cursor: break
        time.sleep(0.6)
    return got

seen={}
for q in ["z-image","zimage","z image","z-image nsfw","z-image turbo"]:
    items=crawl(q); print(f"query '{q}': {len(items)} raw")
    for m in items:
        mid=m["id"]
        if mid in seen: continue
        vers=m.get("modelVersions",[])
        zit=[v for v in vers if (v.get("baseModel") or "").replace(" ","").lower()=="zimageturbo"]
        base=[v for v in vers if (v.get("baseModel") or "").replace(" ","").lower()=="zimagebase"]
        stats=m.get("stats",{})
        seen[mid]={"id":mid,"name":m.get("name"),"nsfw_flag":m.get("nsfw"),
            "nsfwLevel":m.get("nsfwLevel",0),
            "downloads":stats.get("downloadCount",0),"thumbsUp":stats.get("thumbsUpCount",0),
            "base_latest":(vers[0].get("baseModel") if vers else None),
            "has_turbo":bool(zit),"turbo_ver_id":(zit[0].get("id") if zit else None),
            "turbo_ver_name":(zit[0].get("name") if zit else None),
            "has_base_only":bool(base) and not bool(zit),
            "tags":[t.lower() for t in m.get("tags",[])],"url":f"https://civitai.com/models/{mid}"}
    time.sleep(0.8)

rows=list(seen.values())
NSFW_TAGS={"nsfw","porn","sex","explicit","hentai","nude","pussy","anal","blowjob","cumshot","erotic","naked","tits","vagina"}
def tier(r):
    lv=r["nsfwLevel"] or 0; hard=bool(lv & 16); x=bool(lv & 8)
    nude=bool(set(r["tags"]) & NSFW_TAGS)
    if hard: return "XXX"
    if x or nude: return "X-nude"
    if (lv & 4): return "R-soft"
    if bool(r["nsfw_flag"]) or (lv & 2): return "sexy"
    return "sfw"
for r in rows: r["tier"]=tier(r)
rows.sort(key=lambda x:-x["downloads"])
json.dump(rows,open("/workspace/loras_research.json","w"),ensure_ascii=False,indent=2)

turbo=[r for r in rows if r["has_turbo"]]
from collections import Counter
tc=Counter(r["tier"] for r in turbo)
nsfw_turbo=[r for r in turbo if r["tier"] in ("XXX","X-nude","R-soft")]
print(f"\n=== уникальных Z-Image LoRA: {len(rows)}")
print(f"=== с ZImageTurbo-версией: {len(turbo)}")
print(f"=== тиры (Turbo): {dict(tc)}")
print(f"=== NSFW (R-soft+X+XXX, Turbo): {len(nsfw_turbo)}")
print(f"\nTOP NSFW+Turbo по загрузкам:")
print(f"{'DL':>7} {'👍':>5} {'tier':<7} {'lvl':>3}  id        name")
for r in nsfw_turbo[:35]:
    print(f"{r['downloads']:>7} {r['thumbsUp']:>5} {r['tier']:<7} {r['nsfwLevel']:>3}  {r['id']:<9} {r['name'][:44]}")
