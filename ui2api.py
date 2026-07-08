#!/usr/bin/env python3
"""UI→API конвертер ComfyUI-графов (без сабграфов) со всеми выученными фиксами:
- виджеты, сконвертированные-в-инпут, занимают слот в widgets_values (сдвиг)
- seed: пропуск control_after_generate
- V3 DynamicCombo (ResizeImageMaskNode): плоские точечные ключи
Usage: python ui2api.py <ui.json> <api.json>"""
import json,sys,urllib.request

def convert(src,dst,host="http://127.0.0.1:10100"):
    oi=json.load(urllib.request.urlopen(f"{host}/object_info",timeout=120))
    w=json.load(open(src)); nodes=w["nodes"]; links=w["links"]
    lmap={l[0]:(str(l[1]),l[2]) for l in links}
    WT={"INT","FLOAT","STRING","BOOLEAN"}; CTRL={"fixed","increment","decrement","randomize"}
    def is_widget(spec):
        t=spec[0]; return isinstance(t,list) or t in WT
    api={}
    for n in nodes:
        ct=n.get("type")
        if ct in ("Note","MarkdownNote","Reroute"): continue
        nid=str(n["id"]); d=oi.get(ct,{}).get("input",{})
        order=[(k,v) for k,v in d.get("required",{}).items()]+[(k,v) for k,v in d.get("optional",{}).items()]
        linked={}
        for slot in n.get("inputs",[]):
            if slot.get("link") in lmap: linked[slot["name"]]=list(lmap[slot["link"]])
        wv=list(n.get("widgets_values") or []); wi=0; inp={}
        for name,spec in order:
            wdg=is_widget(spec)
            if name in linked:
                inp[name]=linked[name]
                if wdg: wi+=1
                continue
            if not wdg: continue
            t=spec[0]
            # V3 dynamic combo: options list of dicts with sub-inputs
            if isinstance(t,str) and t.startswith("COMFY_DYNAMICCOMBO"):
                sel=wv[wi] if wi<len(wv) else None; wi+=1
                inp[name]=sel
                # consume sub-widgets by selected option schema
                opts=spec[1].get("options",[]) if len(spec)>1 else []
                for o in opts:
                    if o.get("key")==sel:
                        for sk,sv in o.get("inputs",{}).get("required",{}).items():
                            if wi<len(wv): inp[f"{name}.{sk}"]=wv[wi]; wi+=1
                continue
            if isinstance(t,str) and t.startswith("COMFY_MATCHTYPE"):
                continue  # connection-only
            if wi<len(wv):
                inp[name]=wv[wi]; wi+=1
                if name in ("seed","noise_seed") and wi<len(wv) and isinstance(wv[wi],str) and wv[wi] in CTRL: wi+=1
        for name,v in linked.items(): inp.setdefault(name,v)
        api[nid]={"class_type":ct,"inputs":inp}
    json.dump(api,open(dst,"w"),indent=1)
    return api

if __name__=="__main__":
    api=convert(sys.argv[1],sys.argv[2])
    print(f"converted {len(api)} nodes -> {sys.argv[2]}")
