import json,time,os,shutil,urllib.request,copy
HOST="http://127.0.0.1:10100"; COMFY_OUT="/workspace/ComfyUI/output"; OUT="/workspace/gallery/reel_out"
os.makedirs(OUT,exist_ok=True)
BASE=json.load(open("/workspace/wf_ltx_dance.json"))
order=json.load(open("/workspace/reel_shots/order.json"))
def post(g): return json.load(urllib.request.urlopen(urllib.request.Request(HOST+"/prompt",data=json.dumps({"prompt":g}).encode(),headers={"Content-Type":"application/json"}),timeout=120))
def getj(p): return json.load(urllib.request.urlopen(HOST+p,timeout=60))
PROMPT="photorealistic video of a beautiful 24 year old woman with long brown hair, natural glowing skin, tight white outfit, cinematic lighting, highly detailed, smooth motion"
for idx in order:
    g=copy.deepcopy(BASE)
    g["5001"]["inputs"]["file"]=f"drive_{idx}.mp4"
    g["2004"]["inputs"]["image"]="persona_ref.png"       # full-body persona anchor
    g["2483"]["inputs"]["text"]=PROMPT
    g["5011"]["inputs"]["strength_model"]=1.0            # union
    g["5012"]["inputs"]["strength"]=0.75                 # guide
    g["4832"]["inputs"]["noise_seed"]=42                 # единый сид -> консистентность персоны
    t0=time.time()
    try: pid=post(g)["prompt_id"]
    except Exception as e: print(f"shot{idx} SUBMIT_FAIL {str(e)[:120]}"); continue
    while time.time()-t0<900:
        time.sleep(8); h=getj(f"/history/{pid}")
        if pid in h:
            st=h[pid]["status"]
            if st.get("status_str")=="error":
                print(f"shot{idx} ERROR",[str(m)[:120] for m in st.get("messages",[]) if 'error' in str(m).lower()][:1]); break
            if st.get("status_str")=="success" or st.get("completed"): break
    mp4=sorted([f for f in os.listdir(COMFY_OUT) if f.startswith("ltx_dance") and f.endswith(".mp4")],key=lambda f:os.path.getmtime(os.path.join(COMFY_OUT,f)))[-1]
    shutil.copy(os.path.join(COMFY_OUT,mp4),f"{OUT}/shot_{idx}.mp4")
    print(f"shot{idx} OK {time.time()-t0:.0f}s -> shot_{idx}.mp4")
print("REEL_GEN_DONE")
