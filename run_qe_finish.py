import json,time,os,shutil,urllib.request,copy
HOST="http://127.0.0.1:10100"; COMFY_OUT="/workspace/ComfyUI/output"; OUT="/workspace/gallery/qwen_finish"
base=json.load(open("/workspace/wf_qwen_edit_finish.json"))
def post(x): return json.load(urllib.request.urlopen(urllib.request.Request(HOST+"/prompt",data=json.dumps({"prompt":x}).encode(),headers={"Content-Type":"application/json"}),timeout=60))
def getj(p): return json.load(urllib.request.urlopen(HOST+p,timeout=60))
from PIL import Image
STAGES={"33":"A_edit","53":"B_2k","57":"C_skin","62":"D_final"}
JOBS=[
 ("insert","edit_insert.png",8011,"She is inserting a realistic flesh-colored dildo sex toy into her pussy with her hand, gripping the shaft, the tip penetrating between her labia into her vagina. Keep everything else identical: same face, body, pose, lighting and background."),
 ("suck","edit_suck.png",8021,"She is sucking a realistic flesh-colored dildo sex toy, holding it to her mouth with one hand, her lips wrapped around the tip, tongue out. Keep everything else identical: same face, hair, body, lighting and background."),
]
def run(tag,src,seed,prompt):
    g=copy.deepcopy(base); g["5"]["inputs"]["image"]=src; g["20"]["inputs"]["prompt"]=prompt; g["31"]["inputs"]["seed"]=seed
    t0=time.time(); r=post(g); pid=r["prompt_id"]; outs={}
    while time.time()-t0<900:
        time.sleep(3); h=getj(f"/history/{pid}")
        if pid in h:
            st=h[pid]["status"]
            if st.get("status_str")=="error": print(tag,"ERROR",str(st.get("messages"))[:300]); return
            if st.get("status_str")=="success" or st.get("completed"): outs=h[pid]["outputs"]; break
    for nid,stage in STAGES.items():
        im=outs.get(nid,{}).get("images",[])
        if im: shutil.copy(os.path.join(COMFY_OUT,im[0].get("subfolder",""),im[0]["filename"]), f"{OUT}/{tag}__{stage}.png")
    print(f"{tag:8s} OK {time.time()-t0:5.1f}s stages:{[s for n,s in STAGES.items() if os.path.exists(f'{OUT}/{tag}__{s}.png')]}")
for tag,src,seed,prompt in JOBS: run(tag,src,seed,prompt)
print("DONE")
