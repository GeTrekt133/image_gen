#!/usr/bin/env python3
"""Add a Stage-2 low-noise refine tail on top of the VACE Stage-1 (distilled 8-step).
Stage-2 re-noises the Stage-1 latent to sigma[0] and denoises the short tail -> sharpen,
preserving composition/bg/motion. Masked so only the subject refines (bg stays real)."""
import json,urllib.request,time,os,copy,shutil
HOST="http://127.0.0.1:10100"; COMFY_OUT="/workspace/ComfyUI/output"; OUT="/workspace/gallery/sweep"
G=json.load(open("/workspace/wf_vace_graft.json")); G["5001"]["inputs"]["file"]="drive_clean.mp4"; G["4922"]["inputs"]["strength_model"]=0.9
G["2483"]["inputs"]["text"]=("a beautiful young woman with long wavy brown hair wearing a light blue and white vertical pinstripe shirt dress, dancing, pink and purple ambient room light, photorealistic, highly detailed, natural skin")
def post(g): return json.load(urllib.request.urlopen(urllib.request.Request(HOST+"/prompt",data=json.dumps({"prompt":g}).encode(),headers={"Content-Type":"application/json"}),timeout=120))
def getj(p): return json.load(urllib.request.urlopen(HOST+p,timeout=60))
def add_stage2(g, tail, sampler="euler_cfg_pp"):
    # stage-1 video latent = 5013.2 ; re-wrap with audio, re-apply subject mask, refine
    g["7300"]={"class_type":"LTXVConcatAVLatent","inputs":{"video_latent":["5013",2],"audio_latent":["3980",0]}}
    g["7301"]={"class_type":"RandomNoise","inputs":{"noise_seed":42}}
    g["7302"]={"class_type":"KSamplerSelect","inputs":{"sampler_name":sampler}}
    g["7303"]={"class_type":"ManualSigmas","inputs":{"sigmas":tail}}
    g["7304"]={"class_type":"CFGGuider","inputs":{"model":["5011",0],"positive":["1241",0],"negative":["1241",1],"cfg":1}}
    g["7305"]={"class_type":"SamplerCustomAdvanced","inputs":{"noise":["7301",0],"guider":["7304",0],"sampler":["7302",0],"sigmas":["7303",0],"latent_image":["7300",0]}}
    g["7306"]={"class_type":"LTXVSeparateAVLatent","inputs":{"av_latent":["7305",0]}}
    g["5065"]["inputs"]["latents"]=["7306",0]   # decode stage-2 output
    return g
CONFIGS=[("2stage_inpaint_tail","0.725, 0.4219, 0.0"),("2stage_t2v_tail","0.85, 0.725, 0.4219, 0.0")]
for tag,tail in CONFIGS:
    g=add_stage2(copy.deepcopy(G), tail); g["4852"]["inputs"]["filename_prefix"]=tag
    before=set(os.listdir(COMFY_OUT)); t0=time.time()
    try: pid=post(g)["prompt_id"]
    except urllib.error.HTTPError as e: print(tag,"ERR",json.loads(e.read()).get("error",{}).get("message")[:200]); continue
    while time.time()-t0<900:
        time.sleep(4); h=getj(f"/history/{pid}")
        if pid in h and (h[pid]["status"].get("completed") or h[pid]["status"].get("status_str") in ("success","error")):
            if h[pid]["status"].get("status_str")=="error": print(tag,"RUNERR",[str(m)[:200] for m in h[pid]["status"].get("messages",[]) if 'error' in str(m).lower()][:1]); break
            new=[f for f in os.listdir(COMFY_OUT) if f.endswith(".mp4") and f not in before]
            shutil.copy(os.path.join(COMFY_OUT,sorted(new,key=lambda f:os.path.getmtime(os.path.join(COMFY_OUT,f)))[-1]),f"{OUT}/{tag}.mp4")
            print(f"{tag} OK {time.time()-t0:.0f}s"); break
print("TWOSTAGE_DONE")
