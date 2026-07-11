"""NSFW test batch on unStable Revolution + universal finish (2K -> skin-refine -> FaceDetailer).
Synthetic 18+ persona per CLAUDE.md. Flow per PIPELINE_NSFW.md."""
import json, os, shutil, time
import urllib.request

API = "http://127.0.0.1:10100"
COMFY_OUT = "/workspace/ComfyUI/output"
COMFY_IN = "/workspace/ComfyUI/input"

PERSONA = "beautiful young woman with long honey-blonde wavy hair, hazel eyes, natural skin texture with visible pores and faint freckles"

SCENES = {
    "boudoir_topless": f"boudoir photo of a topless {PERSONA}, sitting on the edge of a bed in black lace panties, bare breasts, soft window light across her body, silk sheets, warm intimate atmosphere, photorealistic, sharp focus",
    "lingerie_mirror": f"sensual photo of a {PERSONA} in a sheer black lace lingerie set, standing by a full-length mirror in a dim bedroom, string lights, seductive gaze over her shoulder, photorealistic, high detail",
    "nude_window": f"artistic full nude photo of a {PERSONA}, standing by a tall window in morning light, fully nude, natural breasts and curves, side-lit body with soft shadows, tasteful sensual pose, photorealistic, sharp focus",
    "shower_nude": f"photo of a fully nude {PERSONA} in a walk-in shower, wet skin with water droplets and streams, wet hair slicked back, steam, glass wall with condensation, hands in her hair, photorealistic, high detail",
    "bed_explicit": f"explicit photo of a fully nude {PERSONA} lying on a bed with legs parted, nude spread pose, visible vulva, bare breasts, relaxed sensual expression, warm bedside lamp light, silk sheets, photorealistic, sharp focus",
    "ass_back": f"photo of a fully nude {PERSONA} kneeling on a bed facing away from camera, view from behind, bare back and round ass, looking back over her shoulder with a teasing smile, golden hour window light, photorealistic, high detail",
}

SEED = 21


def submit(wf):
    req = urllib.request.Request(API + "/prompt", json.dumps({"prompt": wf}).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as r:
        return json.loads(r.read())["prompt_id"]


def wait(pid, timeout=600):
    t0 = time.time()
    while time.time() - t0 < timeout:
        with urllib.request.urlopen(f"{API}/history/{pid}") as r:
            h = json.loads(r.read())
        if pid in h:
            e = h[pid]
            if e.get("status", {}).get("completed"):
                for node in e["outputs"].values():
                    for img in node.get("images", []):
                        return os.path.join(COMFY_OUT, img.get("subfolder", ""), img["filename"])
            if e.get("status", {}).get("status_str") == "error":
                raise RuntimeError(json.dumps(e["status"])[:400])
        time.sleep(2)
    raise TimeoutError(pid)


def main():
    base = json.load(open("wf_unstable.json"))
    finish = json.load(open("wf_finish_block.json"))
    os.makedirs("results/nsfw", exist_ok=True)

    for name, prompt in SCENES.items():
        wf = json.loads(json.dumps(base))
        wf["30"]["inputs"]["text"] = prompt
        wf["50"]["inputs"]["seed"] = SEED
        wf["52"]["inputs"]["filename_prefix"] = f"nsfw_base_{name}"
        t = time.time()
        out1 = wait(submit(wf))
        print(f"[nsfw] base {name}: {time.time()-t:.0f}s", flush=True)

        src = f"nsfw_src_{name}.png"
        shutil.copy(out1, os.path.join(COMFY_IN, src))
        wf2 = json.loads(json.dumps(finish))
        wf2["5"]["inputs"]["image"] = src
        wf2["55"]["inputs"]["seed"] = SEED + 1
        wf2["61"]["inputs"]["seed"] = SEED + 2
        wf2["62"]["inputs"]["filename_prefix"] = f"nsfw_fin_{name}"
        t = time.time()
        out2 = wait(submit(wf2))
        print(f"[nsfw] finish {name}: {time.time()-t:.0f}s -> {out2}", flush=True)
        shutil.copy(out2, f"results/nsfw/{name}.png")

    print("[nsfw] done", flush=True)


if __name__ == "__main__":
    main()
