"""Adult-anchored NSFW test on CLEAN Z-Image Turbo (not unStable) + z-image finish.
Goal: check whether strong adult anchoring yields unambiguously-adult appearance.
cfg=1 zeros the negative, so ALL age anchoring lives in the positive prompt."""
import json, os, shutil, time
import urllib.request

API = "http://127.0.0.1:10100"
COMFY_OUT = "/workspace/ComfyUI/output"
COMFY_IN = "/workspace/ComfyUI/input"

# Heavy adult anchoring: explicit age + MATURE build markers (face + body), since cfg1 ignores negatives.
PERSONA = ("a 32-year-old mature adult woman, clearly adult grown woman with a full curvy womanly figure, "
           "wide hips, full large bust, thick thighs, mature adult facial features with defined cheekbones "
           "and laugh lines, confident grown-woman presence, long honey-blonde wavy hair, hazel eyes, "
           "natural skin texture with pores and freckles")

SCENES = {
    "boudoir_topless": f"boudoir photo of a topless {PERSONA}, sitting on the edge of a bed in black lace panties, bare breasts, soft window light, silk sheets, photorealistic, sharp focus",
    "lingerie_mirror": f"sensual photo of {PERSONA} in a black lace lingerie set, standing by a full-length mirror in a dim bedroom, string lights, confident gaze over her shoulder, photorealistic, high detail",
    "nude_window": f"artistic full nude photo of {PERSONA}, standing by a tall window in morning light, fully nude, full mature curves, side-lit body with soft shadows, tasteful pose, photorealistic, sharp focus",
    "shower_nude": f"photo of a fully nude {PERSONA} in a walk-in shower, wet skin with water droplets, wet hair slicked back, steam, hands in her hair, photorealistic, high detail",
    "bed_explicit": f"explicit photo of a fully nude {PERSONA} reclining on a bed, full mature curvy body, bare breasts, relaxed confident expression, warm bedside lamp light, silk sheets, photorealistic, sharp focus",
    "ass_back": f"photo of a fully nude {PERSONA} kneeling on a bed facing away, view from behind, full round hips and ass, looking back over her shoulder, golden hour light, photorealistic, high detail",
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
    base = json.load(open("wf_zimage.json"))          # clean z_image_turbo_bf16
    finish = json.load(open("wf_finish_zimage.json"))  # clean z-image finish
    os.makedirs("results/nsfw_adult", exist_ok=True)

    for name, prompt in SCENES.items():
        wf = json.loads(json.dumps(base))
        wf["4"]["inputs"]["text"] = prompt
        wf["8"]["inputs"]["seed"] = SEED
        wf["10"]["inputs"]["filename_prefix"] = f"za_base_{name}"
        out1 = wait(submit(wf))
        print(f"[za] base {name} ok", flush=True)

        src = f"za_src_{name}.png"
        shutil.copy(out1, os.path.join(COMFY_IN, src))
        wf2 = json.loads(json.dumps(finish))
        wf2["5"]["inputs"]["image"] = src
        wf2["23"]["inputs"]["seed"] = SEED + 1
        wf2["36"]["inputs"]["seed"] = SEED + 2
        wf2["38"]["inputs"]["filename_prefix"] = f"za_fin_{name}"
        out2 = wait(submit(wf2))
        print(f"[za] finish {name} -> {out2}", flush=True)
        shutil.copy(out2, f"results/nsfw_adult/{name}.png")

    print("[za] done", flush=True)


if __name__ == "__main__":
    main()
