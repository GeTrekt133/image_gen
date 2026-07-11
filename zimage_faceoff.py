"""Face-off Z-Image side: repo flow wf_zimage.json (8-step t2i) -> wf_finish_zimage.json
(FaceDetailer + 4x-UltraSharp -> 2K refine). Talks to ComfyUI API on :10100."""
import json, time, shutil, sys, os
import urllib.request

API = "http://127.0.0.1:10100"
COMFY_OUT = "/workspace/ComfyUI/output"
COMFY_IN = "/workspace/ComfyUI/input"

PERSONA = "young woman with long honey-blonde wavy hair, hazel eyes, natural skin texture with pores and faint freckles"

SCENES = {
    "selfie": f"casual handheld phone selfie of a {PERSONA}, white ribbed tank top, thin gold necklace, genuine smile, golden afternoon window light, cozy apartment with beige sofa and monstera plant behind, slight wide-angle phone look, realistic amateur photo, sharp focus",
    "gym": f"candid fitness photo of a {PERSONA}, high ponytail, dusty-rose sports bra and black leggings, light sweat sheen, holding steel water bottle, modern gym with dumbbell racks and big windows behind, bright daylight, 35mm lens, high detail, sharp focus",
    "street": f"street style photo of a {PERSONA}, walking across a european city crosswalk, oversized camel blazer, white t-shirt, straight-leg jeans, white sneakers, small black shoulder bag, overcast daylight, old stone buildings and bicycles behind, deep depth of field, editorial candid, high detail, sharp focus",
    "swimsuit": f"golden hour photo of a {PERSONA} with damp wavy hair, waist-deep at an infinity pool edge overlooking the sea, matte sage-green one-piece swimsuit, water droplets on shoulders, laughing, turquoise sea and sunset sky behind, warm rim light, 50mm lens, high detail, sharp focus",
    "restaurant": f"candid evening photo of a {PERSONA} with soft curls, elegant black satin slip dress, gold earrings, sitting at a restaurant table with red wine glass and candle, chin resting on hand, warm smile, warm pendant lights and bar shelves behind, candlelight ambience, 50mm lens, high detail, sharp focus",
}

SEED = 7


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
            entry = h[pid]
            if entry.get("status", {}).get("completed"):
                for node in entry["outputs"].values():
                    for img in node.get("images", []):
                        return os.path.join(COMFY_OUT, img.get("subfolder", ""), img["filename"])
            if entry.get("status", {}).get("status_str") == "error":
                raise RuntimeError(f"workflow error: {json.dumps(entry['status'])[:500]}")
        time.sleep(2)
    raise TimeoutError(pid)


def main():
    base_wf = json.load(open("wf_zimage.json"))
    finish_wf = json.load(open("wf_finish_zimage.json"))
    os.makedirs("results/faceoff", exist_ok=True)

    for name, prompt in SCENES.items():
        # stage 1: t2i
        wf = json.loads(json.dumps(base_wf))
        wf["4"]["inputs"]["text"] = prompt
        wf["8"]["inputs"]["seed"] = SEED
        wf["10"]["inputs"]["filename_prefix"] = f"zbase_{name}"
        t = time.time()
        out1 = wait(submit(wf))
        print(f"[z] base {name}: {time.time()-t:.0f}s -> {out1}", flush=True)

        # stage 2: finish (FaceDetailer + 2K refine)
        src = f"faceoff_src_{name}.png"
        shutil.copy(out1, os.path.join(COMFY_IN, src))
        wf2 = json.loads(json.dumps(finish_wf))
        wf2["5"]["inputs"]["image"] = src
        wf2["23"]["inputs"]["seed"] = SEED + 1
        wf2["36"]["inputs"]["seed"] = SEED + 2
        wf2["38"]["inputs"]["filename_prefix"] = f"zfinish_{name}"
        t = time.time()
        out2 = wait(submit(wf2))
        print(f"[z] finish {name}: {time.time()-t:.0f}s -> {out2}", flush=True)
        shutil.copy(out2, f"results/faceoff/zimage_{name}.png")

    print("[z] Z-Image side done", flush=True)


if __name__ == "__main__":
    main()
