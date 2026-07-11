#!/usr/bin/env python3
"""Z-Image: генерит персону (фикс. лицо+seed) в 4 нарядах референс-рила, в ОДНОЙ
комнате -> стабильный фон между шотами. Схема как в zimage_faceoff.py (t2i +
FaceDetailer-финиш). Выход -> ComfyUI/input/reeloutfit_{1..4}.png для bg=ref флоу.
"""
import json, os, shutil, time, urllib.request

API = "http://127.0.0.1:10100"
COMFY_OUT, COMFY_IN = "/workspace/ComfyUI/output", "/workspace/ComfyUI/input"
SEED = 7
PERSONA = ("young woman with long honey-blonde wavy hair, hazel eyes, natural skin texture "
           "with pores and faint freckles, slim figure")
# один и тот же фон-комната для всех -> стабильность между шотами
ROOM = ("standing full body, full-length view, in a plain minimal bedroom with a beige wall, "
        "a white door and light wood floor, soft diffused indoor daylight, casual mirror-selfie "
        "framing, realistic amateur phone photo, sharp focus, high detail")
OUTFITS = {
    "1": "wearing an oversized black graphic band t-shirt and brown leopard-print cycling shorts with matching leopard-print mesh long sleeves",
    "2": "wearing a fitted white and grey ditsy small-floral bodycon mini dress with thin straps and sheer floral long sleeves",
    "3": "wearing a fitted black and white large daisy floral bodycon mini dress with thin straps and sheer sleeves",
    "4": "wearing a fitted white bodycon mini dress with red hibiscus flowers and sheer floral long sleeves",
}


def submit(wf):
    req = urllib.request.Request(API + "/prompt", json.dumps({"prompt": wf}).encode(),
                                 {"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req).read())["prompt_id"]


def wait(pid, timeout=600):
    t0 = time.time()
    while time.time() - t0 < timeout:
        h = json.loads(urllib.request.urlopen(f"{API}/history/{pid}").read())
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
    base = json.load(open("wf_zimage.json"))
    finish = json.load(open("wf_finish_zimage.json"))
    for k, outfit in OUTFITS.items():
        prompt = f"{ROOM.split('framing')[0]}a {PERSONA}, {outfit}, casual mirror-selfie framing, realistic amateur phone photo, sharp focus, high detail"
        wf = json.loads(json.dumps(base))
        wf["4"]["inputs"]["text"] = prompt
        wf["7"]["inputs"]["width"] = 576
        wf["7"]["inputs"]["height"] = 1024
        wf["8"]["inputs"]["seed"] = SEED
        wf["10"]["inputs"]["filename_prefix"] = f"reelbase_{k}"
        t = time.time(); out1 = wait(submit(wf))
        print(f"[base {k}] {time.time()-t:.0f}s -> {out1}", flush=True)
        src = f"reelbase_{k}.png"; shutil.copy(out1, os.path.join(COMFY_IN, src))
        wf2 = json.loads(json.dumps(finish))
        wf2["5"]["inputs"]["image"] = src
        wf2["23"]["inputs"]["seed"] = SEED + 1
        wf2["36"]["inputs"]["seed"] = SEED + 2
        wf2["38"]["inputs"]["filename_prefix"] = f"reelfin_{k}"
        t = time.time(); out2 = wait(submit(wf2))
        dst = os.path.join(COMFY_IN, f"reeloutfit_{k}.png"); shutil.copy(out2, dst)
        print(f"[finish {k}] {time.time()-t:.0f}s -> {dst}", flush=True)
    print("REEL_OUTFITS_DONE", flush=True)


if __name__ == "__main__":
    main()
