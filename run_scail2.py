#!/usr/bin/env python3
"""SCAIL-2 (Wan2.1-based): motion transfer end-to-end, без промежуточного скелета.

Движение подаётся сырыми RGB-кадрами driving-видео + цветные SAM3-маски
идентичностей (SCAIL2ColoredMask). Персона — с референс-картинки.
Режимы фона (--bg):
  ref   — Animation mode: сцена берётся с референс-картинки персоны
  video — Replacement mode: персона заменяет танцора в сцене driving-видео
  image — фон из отдельной картинки: персона предварительно композитится на
          --bg-image (SAM3-маска персоны + ImageCompositeMasked), затем
          Animation mode с этим композитом как референсом

Пример:
  python run_scail2.py --bg video
  python run_scail2.py --bg ref --prompt "A young woman dances on a city street..."
  python run_scail2.py --bg image --bg-image bg_loft.png
"""
import argparse, json, time, urllib.request, sys

PROMPT_DEFAULT = (
    "A young woman with long wavy blonde hair, wearing an oversized camel blazer over a white "
    "t-shirt, blue high-waisted jeans and white sneakers, with a black shoulder bag, performs an "
    "energetic dance with smooth natural movements. Her appearance stays consistent. "
    "High quality video, natural lighting.")


def build(a):
    g = {}
    n = lambda i, ct, **inp: g.__setitem__(str(i), {"class_type": ct, "inputs": inp})
    rm = a.bg == "video"  # replacement_mode

    # model chain: SCAIL-2 + DPO LoRA + lightx2v distill (fast 6-step)
    n(1, "UNETLoader", unet_name="wan2.1_14B_SCAIL_2_fp8_scaled.safetensors", weight_dtype="default")
    n(2, "LoraLoaderModelOnly", model=["1", 0],
      lora_name="wan2.1_SCAIL_2_DPO_lora_bf16.safetensors", strength_model=1.0)
    n(3, "LoraLoaderModelOnly", model=["2", 0],
      lora_name="lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors", strength_model=0.8)
    n(4, "ModelSamplingSD3", model=["3", 0], shift=5.0)

    n(5, "CLIPLoader", clip_name="umt5_xxl_fp8_e4m3fn_scaled.safetensors", type="wan", device="default")
    n(6, "CLIPTextEncode", clip=["5", 0], text=a.prompt)
    n(7, "CLIPTextEncode", clip=["5", 0], text="")
    n(8, "CLIPVisionLoader", clip_name="clip_vision_h.safetensors")
    n(11, "VAELoader", vae_name="wan_2.1_vae.safetensors")

    # reference image (опц. композит на новый фон)
    n(9, "LoadImage", image=a.image)
    ref_img = ["9", 0]

    # SAM3
    n(16, "CheckpointLoaderSimple", ckpt_name="sam3.1_multiplex_fp16.safetensors")
    n(17, "CLIPTextEncode", clip=["16", 1], text="human")

    if a.bg == "image":
        n(40, "LoadImage", image=a.bg_image)
        n(41, "GetImageSize", image=ref_img)
        n(42, "ImageScale", image=["40", 0], upscale_method="lanczos",
          width=["41", 0], height=["41", 1], crop="center")
        n(43, "SAM3_VideoTrack", images=ref_img, model=["16", 0], conditioning=["17", 0],
          detection_threshold=0.5, max_objects=1, detect_interval=1)
        n(44, "SAM3_TrackToMask", track_data=["43", 0], object_indices="")
        n(45, "GrowMask", mask=["44", 0], expand=2, tapered_corners=True)
        n(46, "ImageCompositeMasked", destination=["42", 0], source=ref_img,
          mask=["45", 0], x=0, y=0, resize_source=False)
        ref_img = ["46", 0]

    # driving video: первый чанк кадров как есть (SCAIL-2 ест RGB, не скелет)
    n(12, "LoadVideo", file=a.video)
    n(13, "GetVideoComponents", video=["12", 0])
    n(14, "ImageFromBatch", image=["13", 0], batch_index=a.frame_offset, length=a.length)
    n(15, "ImageScale", image=["14", 0], upscale_method="lanczos",
      width=a.width, height=a.height, crop="center")

    # colored identity masks (driving + reference)
    n(18, "SAM3_VideoTrack", images=["15", 0], model=["16", 0], conditioning=["17", 0],
      detection_threshold=0.5, max_objects=1, detect_interval=1)
    n(19, "SAM3_VideoTrack", images=ref_img, model=["16", 0], conditioning=["17", 0],
      detection_threshold=0.5, max_objects=1, detect_interval=1)
    n(20, "SCAIL2ColoredMask", driving_track_data=["18", 0], ref_track_data=["19", 0],
      object_indices="", sort_by="left_to_right", replacement_mode=rm)

    n(10, "CLIPVisionEncode", clip_vision=["8", 0], image=ref_img, crop="none")

    n(21, "WanSCAILToVideo", positive=["6", 0], negative=["7", 0], vae=["11", 0],
      width=a.width, height=a.height, length=a.length, batch_size=1,
      pose_video=["15", 0], pose_video_mask=["20", 0], replacement_mode=rm,
      pose_strength=a.pose_strength, pose_start=0.0, pose_end=1.0,
      reference_image=ref_img, reference_image_mask=["20", 1],
      clip_vision_output=["10", 0], video_frame_offset=0, previous_frame_count=5)

    n(22, "KSamplerSelect", sampler_name="euler")
    n(23, "BasicScheduler", model=["3", 0], scheduler="simple", steps=a.steps, denoise=1.0)
    n(24, "SamplerCustom", model=["4", 0], add_noise=True, noise_seed=a.seed, cfg=1.0,
      positive=["21", 0], negative=["21", 1], sampler=["22", 0], sigmas=["23", 0],
      latent_image=["21", 2])
    n(25, "VAEDecode", samples=["24", 1], vae=["11", 0])
    n(26, "CreateVideo", images=["25", 0], audio=["13", 1], fps=a.fps)
    n(27, "SaveVideo", video=["26", 0], filename_prefix=f"video/scail2_{a.bg}",
      format="auto", codec="auto")
    return g


def run(graph, port):
    req = urllib.request.Request(f"http://127.0.0.1:{port}/prompt",
                                 json.dumps({"prompt": graph}).encode(),
                                 {"Content-Type": "application/json"})
    try:
        r = json.loads(urllib.request.urlopen(req).read())
    except urllib.error.HTTPError as e:
        print("HTTP", e.code, e.read().decode()[:3000]); sys.exit(1)
    if r.get("node_errors"):
        print(json.dumps(r, indent=1, ensure_ascii=False)); sys.exit(1)
    pid = r["prompt_id"]
    print("prompt_id:", pid, flush=True)
    t0 = time.time()
    while True:
        time.sleep(5)
        h = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{port}/history/{pid}").read())
        if pid not in h:
            print(f"  ... running {time.time()-t0:.0f}s", flush=True); continue
        st = h[pid]["status"]
        if st.get("completed"):
            for o in h[pid]["outputs"].values():
                for v in o.get("images", []) + o.get("video", []):
                    print("OUTPUT:", f"/workspace/ComfyUI/output/{v.get('subfolder','')}/{v['filename']}")
            print(f"DONE in {time.time()-t0:.0f}s"); return
        if st.get("status_str") == "error":
            for m in st.get("messages", []):
                if m[0] == "execution_error":
                    print("ERROR node", m[1].get("node_type"), ":", m[1].get("exception_message"))
            sys.exit(1)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--image", default="persona_street.png")
    p.add_argument("--video", default="drive_30fps.mp4", help="driving на исходном fps")
    p.add_argument("--bg", choices=["ref", "video", "image"], default="video")
    p.add_argument("--bg-image", default="bg_loft.png")
    p.add_argument("--prompt", default=PROMPT_DEFAULT)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--height", type=int, default=896)
    p.add_argument("--length", type=int, default=81)
    p.add_argument("--frame-offset", type=int, default=0)
    p.add_argument("--steps", type=int, default=6)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--seed", type=int, default=1106558644923357)
    p.add_argument("--pose-strength", type=float, default=1.0)
    p.add_argument("--port", type=int, default=10100)
    p.add_argument("--dump", action="store_true")
    a = p.parse_args()
    graph = build(a)
    if a.dump:
        print(json.dumps(graph, indent=1, ensure_ascii=False))
    else:
        run(graph, a.port)
