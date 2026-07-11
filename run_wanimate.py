#!/usr/bin/env python3
"""Wan2.2-Animate: i2v + motion transfer (headless, ComfyUI native nodes).

Оживляет персону с картинки движением (тело+лицо) из driving-видео.
Режимы фона (--bg):
  ref   — Move mode: фон берётся с референс-картинки персоны (как в i2v)
  video — Replacement mode: фон из driving-видео, персона вставляется на место
          танцора (SAM3-маска + зачернение + relight LoRA)
  image — фон из отдельной картинки (--bg-image), персона рисуется в него
          по маске танцора из driving-видео (+ relight LoRA)

Пример:
  python run_wanimate.py --bg ref
  python run_wanimate.py --bg video --length 77
  python run_wanimate.py --bg image --bg-image bg_loft.png
"""
import argparse, json, time, urllib.request, sys

NEG = ("色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，"
       "低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，"
       "毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走")


def build(a):
    g = {}
    n = lambda i, ct, **inp: g.__setitem__(str(i), {"class_type": ct, "inputs": inp})

    # model chain
    n(1, "UNETLoader", unet_name="Wan2_2-Animate-14B_fp8_e4m3fn_scaled_KJ.safetensors", weight_dtype="default")
    n(2, "LoraLoaderModelOnly", model=["1", 0],
      lora_name="lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors", strength_model=1.0)
    model_out = ["2", 0]
    if a.bg in ("video", "image"):  # relight — для вставки персоны в чужую сцену
        n(3, "LoraLoaderModelOnly", model=model_out,
          lora_name="WanAnimate_relight_lora_fp16.safetensors", strength_model=1.0)
        model_out = ["3", 0]
    n(4, "ModelSamplingSD3", model=model_out, shift=8.0)

    # text / vision encoders
    n(5, "CLIPLoader", clip_name="umt5_xxl_fp8_e4m3fn_scaled.safetensors", type="wan", device="default")
    n(6, "CLIPTextEncode", clip=["5", 0], text=a.prompt)
    n(7, "CLIPTextEncode", clip=["5", 0], text=NEG)
    n(8, "CLIPVisionLoader", clip_name="clip_vision_h.safetensors")
    n(9, "LoadImage", image=a.image)
    n(10, "CLIPVisionEncode", clip_vision=["8", 0], image=["9", 0], crop="none")
    n(11, "VAELoader", vae_name="wan_2.1_vae.safetensors")

    # driving video -> pose / face control
    n(12, "LoadVideo", file=a.video)
    n(13, "GetVideoComponents", video=["12", 0])
    n(14, "ImageScale", image=["13", 0], upscale_method="lanczos",
      width=a.width, height=a.height, crop="center")
    # оба детектора в TorchScript => GPU через torch (onnx тут падает на CPU/OpenCV)
    dw = dict(resolution=a.dw_res, bbox_detector="yolox_l.torchscript.pt",
              pose_estimator="dw-ll_ucoco_384_bs5.torchscript.pt", scale_stick_for_xinsr_cn="disable")
    n(15, "DWPreprocessor", image=["14", 0], detect_hand="enable", detect_body="enable",
      detect_face="disable", **dw)
    n(16, "DWPreprocessor", image=["14", 0], detect_hand="disable", detect_body="disable",
      detect_face="enable", **dw)

    wan_extra = {}
    if a.bg in ("video", "image"):
        # SAM3: маска танцора в driving-видео (headless, текстовый промпт)
        n(17, "CheckpointLoaderSimple", ckpt_name="sam3.1_multiplex_fp16.safetensors")
        n(18, "CLIPTextEncode", clip=["17", 1], text="human")
        n(19, "SAM3_VideoTrack", images=["14", 0], model=["17", 0], conditioning=["18", 0],
          detection_threshold=0.5, max_objects=1, detect_interval=1)
        n(20, "SAM3_TrackToMask", track_data=["19", 0], object_indices="")
        # Grow силуэт персоны, чтобы тонкие пряди волос (SAM3 их обрезает) попали
        # ВНУТРЬ зоны генерации — иначе волосы оказываются в «замороженном» фоне и
        # дают жёсткий ореол/срез по контуру.
        n(21, "GrowMask", mask=["20", 0], expand=a.mask_grow, tapered_corners=True)
        if a.mask_mode == "feather":
            # Изотропная растушёвка края: MaskToImage -> Gaussian blur -> ImageToMask.
            # Мягкий 0..1 край => нода частично блендит волосы в фон, а «дыра» в фоне
            # имеет плавный край -> сгенерированные пряди перекрывают градиент без ореола.
            n(50, "MaskToImage", mask=["21", 0])
            n(51, "ImageBlur", image=["50", 0], blur_radius=a.mask_blur, sigma=a.mask_sigma)
            n(53, "ImageToMask", image=["51", 0], channel="red")
            char_mask = ["53", 0]
        else:  # blockify — прежний путь (грубые 32px-блоки), для сравнения
            n(50, "MaskToImage", mask=["21", 0])
            n(51, "ImageScale", image=["50", 0], upscale_method="area",
              width=a.width // 32, height=a.height // 32, crop="disabled")
            n(52, "ImageScale", image=["51", 0], upscale_method="nearest-exact",
              width=a.width, height=a.height, crop="disabled")
            n(53, "ImageToMask", image=["52", 0], channel="red")
            n(54, "ThresholdMask", mask=["53", 0], value=0.05)
            char_mask = ["54", 0]
        wan_extra["character_mask"] = char_mask
        n(22, "EmptyImage", width=a.width, height=a.height, batch_size=1, color=0)
        if a.bg == "video":
            # фон = driving-видео; зачернение зоны танцора не нужно, если character_mask
            # уже говорит ноде, где персона. hole=on оставлен для совместимости.
            if a.bg_hole == "on":
                n(23, "ImageCompositeMasked", destination=["14", 0], source=["22", 0],
                  mask=char_mask, x=0, y=0, resize_source=True)
                wan_extra["background_video"] = ["23", 0]
            else:
                wan_extra["background_video"] = ["14", 0]
        elif a.bg_hole == "off":
            # ПОЛНЫЙ фон без «дыры»: character_mask сам указывает ноде зону персоны,
            # модель генерит её поверх фона. Нет чёрной дыры -> нет ореола у волос.
            n(24, "LoadImage", image=a.bg_image)
            n(25, "ImageScale", image=["24", 0], upscale_method="lanczos",
              width=a.width, height=a.height, crop="center")
            n(26, "RepeatImageBatch", image=["25", 0], amount=a.length)
            wan_extra["background_video"] = ["26", 0]
        else:
            # (hole=on) статичная картинка с зачернённой зоной персонажа — старый путь
            n(24, "LoadImage", image=a.bg_image)
            n(25, "ImageScale", image=["24", 0], upscale_method="lanczos",
              width=a.width, height=a.height, crop="center")
            n(26, "RepeatImageBatch", image=["25", 0], amount=a.length)
            n(27, "ImageCompositeMasked", destination=["26", 0], source=["22", 0],
              mask=char_mask, x=0, y=0, resize_source=True)
            wan_extra["background_video"] = ["27", 0]

    n(30, "WanAnimateToVideo", positive=["6", 0], negative=["7", 0], vae=["11", 0],
      width=a.width, height=a.height, length=a.length, batch_size=1,
      clip_vision_output=["10", 0], reference_image=["9", 0],
      face_video=["16", 0], pose_video=["15", 0],
      continue_motion_max_frames=5, video_frame_offset=0, **wan_extra)
    n(31, "KSampler", model=["4", 0], positive=["30", 0], negative=["30", 1],
      latent_image=["30", 2], seed=a.seed, steps=a.steps, cfg=1.0,
      sampler_name="euler", scheduler="simple", denoise=1.0)
    n(32, "TrimVideoLatent", samples=["31", 0], trim_amount=["30", 3])
    n(33, "VAEDecode", samples=["32", 0], vae=["11", 0])
    n(34, "ImageFromBatch", image=["33", 0], batch_index=["30", 4], length=4096)
    n(35, "CreateVideo", images=["34", 0], audio=["13", 1], fps=a.fps)
    n(36, "SaveVideo", video=["35", 0], filename_prefix=f"video/wanim_{a.bg}",
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
            paths = []
            for o in h[pid]["outputs"].values():
                for v in o.get("images", []) + o.get("video", []):
                    p = f"/workspace/ComfyUI/output/{v.get('subfolder','')}/{v['filename']}"
                    paths.append(p); print("OUTPUT:", p)
            print(f"DONE in {time.time()-t0:.0f}s")
            return paths[0] if paths else None
        if st.get("status_str") == "error":
            for m in st.get("messages", []):
                if m[0] == "execution_error":
                    print("ERROR node", m[1].get("node_type"), ":", m[1].get("exception_message"))
            sys.exit(1)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--image", default="persona_street.png", help="файл в ComfyUI/input")
    p.add_argument("--video", default="drive_16fps.mp4", help="driving-видео (16fps) в ComfyUI/input")
    p.add_argument("--bg", choices=["ref", "video", "image"], default="ref")
    p.add_argument("--bg-image", default="bg_loft.png")
    p.add_argument("--prompt", default="A young woman dances energetically, smooth natural motion, "
                                       "consistent appearance, high quality video")
    p.add_argument("--width", type=int, default=576)
    p.add_argument("--height", type=int, default=1024)
    p.add_argument("--length", type=int, default=77)
    p.add_argument("--steps", type=int, default=6)
    p.add_argument("--fps", type=int, default=16)
    p.add_argument("--seed", type=int, default=1106558644923357)
    p.add_argument("--dw-res", type=int, default=768)
    p.add_argument("--mask-grow", type=int, default=12, help="расширение маски персоны (px); больше = волосы внутри зоны")
    p.add_argument("--mask-mode", choices=["feather", "blockify"], default="feather",
                   help="feather = мягкий край (лучше для волос), blockify = грубые 32px-блоки")
    p.add_argument("--mask-blur", type=int, default=9, help="радиус растушёвки края маски (feather)")
    p.add_argument("--mask-sigma", type=float, default=4.0, help="сигма гаусса растушёвки (feather)")
    p.add_argument("--bg-hole", choices=["on", "off"], default="off",
                   help="off = полный фон + character_mask (без ореола); on = зачернять зону персоны")
    p.add_argument("--port", type=int, default=10100)
    p.add_argument("--out", default="", help="скопировать результат в этот путь (абсолютный или относительный)")
    p.add_argument("--dump", action="store_true", help="только напечатать граф")
    a = p.parse_args()
    graph = build(a)
    if a.dump:
        print(json.dumps(graph, indent=1, ensure_ascii=False))
    else:
        result = run(graph, a.port)
        if a.out and result:
            import os, shutil
            dst = os.path.abspath(os.path.expanduser(a.out))
            os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
            shutil.copy(result, dst)
            print("SAVED:", dst)
