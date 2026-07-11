#!/usr/bin/env python3
"""Wan2.2-Animate — ПОЛНЫЙ рил: чанковая генерация + RIFE-интерполяция до 30 fps.

Покрывает весь driving-рил цепочкой WanAnimateToVideo-чанков (continue_motion +
video_frame_offset — нода сама сшивает стык по последним N кадрам и сдвигает
offset), кадры всех чанков накапливаются, затем RIFE VFI ×2 (16→32) и финальный
ретайм ffmpeg до ровно 30 fps с оригинальной аудиодорожкой рила.

  python run_wanimate_full.py --bg ref
  python run_wanimate_full.py --bg video --out-fps 30
"""
import argparse, json, subprocess, sys, time, urllib.request

NEG = ("色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，"
       "低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，"
       "毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走")
INP = "/workspace/ComfyUI/input"
OUT = "/workspace/ComfyUI/output"


def probe(path, *entries):
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v",
                        "-count_frames", "-show_entries",
                        "stream=" + ",".join(entries), "-of", "csv=p=0", path],
                       capture_output=True, text=True)
    return r.stdout.strip().split(",")


def build(a, n_chunks, drv_frames):
    g, cmf = {}, a.continue_frames
    n = lambda i, ct, **inp: g.__setitem__(str(i), {"class_type": ct, "inputs": inp})

    n(1, "UNETLoader", unet_name="Wan2_2-Animate-14B_fp8_e4m3fn_scaled_KJ.safetensors", weight_dtype="default")
    n(2, "LoraLoaderModelOnly", model=["1", 0],
      lora_name="lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors", strength_model=1.0)
    mo = ["2", 0]
    if a.bg in ("video", "image"):
        n(3, "LoraLoaderModelOnly", model=mo, lora_name="WanAnimate_relight_lora_fp16.safetensors", strength_model=1.0)
        mo = ["3", 0]
    n(4, "ModelSamplingSD3", model=mo, shift=8.0)

    n(5, "CLIPLoader", clip_name="umt5_xxl_fp8_e4m3fn_scaled.safetensors", type="wan", device="default")
    n(6, "CLIPTextEncode", clip=["5", 0], text=a.prompt)
    n(7, "CLIPTextEncode", clip=["5", 0], text=NEG)
    n(8, "CLIPVisionLoader", clip_name="clip_vision_h.safetensors")
    n(9, "LoadImage", image=a.image)
    n(10, "CLIPVisionEncode", clip_vision=["8", 0], image=["9", 0], crop="none")
    n(11, "VAELoader", vae_name="wan_2.1_vae.safetensors")

    # driving -> pose/face над ВСЕМ клипом (чанки сикают по offset внутри ноды)
    n(12, "LoadVideo", file=a.video)
    n(13, "GetVideoComponents", video=["12", 0])
    n(14, "ImageScale", image=["13", 0], upscale_method="lanczos", width=a.width, height=a.height, crop="center")
    dw = dict(resolution=a.dw_res, bbox_detector="yolox_l.torchscript.pt",
              pose_estimator="dw-ll_ucoco_384_bs5.torchscript.pt", scale_stick_for_xinsr_cn="disable")
    n(15, "DWPreprocessor", image=["14", 0], detect_hand="enable", detect_body="enable", detect_face="disable", **dw)
    n(16, "DWPreprocessor", image=["14", 0], detect_hand="disable", detect_body="disable", detect_face="enable", **dw)

    common = dict(face_video=["16", 0], pose_video=["15", 0])
    if a.bg in ("video", "image"):
        n(17, "CheckpointLoaderSimple", ckpt_name="sam3.1_multiplex_fp16.safetensors")
        n(18, "CLIPTextEncode", clip=["17", 1], text="human")
        n(19, "SAM3_VideoTrack", images=["14", 0], model=["17", 0], conditioning=["18", 0],
          detection_threshold=0.5, max_objects=1, detect_interval=1)
        n(20, "SAM3_TrackToMask", track_data=["19", 0], object_indices="")
        n(21, "GrowMask", mask=["20", 0], expand=a.mask_grow, tapered_corners=True)
        n(50, "MaskToImage", mask=["21", 0])
        n(51, "ImageScale", image=["50", 0], upscale_method="area", width=a.width // 32, height=a.height // 32, crop="disabled")
        n(52, "ImageScale", image=["51", 0], upscale_method="nearest-exact", width=a.width, height=a.height, crop="disabled")
        n(53, "ImageToMask", image=["52", 0], channel="red")
        n(54, "ThresholdMask", mask=["53", 0], value=0.05)
        n(55, "EmptyImage", width=a.width, height=a.height, batch_size=1, color=0)
        common["character_mask"] = ["54", 0]
        if a.bg == "video":
            n(56, "ImageCompositeMasked", destination=["14", 0], source=["55", 0], mask=["54", 0], x=0, y=0, resize_source=True)
            common["background_video"] = ["56", 0]
        else:
            n(57, "LoadImage", image=a.bg_image)
            n(58, "ImageScale", image=["57", 0], upscale_method="lanczos", width=a.width, height=a.height, crop="center")
            n(59, "RepeatImageBatch", image=["58", 0], amount=a.width * 0 + 1000)  # с запасом, нода клампит по length
            n(60, "ImageCompositeMasked", destination=["59", 0], source=["55", 0], mask=["54", 0], x=0, y=0, resize_source=True)
            common["background_video"] = ["60", 0]

    # цепочка чанков
    acc, prev = None, None
    base = 100
    for k in range(n_chunks):
        wid, ks, tl, vd, fb, ib = (base + k * 10 + j for j in range(6))
        extra = dict(video_frame_offset=0) if k == 0 else \
            dict(video_frame_offset=[str(prev), 5], continue_motion=[str(acc), 0])
        n(wid, "WanAnimateToVideo", positive=["6", 0], negative=["7", 0], vae=["11", 0],
          width=a.width, height=a.height, length=a.length, batch_size=1,
          clip_vision_output=["10", 0], reference_image=["9", 0],
          continue_motion_max_frames=cmf, **common, **extra)
        n(ks, "KSampler", model=["4", 0], positive=[str(wid), 0], negative=[str(wid), 1],
          latent_image=[str(wid), 2], seed=a.seed, steps=a.steps, cfg=1.0,
          sampler_name="euler", scheduler="simple", denoise=1.0)
        n(tl, "TrimVideoLatent", samples=[str(ks), 0], trim_amount=[str(wid), 3])
        n(vd, "VAEDecode", samples=[str(tl), 0], vae=["11", 0])
        n(fb, "ImageFromBatch", image=[str(vd), 0], batch_index=[str(wid), 4], length=4096)
        if k == 0:
            acc = fb
        else:
            n(ib, "ImageBatch", image1=[str(acc), 0], image2=[str(fb), 0])
            acc = ib
        prev = wid

    # Обрезаем накопленные кадры РОВНО по длине рила: последний чанк переполняет
    # длину, и нода достраивает хвост замороженным последним кадром -> иначе выход
    # длиннее рила и «подвисает» в конце. Тут статичный хвост срезается полностью.
    n(89, "ImageFromBatch", image=[str(acc), 0], batch_index=0, length=drv_frames)
    acc = 89

    # RIFE ×mult (при mult>1) -> CreateVideo -> Save; аудио муксим в пост-обработке.
    # mult==1 => интерполяция ОТКЛЮЧЕНА, все кадры нативные (нет «плавучего» слоу-мо).
    frames_ref = [str(acc), 0]
    if a.rife_mult > 1:
        n(90, "RIFE VFI", frames=[str(acc), 0], ckpt_name="rife49.pth",
          clear_cache_after_n_frames=10, multiplier=a.rife_mult, fast_mode=True,
          ensemble=True, scale_factor=1.0, dtype="float32", torch_compile=False, batch_size=1)
        frames_ref = ["90", 0]
    n(91, "CreateVideo", images=frames_ref, fps=a.native_fps * a.rife_mult)
    n(92, "SaveVideo", video=["91", 0], filename_prefix=f"video/wanim_full_{a.bg}", format="auto", codec="auto")
    return g


def run(graph, port):
    req = urllib.request.Request(f"http://127.0.0.1:{port}/prompt",
                                 json.dumps({"prompt": graph}).encode(), {"Content-Type": "application/json"})
    try:
        r = json.loads(urllib.request.urlopen(req).read())
    except urllib.error.HTTPError as e:
        print("HTTP", e.code, e.read().decode()[:3000]); sys.exit(1)
    if r.get("node_errors"):
        print(json.dumps(r, indent=1, ensure_ascii=False)); sys.exit(1)
    pid = r["prompt_id"]; print("prompt_id:", pid, flush=True); t0 = time.time()
    while True:
        time.sleep(5)
        h = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{port}/history/{pid}").read())
        if pid not in h:
            print(f"  ... running {time.time()-t0:.0f}s", flush=True); continue
        st = h[pid]["status"]
        if st.get("completed"):
            for o in h[pid]["outputs"].values():
                for v in o.get("images", []) + o.get("video", []):
                    return f"{OUT}/{v.get('subfolder','')}/{v['filename']}", time.time() - t0
        if st.get("status_str") == "error":
            for m in st.get("messages", []):
                if m[0] == "execution_error":
                    print("ERROR node", m[1].get("node_type"), ":", m[1].get("exception_message"))
            sys.exit(1)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--image", default="persona_street.png")
    p.add_argument("--video", default="drive_16fps.mp4", help="driving для генерации (16 fps)")
    p.add_argument("--audio-src", default="drive_30fps.mp4", help="откуда взять оригинальное аудио")
    p.add_argument("--bg", choices=["ref", "video", "image"], default="ref")
    p.add_argument("--bg-image", default="bg_loft.png")
    p.add_argument("--prompt", default="A young woman dances energetically, smooth natural motion, "
                                       "consistent appearance, high quality video")
    p.add_argument("--width", type=int, default=576)
    p.add_argument("--height", type=int, default=1024)
    p.add_argument("--length", type=int, default=77, help="кадров в чанке (native fps)")
    p.add_argument("--continue-frames", type=int, default=5, help="перекрытие чанков")
    p.add_argument("--chunks", type=int, default=0, help="0 = авто по длине рила")
    p.add_argument("--steps", type=int, default=6)
    p.add_argument("--native-fps", type=int, default=16)
    p.add_argument("--rife-mult", type=int, default=2)
    p.add_argument("--out-fps", type=int, default=30)
    p.add_argument("--seed", type=int, default=1106558644923357)
    p.add_argument("--dw-res", type=int, default=768)
    p.add_argument("--mask-grow", type=int, default=12)
    p.add_argument("--port", type=int, default=10100)
    p.add_argument("--dump", action="store_true")
    a = p.parse_args()

    nb, dur = probe(f"{INP}/{a.video}", "nb_read_frames"), None
    drv_frames = int(nb[0]) if nb and nb[0].isdigit() else a.length
    dur = float(probe(f"{INP}/{a.audio_src}", "duration")[0] or 0) or drv_frames / a.native_fps
    net = a.length - a.continue_frames
    n_chunks = a.chunks or max(1, 1 + -(-(drv_frames - a.length) // net)) if drv_frames > a.length else 1
    print(f"driving={drv_frames}f ~{dur:.2f}s | chunks={n_chunks} x {a.length}f (overlap {a.continue_frames}) "
          f"| RIFE x{a.rife_mult} -> {a.native_fps*a.rife_mult}fps -> out {a.out_fps}fps", flush=True)

    graph = build(a, n_chunks, drv_frames)
    if a.dump:
        print(json.dumps(graph, indent=1, ensure_ascii=False)); sys.exit(0)

    raw, gen_t = run(graph, a.port)
    print(f"RAW: {raw} (gen {gen_t:.0f}s)", flush=True)

    final = raw.replace(".mp4", f"_{a.out_fps}fps_audio.mp4")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", raw, "-i", f"{INP}/{a.audio_src}",
                    "-map", "0:v:0", "-map", "1:a:0?", "-vf", f"fps={a.out_fps}", "-t", f"{dur:.3f}",
                    "-c:v", "libx264", "-crf", "17", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", final],
                   check=True)
    fr = probe(final, "r_frame_rate", "nb_read_frames")
    print(f"OUTPUT: {final}\nfps/frames: {fr}\nDONE", flush=True)
