#!/usr/bin/env python3
"""Wan2.2-Animate — смена ОБРАЗА персоны синхронно с рилом (outfit-change reel).

Рил с несколькими нарядами -> TransNetV2 находит границы шотов -> каждый шот
генерится СО СВОЕЙ референс-картинкой той же девушки (разный наряд), единый
seed для консистентности лица -> склейка (рез совпадает с переходом рила) ->
нативные 30 fps + оригинальное аудио. Надстройка над run_wanimate_full.build().

  python run_wanimate_outfits.py                          # авто-шоты + 4 наряда
  python run_wanimate_outfits.py --outfits outfit_street.png,outfit_restaurant.png
  python run_wanimate_outfits.py --cuts 133,170,207       # ручные границы
"""
import argparse, subprocess, sys, types
import run_wanimate_full as R

INP, OUT = "/workspace/ComfyUI/input", "/workspace/ComfyUI/output"


def detect_shots(video_path, fps, manual_cuts=None):
    total = int(R.probe(video_path, "nb_read_frames")[0])
    if manual_cuts:
        cuts = [int(c) for c in manual_cuts.split(",") if c.strip()]
    else:
        import torch, numpy as np
        from transnetv2_pytorch import TransNetV2
        m = TransNetV2(); m.eval()
        with torch.no_grad():
            out = m.predict_video(video_path, quiet=True)
        preds = (out[1] if isinstance(out, tuple) else out)
        preds = (preds.detach().cpu().numpy() if hasattr(preds, "detach") else np.asarray(preds)).reshape(-1)
        b = preds > 0.3
        cuts = [i for i in range(1, len(b)) if b[i] and not b[i - 1]]
    bounds = [0] + cuts + [total]
    return [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)], total


def cut_clip(src, start, end, dst, fps):
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", src,
                    "-vf", f"select=between(n\\,{start}\\,{end-1}),setpts=N/{fps}/TB",
                    "-frames:v", str(end - start), "-an",
                    "-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p", dst], check=True)


def shot_args(base, video, image, length, seed, fps):
    a = types.SimpleNamespace(**vars(base))
    a.video, a.image, a.length, a.seed = video, image, length, seed
    a.native_fps, a.rife_mult, a.out_fps = fps, 1, fps
    return a


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--driving", default="drive_30fps.mp4")
    p.add_argument("--outfits", default="outfit_street.png,outfit_restaurant.png,outfit_selfie.png,outfit_gym.png",
                   help="референсы-наряды по порядку шотов (round-robin если шотов больше)")
    p.add_argument("--cuts", default="", help="ручные границы шотов в кадрах (иначе TransNetV2)")
    p.add_argument("--prompt", default="A young woman dances energetically, smooth natural motion, "
                                       "consistent face and appearance, high quality video")
    p.add_argument("--width", type=int, default=576)
    p.add_argument("--height", type=int, default=1024)
    p.add_argument("--max-len", type=int, default=77, help="макс кадров в чанке")
    p.add_argument("--continue-frames", type=int, default=5)
    p.add_argument("--steps", type=int, default=6)
    p.add_argument("--seed", type=int, default=1106558644923357)
    p.add_argument("--dw-res", type=int, default=768)
    p.add_argument("--mask-grow", type=int, default=12)
    p.add_argument("--fps", type=int, default=24, help="нативный fps генерации/выхода (24 = компромисс)")
    p.add_argument("--bg", choices=["ref", "image"], default="ref",
                   help="ref = фон из реф-картинки наряда; image = один стабильный фон --bg-image")
    p.add_argument("--bg-image", default="bg_loft.png")
    p.add_argument("--port", type=int, default=10100)
    a = p.parse_args()

    src = f"{INP}/{a.driving}"
    shots, total = detect_shots(src, a.fps, a.cuts or None)
    outfits = [o.strip() for o in a.outfits.split(",") if o.strip()]
    dur = float(R.probe(src, "duration")[0] or total / a.fps)
    print(f"driving={total}f ~{dur:.2f}s | shots={len(shots)}: {shots}")
    print(f"outfits: {[outfits[i % len(outfits)] for i in range(len(shots))]}\n", flush=True)

    base = types.SimpleNamespace(bg=a.bg, bg_image=a.bg_image, prompt=a.prompt, width=a.width, height=a.height,
                                 continue_frames=a.continue_frames, steps=a.steps, dw_res=a.dw_res,
                                 mask_grow=a.mask_grow, mask_mode="feather", mask_blur=9, mask_sigma=4.0,
                                 bg_hole="on", native_fps=a.fps, rife_mult=1, out_fps=a.fps)
    listfile = f"{OUT}/video/outfits_concat.txt"
    seg_paths = []
    for i, (s, e) in enumerate(shots):
        nf = e - s
        clip = f"seg{i}_drv.mp4"
        cut_clip(src, s, e, f"{INP}/{clip}", a.fps)
        # длина чанка: короткий шот = один чанк ровно по кадрам (округл. до 4k+1)
        L = a.max_len if nf > a.max_len else ((nf - 1 + 3) // 4) * 4 + 1
        n_chunks = 1 if nf <= a.max_len else 1 + -(-(nf - L) // (L - a.continue_frames))
        outfit = outfits[i % len(outfits)]
        aa = shot_args(base, clip, outfit, L, a.seed, a.fps)
        aa.bg, aa.bg_image = base.bg, base.bg_image
        print(f"[shot {i}] frames {s}-{e} ({nf}f) outfit={outfit} chunks={n_chunks} L={L}", flush=True)
        graph = R.build(aa, n_chunks, nf)
        # без RIFE-ветки нода 91 берёт кадры напрямую; native 30fps
        raw, t = R.run(graph, a.port)
        seg = f"{OUT}/video/outfit_seg{i}.mp4"
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", raw, "-vf", f"fps={a.fps}",
                        "-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p", seg], check=True)
        seg_paths.append(seg)
        print(f"  -> {seg} ({t:.0f}s)", flush=True)

    with open(listfile, "w") as f:
        for sp in seg_paths:
            f.write(f"file '{sp}'\n")
    final = f"{OUT}/video/wanim_outfits_FINAL.mp4"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", listfile,
                    "-i", src, "-map", "0:v:0", "-map", "1:a:0?", "-t", f"{dur:.3f}",
                    "-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", final],
                   check=True)
    fr = R.probe(final, "r_frame_rate", "nb_read_frames")
    print(f"\nOUTPUT: {final}\nfps/frames: {fr}\nDONE", flush=True)
