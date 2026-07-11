"""Batch T2I for HiDream-O1-Image-Dev-2604 — thin wrapper over the OFFICIAL inference.py.

Mirrors the repo's t2i example exactly (only `--model_type dev` + prompt/output/size/seed);
dev params (28 steps / guidance 0.0 / shift 1.0 / flash) are set INSIDE inference.py — we never
hand-pass CFG/shift, so there's nothing to get wrong.

Usage:
  python run_hidream_o1.py --model_path models/hidream-o1-dev-2604 \
      --prompts prompts.txt --outdir results/ --height 1024 --width 1024
  python run_hidream_o1.py --model_path models/hidream-o1-dev-2604 \
      --prompt "an ultra-photorealistic close-up portrait ..." --outdir results/

⚠️ Reloads the 8B model per prompt (subprocess) — fine for small batches; for large runs, an
in-process loop is a pod-side optimization once models/pipeline.py's load path is confirmed.
"""
from __future__ import annotations
import argparse, os, re, sys, subprocess, time

def slug(s, i):
    s = re.sub(r"[^a-zA-Z0-9]+", "_", s).strip("_")[:48] or "img"
    return f"{i:03d}_{s}"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True, help="downloaded HiDream-O1-Image-Dev-2604 dir")
    ap.add_argument("--repo", default="HiDream-O1-Image", help="cloned official repo dir")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--prompts", help="text file, one prompt per line")
    g.add_argument("--prompt", help="single prompt")
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--height", type=int, default=2048)   # snapped to ~2048-class buckets anyway; sets aspect only
    ap.add_argument("--width", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=32)
    ap.add_argument("--vary_seed", action="store_true", help="seed = base_seed + index")
    ap.add_argument("--model_type", default="dev", choices=["dev", "full"])
    ap.add_argument("--python", default=sys.executable)
    a = ap.parse_args()

    inf = os.path.join(a.repo, "inference.py")
    if not os.path.exists(inf):
        sys.exit(f"inference.py not found at {inf} — run setup_hidream_o1.sh first")
    os.makedirs(a.outdir, exist_ok=True)

    if a.prompts:
        prompts = [l.strip() for l in open(a.prompts, encoding="utf-8") if l.strip() and not l.startswith("#")]
    else:
        prompts = [a.prompt]

    print(f"{len(prompts)} prompt(s) · {a.model_type} · {a.width}x{a.height}")
    ok = 0; t0 = time.time()
    for i, p in enumerate(prompts):
        out = os.path.join(a.outdir, slug(p, i) + ".png")
        seed = a.seed + (i if a.vary_seed else 0)
        env = {**os.environ, "FA_VERSION": os.environ.get("FA_VERSION", "auto")}  # dev branch hard-imports flash_attn otherwise
        cmd = [a.python, inf,
               "--model_path", a.model_path,
               "--prompt", p,
               "--output_image", out,
               "--model_type", a.model_type,     # dev → 28/0.0/1.0/flash internally
               "--height", str(a.height), "--width", str(a.width),
               "--seed", str(seed)]
        t = time.time()
        r = subprocess.run(cmd, env=env)
        dt = time.time() - t
        good = r.returncode == 0 and os.path.exists(out)
        ok += good
        print(f"[{i+1}/{len(prompts)}] {'OK' if good else 'FAIL'} {dt:.0f}s -> {out}")
    print(f"\ndone: {ok}/{len(prompts)} in {(time.time()-t0)/60:.1f}min -> {a.outdir}")

if __name__ == "__main__":
    main()
