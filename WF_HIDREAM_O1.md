# HiDream-O1-Image-Dev-2604 — headless T2I workflow

> Studied from the **official repo** `github.com/HiDream-ai/HiDream-O1-Image` (not guessed).
> 8B **pixel-native Unified Transformer** on a **Qwen3-VL backbone** — VAE-less, no separate text
> encoder, generates raw pixels up to 2048×2048. MIT. Tops open-source realism benchmarks
> (beats FLUX.2 Dev / Qwen-Image at 7×/3.4× fewer params). NSFW: **base not confirmed uncensored —
> decide later** (community uncensored variants exist, see bottom).

## ⚠️ The facts that break it if you get them wrong
- **Loads as `Qwen3VLForConditionalGeneration`** (`torch_dtype=bfloat16`, `device_map="cuda"`), **NOT** a
  diffusers pipeline. It's `inference.py` + `models/pipeline.py`, driven by CLI.
- **Dev params are set INTERNALLY by `--model_type dev`** → **28 steps · guidance_scale 0.0 (CFG OFF) ·
  shift 1.0 · scheduler `flash`**. The official t2i example passes ONLY `--model_type dev` — **do NOT
  hand-pass `--guidance_scale`/`--shift`** (the argparse defaults 5.0/3.0 are for the FULL model).
  (A blog claiming "dev CFG 5.0" is WRONG — verified against the repo code.)
- **No conversion step.** `--model_path` points straight at the downloaded HF repo dir.
- **Deps are pinned:** `torch>=2.10`, **`transformers==4.57.1`**, diffusers/accelerate/einops/… .
  PyTorch **2.9.x is known-broken** — use ≥2.10.
- **flash-attn:** highly recommended. If you can't install it, edit `models/pipeline.py` **line 341**
  `"use_flash_attn": True` → `False`, else inference fails. (`setup_hidream_o1.sh` auto-patches.)

## Params (from the code)
| model_type | steps | guidance_scale | shift | scheduler | dtype |
|---|---|---|---|---|---|
| **dev** (2604) | **28** | **0.0** | **1.0** | flash (t2i) / flow_match (edit) | bf16 |
| full | 50 | 5.0 | 3.0 | default | bf16 |

Other argparse defaults: `--seed 32`, `--height/--width 2048`, `--noise_scale_start/end 7.5`,
`--noise_clip_std 2.5`. Native res is 2048² but **pixel-native = heavy** — see VRAM.

## Models
| Repo (HF) | What |
|---|---|
| `HiDream-ai/HiDream-O1-Image-Dev-2604` | **the checkpoint we use** (distilled, latest) |
| `HiDream-ai/HiDream-O1-Image` | full (higher quality, 50-step, slower) |
| `HiDream-ai/Prompt-Refine` + `google/gemma-4-31B-it` | optional reasoning prompt refiner (heavy — skip) |

## Run
```bash
# 1. setup (clone official repo + venv deps + flash-attn/patch)
bash setup_hidream_o1.sh
# 2. download weights
bash dl_hidream_o1.sh
# 3a. single image (mirrors the official example exactly)
python HiDream-O1-Image/inference.py \
    --model_path models/hidream-o1-dev-2604 \
    --prompt 'A dog holds a sign that says "HiDream-O1-Image release."' \
    --output_image results/t2i_dev.png \
    --model_type dev --height 1024 --width 1024
# 3b. batch from a prompts file (one prompt per line)
python run_hidream_o1.py --model_path models/hidream-o1-dev-2604 \
    --prompts prompts.txt --outdir results/ --height 1024 --width 1024
```

## VRAM (from the actual repo files — pick the pod by THIS)
Dev-2604 is a **full checkpoint: 8 bf16 shards ≈ 35 GB weights** (bigger than the "8B" headline — likely
counts the full multi-tower + vision). So:
- **Weights alone need ~35 GB** → a 24 GB card CANNOT load it in bf16.
- **pixel-native at 2048² adds a huge activation seq** on top → budget well above the 35 GB.
- **Rent 80 GB+** (A100-80 / H100-80 / **RTX 6000 96GB**) for comfort at 1024→2048.
  **48 GB (A6000/L40S)** may load + run **1024** but is tight — start there and watch OOM.
- **<48 GB:** don't use this bf16 checkpoint — use the community **fp8/GGUF** path (§ below).
- **Always start 1024×1024**, push to 1536/2048 only if headroom.

## Prompt refiner (optional — off by default)
`prompt_agent.py` rewrites a terse instruction into a dense prompt via **Gemma-4-31B-it** (local, heavy) or
any OpenAI-compatible API (`--backend api --base_url … --api_key …`). Output JSON `prompt` field → feed to
`inference.py`. Not needed for hand-written prompts; skip for now.

## NSFW — "разберёмся потом"
Base O1 doesn't advertise uncensored. Community uncensored variants already exist (HiDream lineage):
Civitai **Rebels HiDream-O1-Image** (2612142) / **-Dev** (2611889), uncensored ComfyUI workflows.
Reported maturity: **topless reliable, full nudity still weak**; CLIP/T5 hurt NSFW. Revisit when we test.

## Alt path — fp8/GGUF in ComfyUI (NOT the official repo)
Official repo has no fp8/GGUF/ComfyUI. Community: `Comfy-Org/HiDream-O1-Image` repack + GGUF nodes
(upfront-dequant). Lower VRAM, fits smaller cards, but a separate stack from this headless runner.

## Files
- `setup_hidream_o1.sh` — clone official repo + install pinned deps + flash-attn (+ line-341 fallback patch)
- `dl_hidream_o1.sh` — download Dev-2604 (+ optional full / Prompt-Refine)
- `run_hidream_o1.py` — batch T2I: loops a prompts file, calls the official `inference.py` per prompt with
  the correct `--model_type dev` flags (⚠️ reloads the 8B model per prompt — fine for small batches; an
  in-process batch loop is a pod-side optimization once we read `models/pipeline.py` load path).

**Confirmed remotely:** Dev-2604 = full checkpoint (8 bf16 shards, ~35 GB, config.json + Qwen3VL processor),
no LoRA/merge. argparse flags used by `run_hidream_o1.py` all exist in `inference.py`.
**Still verify on the pod (can't test here):** real VRAM/OOM at 1024 vs 2048; flash-attn build on the pod's
CUDA (else auto-patch); torch>=2.10 / transformers==4.57.1 vs the pod's base image.
