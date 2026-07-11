# HiDream-O1-Image-Dev-2604 — headless T2I workflow

> Studied from the **official repo** `github.com/HiDream-ai/HiDream-O1-Image` (not guessed).
> 8B **pixel-native Unified Transformer** on a **Qwen3-VL backbone** — VAE-less, no separate text
> encoder, generates raw pixels up to 2048×2048. MIT. Tops open-source realism benchmarks
> (beats FLUX.2 Dev / Qwen-Image at 7×/3.4× fewer params). NSFW: **base not confirmed uncensored —
> decide later** (community uncensored variants exist, see bottom).

## ⚠️ The facts that break it if you get them wrong
- **Dev-2604 requires the `dev` BRANCH of the repo** (README model table: "inference.py (dev branch)").
  main's inference.py loads the model in **bfloat16** → visibly SOFT, low-detail images with this fp32
  checkpoint (confirmed by pod A/B, same prompt+seed). dev branch = **`torch_dtype=torch.float32`**
  (code comment: "float32 will generate more detailed images") + noise defaults 8.0/8.0/**clip_std 8.0**
  (main had 7.5/7.5/2.5) + default seed 42. `setup_hidream_o1.sh` clones/checks out `dev`.
- **Loads as `Qwen3VLForConditionalGeneration`** (`device_map="cuda"`), **NOT** a diffusers pipeline.
  It's `inference.py` + `models/pipeline.py`, driven by CLI.
- **Dev params are set INTERNALLY by `--model_type dev`** → **28 steps · guidance_scale 0.0 (CFG OFF) ·
  shift 1.0 · scheduler `flash`**. **Do NOT hand-pass `--guidance_scale`/`--shift`** (argparse defaults
  5.0/3.0 are for the FULL model). (A blog claiming "dev CFG 5.0" is WRONG — verified against the code.)
- **Resolution is snapped to ~2048-class buckets** (`PREDEFINED_RESOLUTIONS`: 2048², 2304×1728,
  2560×1440, 2496×1664, …). Asking 1024×1024 prints `[warning] Resolution snapped … to 2048x2048` —
  **you always pay 2048-class compute**; pick width/height only to choose the aspect ratio.
- **No conversion step.** `--model_path` points straight at the downloaded HF repo dir.
- **Deps are pinned:** `torch>=2.10`, **`transformers==4.57.1`**, diffusers/accelerate/einops/… .
  PyTorch **2.9.x is known-broken** — use ≥2.10. (Pod-verified on torch 2.12.0+cu130.)
- **flash-attn: skip it.** dev branch runs fp32 (flash-attn kernels are fp16/bf16-only) and the SDPA
  fallback is fast (2.37 it/s @2048² on RTX 6000 Blackwell). Two things needed without it, both handled
  by our scripts: patch `models/pipeline.py` `"use_flash_attn": True→False` (line ~341), and export
  **`FA_VERSION=auto`** (dev branch hard-imports `flash_attn` at import time otherwise — main defaulted
  to `auto`, dev defaults to `2`).

## Params (from the code)
| model_type | steps | guidance_scale | shift | scheduler | dtype |
|---|---|---|---|---|---|
| **dev** (2604) | **28** | **0.0** | **1.0** | flash (t2i) / flow_match (edit) | bf16 |
| full | 50 | 5.0 | 3.0 | default | bf16 |

Other argparse defaults: `--seed 32`, `--height/--width 2048`, `--noise_scale_start/end 7.5`,
`--noise_clip_std 2.5`. Native res is 2048² but **pixel-native = heavy** — see VRAM.

## Models — POD A/B VERDICT (2026-07-11): use FULL for photo quality
| Repo (HF) | What |
|---|---|
| `HiDream-ai/HiDream-O1-Image` | **full, 50-step, CFG 5 — the photo-quality checkpoint.** Real skin microtexture (pores/freckles), separated hair strands, film-like grade. ~65 s/img. |
| `HiDream-ai/HiDream-O1-Image-Dev-2604` | distilled 28-step CFG-off — **fast drafts only**: same composition but visibly smoother "AI-gloss" skin (distillation eats microtexture). ~40 s/img. |
| `HiDream-ai/Prompt-Refine` + `google/gemma-4-31B-it` | reasoning prompt refiner. Skip the 31B model — **write SCALIST-style dense prompts by hand/LLM** (single-paragraph Creative Director's Brief, see `prompt_agent_v2.py` REWRITE_SYSTEM_PROMPT); dense prompts measurably beat short tag-style ones. |

## Run
```bash
# 1. setup (clone official repo + venv deps + flash-attn/patch)
bash setup_hidream_o1.sh
# 2. download weights
bash dl_hidream_o1.sh
# 3a. single image (mirrors the official example exactly; FA_VERSION=auto — see flash-attn note)
FA_VERSION=auto python HiDream-O1-Image/inference.py \
    --model_path models/hidream-o1-dev-2604 \
    --prompt 'A dog holds a sign that says "HiDream-O1-Image release."' \
    --output_image results/t2i_dev.png \
    --model_type dev
# 3b. batch from a prompts file (one prompt per line; runner exports FA_VERSION itself)
python run_hidream_o1.py --model_path models/hidream-o1-dev-2604 \
    --prompts prompts.txt --outdir results/
# 3c. FINALS — full model (photo quality): --model_type full sets 50 steps/CFG 5 internally
python run_hidream_o1.py --model_path models/hidream-o1-full --model_type full \
    --prompts prompts.txt --outdir results/
```

## VRAM & speed (POD-MEASURED, RTX PRO 6000 Blackwell 96GB, torch 2.12.0+cu130)
Dev-2604 on disk = **8 safetensors shards, 33 GB, dtype F32** (checked the shard headers — the earlier
"~35 GB bf16" note was wrong; it's an fp32 dump of the ~8B multi-tower).
- **bf16 load (main branch — don't use):** measured **peak 18.6 GB** @2048², 28 steps, 3.15 it/s
  (~9 s denoise + ~10-17 s load). Fits a 24 GB card — but produces the soft low-detail output.
- **fp32 load (dev branch — the correct path):** weights ~33 GB + activations; 2.37 it/s @2048²
  (~12 s denoise + ~24 s load). Comfortable on 96 GB; should fit 48 GB (A6000/L40S) — verify OOM there.
- **full model (fp32, 50 steps + CFG≈2 passes):** 1.30 it/s @2048² → ~38 s denoise + ~24 s load ≈
  **~65 s/img**. Same 33 GB weights; ran fine on 96 GB.
- Every run is 2048-class regardless of requested size (see resolution snapping above), so there is no
  cheap 1024 mode to hide in.
- **≤24 GB:** bf16 quality loss or the community **fp8/GGUF** ComfyUI path (§ below).

## Prompt refiner — NOT optional for top quality (verified on pod 2026-07-11)
**The leaderboard "Dev-2604" is the pipeline refiner+model** (authors' words, HF discussion #2); the
official Space has "Rewrite prompt before generation" ON by default. Raw prompts — even hand-written
dense ones — measurably underperform: the refiner (`HiDream-ai/Prompt-Refine`, Gemma-4-31B FT, **59 GB**)
rewrites a terse brief into the model's training-caption distribution: dry, factual, spatially-anchored
scene descriptions (no "85mm f/1.8 bokeh" jargon). Pod A/B: refined prompts fixed signage text, scene
geography and background blobs on BOTH full and dev.
- Run locally: `run_refiner.py` (transformers, bf16, ~60 GB VRAM — unload generators first).
- **Gotchas:** needs `transformers>=5.x` (`gemma4` arch) while HiDream pins 4.57.1 — upgrade, refine,
  downgrade back (или отдельный venv); patch `tokenizer_config.json` `extra_special_tokens: [] → {}`;
  in transformers 5.x `apply_chat_template(..., return_dict=True)`.
- Refined prompts cache: `results/refined_prompts.json`.

## NSFW — "разберёмся потом"
Base O1 doesn't advertise uncensored. Community uncensored variants already exist (HiDream lineage):
Civitai **Rebels HiDream-O1-Image** (2612142) / **-Dev** (2611889), uncensored ComfyUI workflows.
Reported maturity: **topless reliable, full nudity still weak**; CLIP/T5 hurt NSFW. Revisit when we test.

## Alt path — fp8/GGUF in ComfyUI (NOT the official repo)
Official repo has no fp8/GGUF/ComfyUI. Community: `Comfy-Org/HiDream-O1-Image` repack + GGUF nodes
(upfront-dequant). Lower VRAM, fits smaller cards, but a separate stack from this headless runner.

## Files
- `setup_hidream_o1.sh` — clone official repo **(dev branch!)** + install pinned deps + SDPA patch (no flash-attn)
- `dl_hidream_o1.sh` — download Dev-2604 (+ optional full / Prompt-Refine)
- `run_hidream_o1.py` — batch T2I: loops a prompts file, calls the official `inference.py` per prompt with
  the correct `--model_type dev` flags (⚠️ reloads the 8B model per prompt — fine for small batches; an
  in-process batch loop is a pod-side optimization once we read `models/pipeline.py` load path).

**POD-VERIFIED 2026-07-11 (RTX PRO 6000 Blackwell 96GB), quality ladder from 3 A/B rounds
(same prompts+seeds, results in `results/`):**
1. ~~main branch + bf16~~ → soft, low detail (the "говно" tier). Never use with 2604.
2. dev branch fp32 + short prompts → much better textures, still stock-photo smooth skin.
3. dev branch + **SCALIST dense prompts** → strong prompt adherence, detail up; skin still AI-glossy.
4. FULL model + dense hand prompts → pores/freckles, separated hair, believable light; still loses
   scene coherence to Z-Image stack on insta-scenes (5-scene face-off, artifact page).
5. **+ official Prompt-Refine → the real config.** Signage renders correctly, scene geography coherent,
   background blobs gone; full+Refine trades blows with the Z-Image stack (wins scene/adherence, loses
   close-up skin polish and speed). dev-2604+Refine = the leaderboard combo: strong, but skin simpler
   than full. **Finals: brief → Prompt-Refine → full (50/CFG5/UniPC) → optionally Z-Image finish stack.**
Технически: flash-attn not needed (forward runs under bf16 autocast anyway — pipeline.py:334 — so SDPA
path is equivalent; `FA_VERSION=auto` required on dev branch). Note the pipeline autocasts to bf16
regardless of fp32 weights: the dev-branch gain comes from fp32 master weights + noise_clip_std 8.0.
Remaining ideas if we ever want more: run our finish stack (2K skin-refine + FaceDetailer) on top, or
the real Prompt-Refine agent via API.
