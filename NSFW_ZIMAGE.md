# NSFW on Z-Image Turbo — what we use + how to run

Deep-research (104 agents, 3-vote adversarial verify, 2026-07). Bottom line: **Z-Image Turbo is
uncensored out of the box and already has a real NSFW-LoRA ecosystem** → we do NSFW natively on
Z-Image, no mandatory FLUX→SDXL handoff.

## What we use (this repo)
Z-Image **Turbo, BF16** — `z_image_turbo_bf16.safetensors` + `qwen_3_4b` text-enc (lumina2) +
`z_image_ae` vae (`Comfy-Org/z_image_turbo`). Sampling: 8 steps / cfg 1 / `res_multistep` /
`ModelSamplingAuraFlow shift 3` (see `wf_zimage.json`).

## Findings (verified)
- 🟢 **Z-Image = uncensored out of the box** `[high, 3×3-0]` — generates nude/explicit directly from
  prompts, no adapter. Confirmed on its [HF repo #50](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo/discussions/50)
  + adversarial [pollinations #6600](https://github.com/pollinations/pollinations/issues/6600) safety-bug report.
  (Hosted-API ToS bans it — irrelevant to local weights.)
- 🔴 **FLUX.2-dev** = heavily filtered (pre-train NSFW/CSAM removed + post-train T2I/I2I suppression per
  [its model card](https://huggingface.co/black-forest-labs/FLUX.2-dev)) → needs uncensor-LoRA. Not our path.
- 🟡 **Qwen-2512** = no refusal but weak explicit anatomy, immature NSFW-LoRA scene. Fallback only.
- ⚠️ **Gap:** no dedicated NSFW ControlNet/inpaint for the new-gen bases → for explicit-region inpaint /
  touch-ups, our **SDXL-NSFW (Big Lust) + SDXL inpaint** still has a role.

## Verified NSFW LoRAs for Z-Image **Turbo** (real Civitai IDs, checked via API)
| Downloads | Name | ID / link |
|---|---|---|
| 23,270 | Z-Image Turbo NSFW LoRA | [2279079](https://civitai.com/models/2279079) |
| 6,577 | Z-Image Turbo NSFW | [2299623](https://civitai.com/models/2299623) |
| 4,450 | Z-image-cosplay-NSFW (1200 img, Very Positive ×256) | [2212300](https://civitai.com/models/2212300) |
| 2,554 | Zimage nsfw LORA (train on base) | [2225054](https://civitai.com/models/2225054) |
| 2,456 | Z-Image turbo Anime nsfw | [2221829](https://civitai.com/models/2221829) |
| 1,340 | [LuisaP] Z-IMAGE HYPHORIA [NSFW BIASED] | [2209500](https://civitai.com/models/2209500) |
| 519 | **[Z-Image Turbo] Consistent character / NSFW** | [2543619](https://civitai.com/models/2543619) |
| 240 | ZIN (Z IMAGE NSFW) LORA v0.1 | [2343983](https://civitai.com/models/2343983) |

> Use `ZImageTurbo`-base LoRAs only. **Skip** `ZImageBase` ones (`2415629`, `2627180`) — they target the
> non-distilled base, sit poorly on Turbo. Full HF finetune tree:
> [Tongyi-MAI/Z-Image-Turbo finetunes](https://huggingface.co/models?other=base_model:finetune:Tongyi-MAI/Z-Image-Turbo).

⚠️ **Civitai gates NSFW behind login** — unauth API/browse returns `nsfw=false` and hides most. Enable
*Settings → Content → Show NSFW*; your `CIVITAI_TOKEN` bypasses the gate for downloads.

## Download (uses the existing `dl_civitai` / `civitai_url` helpers)
Already wired into `download_models.sh` section 7:
```bash
dl_civitai "$(civitai_url 2279079)" "$M/loras" "zimage_nsfw.safetensors"            "Z-Image NSFW LoRA (23K dl)"
dl_civitai "$(civitai_url 2543619)" "$M/loras" "zimage_consistent_nsfw.safetensors" "Z-Image Consistent char/NSFW"
```

## Workflow — `wf_zimage_nsfw.json`
Your `wf_zimage.json` + one `LoraLoaderModelOnly` node between `UNETLoader` and `ModelSamplingAuraFlow`:
```
UNETLoader(z_image_turbo_bf16) → LoraLoaderModelOnly(zimage_nsfw.safetensors, 1.0)
  → ModelSamplingAuraFlow(shift 3) → KSampler(8 steps, cfg 1, res_multistep)
```
Put explicit terms in the positive prompt. Settings unchanged from clean Z-Image (8 / cfg1 / res_multistep).

## Stacking with your character-LoRA (Phase 6)
Both LoRAs modify the same weights and **fight** ([ZipLoRA arXiv 2311.13600](https://arxiv.org/abs/2311.13600)) → balance:
```
UNETLoader → LoraLoaderModelOnly(nsfw, ~1.0) → LoraLoaderModelOnly(persona_char, ~0.8) → ModelSampling → KSampler
```
- Baseline: **char ~0.8 + NSFW ~1.0**.
- If the persona drifts (distilled Turbo weakens identity binding) → raise **char to 1.0-1.1** and/or run
  final frames at **full non-distilled steps on Z-Image *base*** (`Tongyi-MAI/Z-Image`).
- Train the char-LoRA on **base** (holds identity better), infer on Turbo — per Phase 6 plan.

## Legal
Legal adult content only. Never CSAM / non-consensual / real-person sexual impersonation.
