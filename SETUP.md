# Pod setup spec — AI-influencer generation stack (real-flow testing)

You are **Claude Code running as root inside a Vast.ai GPU container**
(target now: **A100 80GB or RTX 6000 Blackwell 96GB** — full-precision + video fit).
Goal: stand up the **July-2026 AI-influencer stack** from `AI-influencer-stack-2026.md` and
make **most models testable in a real flow** — multiple image bases, the character-consistency
stack, a trained character-LoRA, and the video branch. Legal adult content is in scope
(Big Lust branch); hard limits per `CLAUDE.md`.

Secrets in env (sourced from `secrets.env` / `env.sh`): `HF_TOKEN`, `CIVITAI_TOKEN`.
**Never print secret values.**

## Operating rules
- Work in `/workspace`. Activate the env first: `source /workspace/env.sh` (venv `/venv/main`,
  `HF_HOME=/workspace/hf`, `HF_HUB_ENABLE_HF_TRANSFER=1`, `HF_XET_HIGH_PERFORMANCE=1`).
- Long downloads / servers / training run in **tmux** so an SSH drop can't kill them.
- **Idempotent**: skip anything already present (dir cloned, file non-zero, server up, LoRA trained).
- After each phase: **verify**, print `✅ Phase N` or `❌ Phase N` + the failure; fix before moving on.
- Downloads: HF via `hf download` (Xet, ~0.5 GB/s). Civitai via `aria2c` with the token **in the query
  string, NO `Authorization` header** (a header leaks to R2 → 403 on the signed URL). Reuse the
  `dl_hf` / `dl_civitai` helpers in `download_models.sh`.
- Do **not** stop until every *Definition of Done* item passes; then print the final report.

---

## Phase 1 — System + env
- `apt-get install -y git git-lfs aria2 ffmpeg tmux jq build-essential`; verify `nvidia-smi`, python, pip.
- Ensure `/workspace/env.sh` exists and activates venv + exports HF vars + sources `secrets.env`.
  `torch.cuda.is_available()` → True. Note GPU model + VRAM (drive precision choices below).

## Phase 2 — ComfyUI + custom nodes
- ComfyUI at `/workspace/ComfyUI` (`pip install -r requirements.txt`), on a recent build (native
  Z-Image / FLUX.2 / Qwen-Image / Wan 2.2 support).
- `custom_nodes/` (skip if present, `pip install -r requirements.txt` each):
  - `ltdrdata/ComfyUI-Manager`, `ltdrdata/ComfyUI-Impact-Pack` (FaceDetailer)
  - `cubiq/ComfyUI_IPAdapter_plus`, `Fannovel16/comfyui_controlnet_aux`
  - `cubiq/ComfyUI_InstantID`, `balazik/ComfyUI-PuLID-Flux` (+ EcomID if a maintained node exists)
  - `ssitu/ComfyUI_UltimateSDUpscale`
  - **Video:** `kijai/ComfyUI-WanVideoWrapper` (Wan 2.2 i2v + S2V) — or native Wan nodes if the ComfyUI build has them.
  - Keep the working `custom_nodes/HiDream_O1-ComfyUI` if already installed.
- Verify all nodes import without error (check ComfyUI startup log).

## Phase 3 — Trainers
- `ostris/ai-toolkit` — primary (Z-Image / FLUX.2 / Qwen char-LoRA). `pip install -r requirements.txt`.
- `bmaltais/kohya_ss` — SDXL / FLUX.1 LoRA.
- `kohya-ss/musubi-tuner` — Qwen / Wan 2.2 LoRA (optional, for later video-LoRA).

## Phase 4 — Models → `ComfyUI/models/<subfolder>`
Use `download_models.sh` (extend it). Group by branch; place split files in the RIGHT subfolder
(`diffusion_models/ text_encoders/ vae/ checkpoints/ loras/ controlnet/ ipadapter/ instantid/
pulid/ clip_vision/ upscale_models/`). For repackaged models, prefer the **Comfy-Org** split-file
repos and resolve exact filenames via ComfyUI-Manager's model DB — verify each loads in `/object_info`.

**Image bases (the point — test most of them):**
- **Z-Image Turbo** (Apache, leader): `Tongyi-MAI/Z-Image-Turbo` (or Comfy-Org repackaged) →
  diffusion_models + `text_encoders/qwen_3_4b` + vae. 8-step, cheapest. *(already working on prior pod)*
- **FLUX.2 Dev** (top detail): `Comfy-Org/flux2-dev` split files (diffusion fp8mixed + Mistral text-enc + vae). *(working)*
- **Qwen-Image / Qwen-Image-Edit-2511+** (control / multi-reference): Comfy-Org Qwen-Image split files
  (diffusion + `text_encoders/qwen_2.5_vl` + vae). Edit variant enables face+clothes+pose reference.
- **SDXL-realism**: **Juggernaut XL** (Civitai `133005`, latest "Ragnarok") + **RealVisXL V5** (Civitai `139562`) → `checkpoints/`.
- **Big Lust v1.6** (SDXL-NSFW branch): Civitai `575395` → `checkpoints/`. *(working)*
- **HiDream-O1**: keep if already installed (research: unverified — smoke it, don't over-invest).

**Character-consistency stack:**
- **ControlNet**: FLUX `Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro-2.0`; SDXL control (openpose/depth via `controlnet_aux`); Qwen native pose.
- **IP-Adapter**: `h94/IP-Adapter` (+ SDXL/FLUX variants) → `ipadapter/` + `clip_vision/`.
- **Face adapters**: `InstantX/InstantID` → `instantid/`; PuLID-Flux weights → `pulid/`.

**Finishing:**
- Upscaler `4x-UltraSharp` (or RealESRGAN) → `upscale_models/`. FaceDetailer uses a face-detection model
  (bbox/seg) — let Impact-Pack fetch it. Optional skin-realism LoRA.

Verify: print a table `model | subfolder | size | loads? (in /object_info)`.

## Phase 5 — Launch + per-base smoke
- ComfyUI in tmux (session `comfy`). NOTE: on the prior pod it ran on port **10100** (Vast didn't forward 8188).
  Pick an **opened** port on this pod (check the Vast card / `-p` mapping); update `smoke_submit.py` if it changes.
- Poll until `/object_info` lists the checkpoints/diffusion models.
- Smoke **each base** via the API (reuse `wf_*.json`): **Z-Image, FLUX.2, Qwen-Image, SDXL (Juggernaut or RealVisXL), Big Lust**.
  Confirm an image per base in `output/`; record **it/s + peak VRAM** per base in a table.

## Phase 6 — Character-LoRA training (the identity backbone)
Per the research, char-LoRA — not adapters — is what makes a consistent persona.
- Prepare a small dataset (`/workspace/persona_A/dataset/`): ~15-25 consistent images of one face
  (if none provided, generate a seed face on FLUX.2/Z-Image + vary it, then curate). Caption them.
- Configure **ai-toolkit** for a **Z-Image** (fast) or **FLUX.2** (detail) LoRA (rank 16-32, ~1500-2500 steps).
- Train in tmux; save `persona_A_<base>.safetensors` → `ComfyUI/models/loras/`.
- Verify: generate the persona from the LoRA alone across ~8 seeds — face stays on-model, not "plastic".

## Phase 7 — Real-flow test (the deliverable)
Assemble the full consistency flow and produce a persona image set:
`base + character-LoRA(0.7-0.9) + IP-Adapter(framing) + ControlNet(pose) + FaceDetailer + upscale`.
- Save the workflow as `wf_realflow.json` (API format).
- Batch ~10 frames with `smoke_submit.py`/`generate.py`; confirm consistent identity + varied poses.

## Phase 8 — Video branch (reels)
- **Wan 2.2 i2v** (`Comfy-Org/Wan_2.2_ComfyUI_Repackaged` or Wan-AI repos, fp8 + Lightning/LightX2V distill-LoRA
  for ~6-step) → `diffusion_models/` + loras. Smoke: one persona still → short i2v clip in `output/`.
- **Wan2.2-S2V-14B** (`Wan-AI/Wan2.2-S2V-14B`) — talking avatar: still + audio → lip-synced clip. Smoke one.
- Record VRAM + seconds per clip.

---

## Definition of Done — every item ✅
- [ ] `torch.cuda` True; GPU/VRAM noted; ComfyUI serving, models visible in `/object_info`
- [ ] Per-base smoke image for **Z-Image, FLUX.2, Qwen-Image, SDXL-realism, Big Lust** (it/s + VRAM table)
- [ ] Consistency stack installed (IP-Adapter, ControlNet, InstantID/PuLID, FaceDetailer, upscaler) and loads
- [ ] **Character-LoRA trained** and reproduces the persona across seeds
- [ ] **Real-flow** batch produced (`wf_realflow.json` + ~10 consistent frames)
- [ ] **Wan 2.2 i2v** clip generated; (bonus) **Wan2.2-S2V** talking-avatar clip
- [ ] Trainers cloned (ai-toolkit, kohya, musubi); `startup.sh`/`provision.sh`/`download_models.sh` updated

**Final report:** model matrix (base | it/s | peak VRAM | loads | smoke ✅), char-LoRA result, real-flow
sample count, video timings, the ComfyUI URL, and any model that still needs a manual file.

---

# APPENDIX — working conventions carried over from the prior pod (2026-07-07)

- **Secrets** in `/workspace/secrets.env` (chmod 600), loaded via `source /workspace/env.sh`
  (activates venv `/venv/main`, sets `HF_HOME`, `HF_XET_HIGH_PERFORMANCE=1`). Never inline token values into committed files.
- **Downloads**: HF via Xet ~0.5 GB/s. **Civitai gotcha**: token in the `?token=` query, NO
  `Authorization` header (else the header rides to R2 and 403s the signed URL). See `dl_civitai`.
- **Port**: prior pod served ComfyUI on **10100** (Vast forwarded 8188? no). Confirm this pod's opened
  port and keep `smoke_submit.py` in sync.
- **Blackwell note** (if RTX 6000 / 5090-class): torch cu13x, `numpy==2.4.6` + `scipy>=1.13` pinned
  or ComfyUI won't start (learned on the 5080 sm_120 pod).
- **Model landscape**: full report in `AI-influencer-stack-2026.md`. TL;DR — Z-Image Turbo (cost/speed
  leader) · FLUX.2 Dev (top detail, pricey) · Qwen-Image/Edit (control) · SDXL-realism (ecosystem) ·
  FLUX.1-dev & SD3.5 fading. Consistency = char-LoRA first, adapters second. Video = Wan 2.2 / S2V.
- **Persistence**: no external store — models re-download on a fresh pod. Use Vast **Stop** to keep the
  disk; pull trained **LoRAs** off the pod (laptop / private HF repo) before **Destroy**.
