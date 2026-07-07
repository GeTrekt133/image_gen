# Pod setup spec — neuro-avatar generation stack

You are **Claude Code running as root inside an ephemeral Vast.ai GPU container**
(target: A100 80GB). Your job: build the full NSFW image-generation + character-LoRA
training stack **end to end, autonomously**, verifying every phase.

Secrets are already exported in your environment (sourced from `secrets.env`):
`HF_TOKEN`, `CIVITAI_TOKEN`. **Never print secret values.**

## Operating rules
- Work in `/workspace`.
- Run long downloads / servers inside **tmux** (or `nohup`) so an SSH drop doesn't kill them.
- Be **idempotent**: skip any step whose output already exists (dir cloned, file present with non-zero size, server already up).
- After each phase: **verify**, print `✅ Phase N OK` or `❌ Phase N` + the failure. If ❌, diagnose and fix before continuing.
- Do **not** stop until every item in *Definition of Done* passes. Then print the final summary.
- Downloads come straight from Hugging Face / Civitai (no external store). Use the tokens above for gated/authed downloads.

---

## Phase 1 — System + base
- `apt-get update && apt-get install -y git git-lfs aria2 ffmpeg tmux jq build-essential`
- Verify: `nvidia-smi` prints the GPU; `python3 --version`; `pip --version`.
- `pip install -U huggingface_hub hf_transfer`; export `HF_HUB_ENABLE_HF_TRANSFER=1` and `HF_HOME=/workspace/hf`.
- Authenticate HF: `huggingface-cli login --token "$HF_TOKEN"` (needed for gated FLUX.2).

## Phase 2 — ComfyUI + custom nodes
- Clone into `/workspace/ComfyUI`, `pip install -r requirements.txt`.
- Into `ComfyUI/custom_nodes/` clone (skip if present):
  - `https://github.com/ltdrdata/ComfyUI-Manager`
  - `https://github.com/cubiq/ComfyUI_InstantID`
  - `https://github.com/balazik/ComfyUI-PuLID-Flux`
  - `https://github.com/cubiq/ComfyUI_IPAdapter_plus`
  - `https://github.com/Fannovel16/comfyui_controlnet_aux`
  - `https://github.com/ltdrdata/ComfyUI-Impact-Pack`
  - `https://github.com/ssitu/ComfyUI_UltimateSDUpscale`
  - For each, if it has `requirements.txt`, `pip install -r` it.
- Verify: `python -c "import torch; print(torch.cuda.is_available())"` → `True`.

## Phase 3 — Trainers
- Clone `https://github.com/ostris/ai-toolkit` (FLUX.2 LoRA) → `pip install -r requirements.txt`.
- Clone `https://github.com/bmaltais/kohya_ss` (SDXL LoRA).
- Verify both dirs exist.

## Phase 4 — Models → `ComfyUI/models/<subfolder>`
Download **directly from HF / Civitai** into the correct subfolder. Idempotent: skip any
file that already exists with non-zero size.
- `checkpoints/`
  - **FLUX.2-dev** (fp8, gated): `hf download Comfy-Org/flux2-dev --local-dir checkpoints/flux2`
  - **Chroma** (uncensored, optional): `hf download lodestones/Chroma --local-dir checkpoints/chroma`
  - **Big Lust** (SDXL-NSFW): resolve the latest version →
    `url=$(curl -s https://civitai.com/api/v1/models/575395 | jq -r '.modelVersions[0].files[0].downloadUrl')`
    then `aria2c -x8 -o big-lust.safetensors "$url?token=$CIVITAI_TOKEN"` into `checkpoints/`.
  - **Pony V6 XL** (anime base): same via model id `257749`.
- `controlnet/`: `hf download Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro-2.0 --local-dir controlnet/flux-union`
- `instantid/`: `hf download InstantX/InstantID --local-dir instantid`
- `pulid/`, `ipadapter/`, `clip_vision/`, `upscale_models/`: fetch the models those nodes need
  (PuLID-Flux weights, IPAdapter models, CLIP-Vision, a `4x-UltraSharp`/`RealESRGAN` upscaler).
  Use ComfyUI-Manager's model DB (`python -m ...` or its CLI) or HF; put each in its correct subfolder.
- Verify: every expected file exists with non-zero size; print a table `model | subfolder | size`.

> No external store is used. A fully fresh pod will re-download these (fast on the pod's NIC).
> To avoid re-downloading between sessions, **Stop** the Vast instance (keeps its disk) rather
> than **Destroy**. Trained LoRAs are the one thing worth pulling off the pod (download to your
> laptop / push to a private HF repo) before you Destroy.

## Phase 5 — Launch + smoke test
- Start ComfyUI in tmux: `tmux new -s comfy -d 'cd /workspace/ComfyUI && python main.py --listen 0.0.0.0 --port 8188'`.
- Poll until `curl -s http://127.0.0.1:8188/ | head` returns HTML and `/object_info` lists the checkpoints.
- Run **two smoke generations via the API** (`POST /prompt`): one minimal SDXL graph on **Big Lust**,
  one on **FLUX.2**. Confirm an image lands in `ComfyUI/output/`. Record it/s (from logs) and peak VRAM (`nvidia-smi`).

## Phase 6 — Reproducibility for next pods
- Write `/workspace/startup.sh`: re-run the Phase 4 downloads (idempotent) + launch ComfyUI in tmux.
- Write `/workspace/provision.sh` summarizing phases 1-3 (so a fresh pod can re-run the whole build).

---

## Definition of Done — every item must be ✅
- [ ] `nvidia-smi` OK and `torch.cuda.is_available()` → True
- [ ] ComfyUI serving on `:8188`; checkpoints visible in `/object_info`
- [ ] Big Lust (SDXL) smoke image generated in `output/`
- [ ] FLUX.2 smoke image generated in `output/`
- [ ] All expected models present locally with non-zero size
- [ ] `ai-toolkit` + `kohya_ss` cloned
- [ ] `startup.sh` + `provision.sh` written

**Final report:** print a table with it/s + peak VRAM per model, the full model list with sizes,
the ComfyUI URL (`http://<pod-host>:8188`), and any manual follow-ups (e.g. a node that still
needs a model the user must supply).

---

# APPENDIX — обновления сессии (актуально на 2026-07-07)

## Credentials / секреты
Токены **лежат в плейнтексте в `/workspace/secrets.env`** (chmod 600) — это санкционированное
место (CLAUDE.md: секреты sourced from `secrets.env`). В саму спеку значения НЕ инлайним — файл
шарится/коммитится, а это живые креды. Формат:
```bash
# /workspace/secrets.env
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxx        # HF-аккаунт (текущий: GeTrekt)
export CIVITAI_TOKEN=xxxxxxxxxxxxxxxxxxxxxxxxxxxx  # Civitai
```
Подключение: `source /workspace/env.sh` (активирует venv `/venv/main`, грузит secrets, ставит
`HF_HOME=/workspace/hf`, `HF_XET_HIGH_PERFORMANCE=1`).

## Фактическое состояние этого пода
- **GPU:** RTX 5080 16 ГБ (не A100 80 ГБ из спеки). venv `/venv/main` = torch 2.12.0+cu130,
  Blackwell sm_120 работает. numpy pinned 2.4.6 + scipy>=1.13 (иначе ComfyUI не стартует).
- **ComfyUI:** запущен в tmux `comfy` на порту **10100** (проброшен наружу как `108.255.76.60:42091`;
  8188 у Vast НЕ проброшен). Смена порта учтена в `smoke_submit.py`.
- **Установленные модели (models/):** Big Lust v1.6 (SDXL, checkpoints/), FLUX.2 fp8-набор
  (diffusion_models + text_encoders/mistral + vae), Z-Image Turbo (diffusion_models + text_encoders/qwen_3_4b + vae),
  HiDream-O1 (Comfy-Org dev + gemma-4 encoder; + настоящий Dev-2604 fp8 через кастом-ноду
  `custom_nodes/HiDream_O1-ComfyUI`).
- **Скрипты:** `download_models.sh`, `startup.sh`, `provision.sh`, `env.sh`, `smoke_submit.py`,
  `wf_*.json` (готовые API-воркфлоу под каждую модель).
- **Замеры скорости скачки:** HF через Xet ~0.5 ГБ/с; Civitai через aria2c (`?token=` в query,
  БЕЗ Authorization-заголовка — иначе R2 отдаёт 403).

## Актуальный model-landscape (июль 2026) — для генерации людей / AI-influencer
Полный deep-research отчёт: **`/workspace/AI-influencer-stack-2026.md`**. Кратко:
- **На пике:** Z-Image Turbo (лидер цена/скорость/доступность, #1 open Elo), FLUX.2 Dev (топ детали,
  дорогой), Qwen-Image/Edit-2512 (контроль/эдитинг), SDXL-realism (Juggernaut/RealVisXL — на экосистеме).
- **Угасает:** FLUX.1-dev (legacy, non-commercial лицензия), SD3.5.
- **Костяк консистентности персонажа = обученная character-LoRA** (ai-toolkit для Z-Image/FLUX.2/Qwen,
  kohya/musubi для SDXL/FLUX.1), + IP-Adapter/ControlNet (кадры/поза) + PuLID/InstantID/EcomID гибрид
  (лицо, ни один не 100%) + FaceDetailer/skin-realism/upscale (финиш).
- **Видео (reels):** Wan 2.2 (i2v, FP8 + Lightning/LightX2V LoRA) + Wan2.2-S2V (говорящий аватар из аудио).
- **Рекомендованный стек:** char-LoRA на Z-Image (быстро) или FLUX.2 (детально) + IP-Adapter/ControlNet
  + face-адаптер + FaceDetailer + Wan 2.2 для видео.

## Ближайший следующий шаг
ai-toolkit уже установлен → не хватает только **обучения character-LoRA** (подготовка датасета →
конфиг → запуск в tmux) — это то, что делает узнаваемого персонажа во всех кадрах.
