# neuro-avatar — pod scripts & ComfyUI workflows

Provisioning scripts and ComfyUI workflow JSONs for a Vast.ai GPU pod (RTX PRO 6000 Blackwell 96GB)
used for AI-influencer image generation and character-LoRA training.

**Состояние на 2026-07-08:** фото-стек собран и выверен глазами (A/B на кропах 1:1) —
базы, pose-control, эдиты, финиш-пайплайн. Подробности и находки → `SETUP.md` (APPENDIX 2).

## Рекомендованный пайплайн (выверено)

```
Лента (поток):  Z-Image bf16 (4c) → FaceDetailer → 2K-рефайн → +Skin-LoRA   ≈ 24 c/кадр 2K
Hero-кадры:     Qwen-2512 bf16 (50 шагов, 1328², анти-пластик негатив)  или  FLUX.2 Q8-GGUF
Поза:           DWPose → Z-Image Fun-Union-2.1 (ModelPatchLoader → model_patches/!)  +4 c
Эдит/стиль:     Qwen-Edit-2511 (переодеть/снять слой/стиль с референс-картинки)  ~70-120 c
Идентичность:   char-LoRA (Phase 6, ai-toolkit) — ещё не тренирована
```

⚠️ Два критичных «на глаз» вывода: **Qwen — только 2512** (базовая qwen_image мыльная);
**FLUX.2 — только Q8-GGUF + bf16 Mistral** (fp8mixed «пластиковит» кожу).

## Scripts
- `provision.sh` — сборка пода с нуля (ComfyUI + 11 кастом-нод + тренеры: ai-toolkit, kohya, musubi).
- `startup.sh` — быстрый старт (докачка моделей + ComfyUI в tmux на **:10100**).
- `env.sh` — вход в окружение (venv `/venv/main`, `secrets.env`, HF-переменные).
- `download_models.sh` — идемпотентная закачка всего фото-стека (см. шапку файла про диск).
- `smoke_all.py` — смоук всех баз одним промптом с замером it/s + пиковой VRAM.
- `gen_matrix.py` — матрица 5 промптов × все базы (portrait/fitness/street/beach/evening).
- `smoke_submit.py` — сабмит одного воркфлоу в ComfyUI API.

## ComfyUI workflows (API-формат)
| Файл | Назначение |
|---|---|
| `wf_zimage.json` | Z-Image Turbo t2i (bf16, 8 шагов) — основа потока |
| `wf_qwen.json` | Qwen-Image-**2512** t2i (bf16, 50 шагов, 1328², анти-пластик негатив) |
| `wf_flux2_gguf.json` | FLUX.2 Dev t2i (**Q8 GGUF** + bf16 Mistral, 28 шагов) — рабочий |
| `wf_flux2.json` | FLUX.2 fp8 — легаси, для качества не использовать |
| `wf_sdxl_biglust / juggernaut / realvis.json` | SDXL-ветка |
| `wf_pose_zimage.json` | Pose-control Z-Image (DWPose → Fun-Union-2.1 патч) |
| `wf_finish_zimage.json` | Финиш «уровень D»: FaceDetailer + 2K-рефайн + Skin-LoRA |
| `wf_qwen_edit.json` | Qwen-Edit-2511: замена одежды / перенос стиля с референса |

## Not included (by design)
- `secrets.env` and any tokens — never committed.
- Third-party repos (`ComfyUI/`, `ai-toolkit/`, `kohya_ss/`, `musubi-tuner/`) and model weights — see `.gitignore`.

## Usage
`env.sh` expects a `secrets.env` next to it defining `HF_TOKEN` and `CIVITAI_TOKEN`.
Create it locally (chmod 600); it is gitignored.

Свежий под: `bash provision.sh && bash startup.sh` → ComfyUI на :10100 → `python smoke_all.py`.
После recycle не забыть 2 локальных патча кода (PyAV rotation + flux2fun делегат) — см. SETUP.md APPENDIX 2.
