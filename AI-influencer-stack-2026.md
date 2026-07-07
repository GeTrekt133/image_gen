# AI-influencer / insta-like стек в ComfyUI — состояние на июль 2026

> Deep-research отчёт (104 агента, ~474 web-запроса, adversarial-верификация фактов в 3 голоса,
> факт убивается при 2/3 refute). Снимок быстро устаревает — перепроверяй версии.

## Итог одной строкой
Сцена сместилась на новое поколение баз: **Z-Image Turbo** — лидер по доступности/цене,
**FLUX.2 Dev** — чуть выше по детали/следованию промпту (но дорогой), **FLUX.1-dev угасает**
(legacy + non-commercial лицензия), **SDXL** держится на экосистеме. Костяк консистентности
персонажа — **обученная character-LoRA**, а не face-адаптеры.

## 1. Базовые модели — на пике vs угасающие

| Модель | Статус | Факты (verified) |
|---|---|---|
| **Z-Image Turbo** (6B, Apache 2.0, S3-DiT) | 🔥 лидер доступности | 8 шагов, ~2-3 c/1024px (4090); BF16 14-16G, fp8 ~8G, GGUF 5-6G; #1 open по Elo; 100 img ≈ 279 c |
| **FLUX.2 Dev** (32B) | ⬆️ топ качества, дорогой | «чуть-чуть» опережает Z-Image по prompt-adherence/детали; 100 img ≈ 1152 c (×4 медленнее); 32G+ (Q8) |
| **Qwen-Image / Edit-2512** (20B) | ✅ силён в контроле | мульти-референс (лицо+одежда+поза), региональный эдитинг, нативный pose/ControlNet |
| **SDXL** (Juggernaut XL, RealVisXL) | ➡️ живёт на экосистеме | 5000+ LoRA на Civitai, глубочайший ControlNet/inpaint-стек |
| **FLUX.1-dev** (12B, авг 2024) | 📉 угасает | часть гайдов ещё зовёт «best overall», но non-commercial лицензия, вытеснен FLUX.2 и Z-Image |

## 2. Важные поправки (что НЕ подтвердилось на верификации)
- ❌ «FLUX.2 слабая по коже/лицам» — **refuted (1-2)**. Наоборот, держит небольшой перевес по детали.
- ❌ «Z-Image однозначно лучший по коже vs всех» — **refuted (1-2)**. Он лидер по цене/скорости/Elo, но суперлатив по коже не выдержал.
- ❌ «Qwen Edit даёт ~100% консистентность с одного фото» — **refuted (0-3)**.
- Вывод: одного бесспорного победителя нет — выбор по бюджету VRAM/скорости; «near-100%» — маркетинг.

## 3. Workflow консистентного персонажа
1. **Character-LoRA = костяк identity** (не адаптеры). Каноничный инфлюенсер-пайплайн (NextDiffusion)
   тренирует char-LoRA и вообще не использует InstantID/PuLID/IPAdapter.
2. **Полный стек:** `character-LoRA (внешность) + IP-Adapter (кадры) + ControlNet (поза)`.
3. **Face-адаптеры** (PuLID / InstantID / EcomID): ни один не даёт 100%; InstantID лучше на ракурсах,
   PuLID — фронтально, EcomID — гибрид → лучшая практика = комбинировать.
4. **Финишинг от «пластика»:** FaceDetailer / low-sigma проход по лицу (~97% узнаваемость на
   SDXL+char-LoRA) + skin-realism пасс + upscale.

## 4. Тренировка LoRA — под какие базы
- **AI-Toolkit (Ostris)** — стандарт для новых: FLUX.2 dev/klein, **Z-Image**, Qwen.
- **Kohya / Musubi Tuner** — SD1.5, SDXL, FLUX.1; Musubi ещё Qwen/Wan 2.2.
- Тренировка идёт **прямо в ComfyUI** (comfyUI-Realtime-Lora) под Z-Image, FLUX Klein, Qwen, Wan 2.2, SDXL.

## 5. Видео-ветка (reels)
- **Wan 2.2** — доминирует: FP8 + Lightning/LightX2V distill-LoRA → быстрый (~6 шагов) image-to-video.
- **Wan2.2-S2V (14B)** — говорящий аватар из аудио: фото + аудио → липсинк-видео (480/720p).
- LTX-Video / HunyuanVideo — альтернативы; Wan 2.2 лидирует по стабильности лица.

## 6. Рекомендованный стек «сегодня»
```
База:        Z-Image Turbo (скорость/VRAM)  ИЛИ  FLUX.2 Dev (максимум детали, если есть VRAM)
Персонаж:    character-LoRA (ai-toolkit) ← ГЛАВНОЕ
Кадры/поза:  IP-Adapter + ControlNet
Лицо:        PuLID + InstantID гибрid (опционально)
Финиш:       FaceDetailer + skin-realism + upscale
Видео:       Wan 2.2 (i2v) + Wan2.2-S2V (говорящий аватар)
```

## 7. Оговорки и открытые вопросы
- Быстро устаревает (июль 2026); версии сдвинулись (Juggernaut «Ragnarok», RealVisXL V5, Qwen-Edit-2511).
- Тайминги (279 vs 1152 c) — из одиночных блог-бенчей, directional.
- **HiDream-O1, Chroma1-HD, SD3.5** — по ним не выжило ни одного проверенного факта; статус не освещён.

## Ключевые источники
- Z-Image: huggingface.co/Tongyi-MAI/Z-Image-Turbo (arxiv 2511.22699); localaimaster.com/blog/z-image-turbo-comfyui
- Model rundown: medium.com/diffusion-doodles/model-rundown-z-image-turbo-qwen-image-2512-edit-2511-flux-2-dev
- Benchmark: digitalocean.com/community/tutorials/image-generation-model-review
- Инфлюенсер-пайплайн: nextdiffusion.ai/tutorials/create-consistent-ai-influencers-comfyui-earn-online-fanvue
- Face-адаптеры: myaiforce.com/flux-pulid-vs-ecomid-vs-instantid/
- LoRA-тренировка: sanj.dev/post/lora-training-2025-ultimate-guide/ ; github.com/shootthesound/comfyUI-Realtime-Lora
- Видео: huggingface.co/Wan-AI/Wan2.2-S2V-14B ; runcomfy.com wan2-2-s2v workflow
