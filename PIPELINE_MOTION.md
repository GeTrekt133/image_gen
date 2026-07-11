# Motion transfer: Wan2.2-Animate + SCAIL-2 (i2v + перенос движения)

> Сессия 2026-07-11, RTX PRO 6000 Blackwell 96GB. Дополняет `PIPELINE_VIDEO.md` (LTX-стек).
> Итог: **оба пайплайна работают headless через ComfyUI API, с выбором фона из 3 источников.**

## TL;DR
- **`run_wanimate.py`** — Wan2.2-Animate-14B (fp8 KJ): DWPose (тело+руки) + face-crops
  ведут персону с картинки; 1:1 повтор движения и мимики driving-видео, кадрирование
  как в референсе движения. 77 кадров @ 16fps 576×1024 ≈ **80–105 с** (6 шагов, lightx2v).
- **`run_scail2.py`** — SCAIL-2 14B (Wan2.1-based, fp8): движение из сырых RGB-кадров
  (без скелета!) + цветные SAM3-маски идентичностей. Сам выбирает композицию (обычно
  full-body), движение чуть свободнее, зато нет глюков скелетной детекции. 81 кадр
  @ 30fps 512×896 ≈ **65–75 с**.
- Оба: `--bg ref|video|image` — фон с референс-картинки / из driving-видео / с отдельной
  картинки. Модели общие: umt5_xxl_fp8, clip_vision_h, wan_2.1_vae, lightx2v distill LoRA,
  sam3.1_multiplex (маски).

## Режимы фона
| --bg | Wan-Animate | SCAIL-2 |
|---|---|---|
| `ref` | Move mode: фон из картинки персоны, маски не нужны | Animation mode (`replacement_mode=False`) |
| `video` | Replacement: SAM3-маска танцора → blockify → зачернение + relight LoRA | Replacement mode (`True`); **сцену ОБЯЗАТЕЛЬНО описать в промпте** — иначе модель придумает фон сама |
| `image` | как video, но подложка = RepeatImageBatch(картинка) с зачернённой зоной маски | пре-композит: SAM3-маска персоны → ImageCompositeMasked на фон → Animation mode |

## ⚠️ Гочи (стоили по прогону каждая — НЕ повторять)
1. **Чёрный ореол в replacement (Wan-Animate).** `character_mask` при VAE-даунскейле в латент
   «усыхает» относительно зачернённой зоны → чёрная кайма. В шаблоне это решает нода
   `BlockifyMask` (KJNodes), в ядре её нет — эмулируем: GrowMask(10) → MaskToImage →
   ImageScale area W/32×H/32 → nearest-exact обратно → ImageToMask → ThresholdMask(0.05).
   Зачернение и character_mask — ОДНА И ТА ЖЕ блокифицированная маска.
2. **`--bg image` без зачернения = пустая сцена.** Replacement-обучение Wan-Animate ждёт
   «чёрную дыру» на месте персонажа; на чистой подложке персона не рисуется вовсе.
3. **`WanSCAILToVideo.previous_frame_count` обязателен** (=5), иначе 400 на валидации.
4. **DWPose: onnx-детектор падает на CPU/OpenCV** (onnxruntime без CUDA EP) — оба детектора
   брать `.torchscript.pt` (yolox_l + dw-ll_ucoco_384_bs5) → GPU через torch, препроцессинг
   секунды вместо минут. onnxruntime-gpu не ставить (CUDA 13 конфликты).
5. **SCAIL-2 replacement: сцена живёт в промпте.** Driving-кадры дают позу (half-res
   conditioning), фон при cfg=1+distill модель берёт из текста. Шаблонный промпт Comfy
   детально описывает сцену driving-видео — делать так же.
6. Driving для Wan-Animate — пересемплить в 16 fps (`ffmpeg -vf fps=16`), иначе слоу-мо;
   SCAIL-2 ест исходные 30 fps (fps выхода = fps входа).

## Модели (все скачаны, `dl_wan_animate.sh`/`dl_scail2.sh` в скретчпаде сессии)
| Файл | Размер | Папка |
|---|---|---|
| Wan2_2-Animate-14B_fp8_e4m3fn_scaled_KJ (Kijai/WanVideo_comfy_fp8_scaled) | 18.4G | diffusion_models/ |
| wan2.1_14B_SCAIL_2_fp8_scaled (Comfy-Org/SCAIL-2; есть fp16 32.8G) | 17.7G | diffusion_models/ |
| umt5_xxl_fp8_e4m3fn_scaled | 6.7G | text_encoders/ |
| sam3.1_multiplex_fp16 (Comfy-Org/sam3.1) | 1.75G | checkpoints/ |
| clip_vision_h | 1.26G | clip_vision/ |
| wan_2.1_vae | 254M | vae/ |
| lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16 | 738M | loras/ |
| WanAnimate_relight_lora_fp16 (только для --bg video/image) | 1.4G | loras/ |
| wan2.1_SCAIL_2_DPO_lora_bf16 | 1.2G | loras/ |

## Ноды
- Ядро (ComfyUI ≥ 2026-07): WanAnimateToVideo, WanSCAILToVideo, SCAIL2ColoredMask,
  SAM3_VideoTrack/TrackToMask (headless-маски по текст-промпту «human» — вместо
  интерактивных SAM2+PointsEditor из шаблона), ImageCompositeMasked, ThresholdMask.
- Custom: `comfyui_controlnet_aux` (DWPreprocessor) — только для Wan-Animate.
  Зависимости ставились точечно (uv pip install importlib_metadata addict yacs omegaconf
  ftfy python-dateutil yapf onnxruntime einops fvcore) — полный requirements.txt НЕ ставить,
  снесёт пин numpy==2.4.6.

## Сэмплинг (оба — быстрый distill-путь из официальных шаблонов)
- Wan-Animate: KSampler 6 шагов, cfg 1, euler/simple, ModelSamplingSD3 shift 8, lightx2v 1.0.
- SCAIL-2: SamplerCustom + BasicScheduler 6 шагов, cfg 1, euler/simple, shift 5,
  DPO LoRA 1.0 + lightx2v 0.8. Качественный путь (без lightx2v): 40 шагов cfg 5.

## Полный рил + RIFE 30fps (`run_wanimate_full.py`)
- **Зачем.** Wan нативно 16 fps, а один чанк = 77 кадров ≈ 4.8с. Для целого рила (8.4с)
  нужна чанковая генерация + интерполяция до 30 fps.
- **Чанкинг.** Цепочка WanAnimateToVideo: чанк0 (offset 0), каждый следующий берёт
  `continue_motion` = накопленные кадры (нода сама режет до последних
  `continue_motion_max_frames`=5 для стыка) и `video_frame_offset` = слот-5 предыдущей
  ноды. pose/face/bg/mask считаются 1 раз над ВСЕМ клипом — чанки сикают по offset
  ВНУТРИ ноды (`pose_video[offset:]`, `character_mask[offset:]`). Кадры копятся `ImageBatch`.
  Число чанков авто: `1 + ceil((driving_frames - length)/(length - overlap))`. 132 кадра → 2 чанка.
  Стык проверен покадрово (±0.15с вокруг 4.8с) — рывка/дубля нет.
- **RIFE.** Нода зарегистрирована как **`"RIFE VFI"`** (с пробелом, не `RIFE_VFI`!),
  пакет `ComfyUI-Frame-Interpolation` (deps уже есть: kornia/einops/scipy/cv2; НЕ ставить
  requirements — снесёт пины). `multiplier=2` (16→32), `rife49.pth` качается при 1-м прогоне.
- **Точный 30 fps + аудио.** RIFE даёт 32 fps (2×16). Пост: `ffmpeg -vf fps=30 -t <dur>`
  ретаймит в ровно 30 и обрезает паддинговый хвост, `-map 1:a` муксит оригинальную дорожку
  рила (из `drive_30fps.mp4`) — длительность/аудио синхронны. Итог: 30/1 fps, 8.27с, 248 кадров.
- **Тайминг.** Весь рил (2 чанка + RIFE + ретайм) ≈ **206с генерации** на 96GB.
- Для SCAIL-2 та же схема продления — подграф Extend (`previous_frames`+`previous_frame_count`,
  `ColorTransfer` между чанками); в `run_scail2.py` пока один чанк (аналогичный full-раннер — TODO).

## Смена образа синхронно с рилом (`run_wanimate_outfits.py`)
- **Идея (как LTX reel-pipeline):** TransNetV2 режет рил на шоты -> каждый шот генерится
  СО СВОЕЙ реф-картинкой наряда той же девушки, единый seed -> concat (рез = переход рила).
- **Наряды под рил (`gen_reel_outfits.py`):** Z-Image генерит персону в нарядах рила по схеме
  `zimage_faceoff.py` — фикс. описание персоны + **SEED=7** + FaceDetailer держат ОДНО лицо;
  ОДНА и та же комната в промпте -> **стабильный фон** между шотами. Выход `reeloutfit_{1..4}.png`.
- **Режим фона (важно!):** брать **`--bg ref`**, НЕ `--bg image`. bg=ref не использует маску:
  свои длинные волосы, нет ореола, фон из реф-картинки (одинаковая комната = стабильно).
  bg=image/video конформят персону под силуэт ТАНЦОРА (у него каре) -> короткие волосы + маска.
- **24 fps компромисс:** `--fps 24` (нативно, без RIFE — гладко, без «плавучести» слоу-мо).
  Драйвинг пересемплить в 24: `ffmpeg -vf fps=24`.
- **Запуск:**
  ```
  python gen_reel_outfits.py                       # 4 наряда рила (Z-Image, ~10с каждый)
  python run_wanimate_outfits.py --driving drive_24fps.mp4 --fps 24 --bg ref \
      --outfits reeloutfit_1.png,reeloutfit_2.png,reeloutfit_3.png,reeloutfit_4.png
  ```
  Итог: `wanim_outfits_FINAL.mp4` (24fps, смена нарядов, стабильная комната, одно лицо, +аудио).
- **Тайминг:** 4 шота (106+30+29+33 кадра) ≈ 175+35+30+35 = **~275с** + Z-Image нарядов ~40с.

## Маска волос в bg=image/video (если всё же нужен композит)
- **feather >> blockify.** `--mask-mode feather` (MaskToImage->ImageBlur->ImageToMask, мягкий
  0..1 край) убирает жёсткий чёрный ореол, который даёт грубая 32px-блокификация на ярком фоне.
  Параметры: `--mask-grow 18 --mask-blur 9 --mask-sigma 4`.
- **Чёрная «дыра» ОБЯЗАТЕЛЬНА** (`--bg-hole on`): без зачернения зоны персоны (`off`) модель
  вообще не рисует персону (пустой фон). Проверено: hole=off -> пустой лофт.
- **Фундаментальное ограничение:** character_mask берётся из SAM3 по ТАНЦОРУ рила -> персона
  наследует его силуэт (короткие волосы). Для своих волос — только bg=ref (см. выше).

## Что выбрать
- Точный 1:1 повтор движения/мимики, липсинк с видео, вставка в исходный рил → **Wan-Animate**.
- Full-body композиция, свобода кадра, нет скелетных артефактов на сложных позах,
  выше стабильность одежды/лица между режимами → **SCAIL-2**.
- Длинные видео: оба чанкуются (Wan-Animate: `video_frame_offset`+`continue_motion`;
  SCAIL-2: `previous_frames`+`previous_frame_count`, см. Extend-подграфы шаблонов) — в
  раннерах пока один чанк (77/81 кадр), продление = следующий шаг.

## Файлы репо
- `run_wanimate.py` / `run_scail2.py` — построение API-графа + сабмит + поллинг.
- Входы в `ComfyUI/input/`: `persona_street.png`, `drive_16fps.mp4`, `drive_30fps.mp4`,
  `bg_loft.png` (сгенерён Z-Image: пустой лофт).
- Выходы: `ComfyUI/output/video/wanim_{ref,video,image}_*.mp4`, `scail2_{ref,video,image}_*.mp4`
  (версии `_00001_` у wanim video/image — битые прогоны до фиксов 1–2, оставлены для истории).
