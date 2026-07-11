# Motion-transfer mechanisms — сводка

Перенос движения с референс-видео на AI-персону через ComfyUI (headless API на `:10100`).
Полная техничка и грабли — в [`PIPELINE_MOTION.md`](PIPELINE_MOTION.md); видео-LTX-стек — в
[`PIPELINE_VIDEO.md`](PIPELINE_VIDEO.md). Ниже — карта механизмов и точки входа.

## Модели
| Модель | Что | Файл-раннер |
|---|---|---|
| **Wan2.2-Animate-14B** (fp8) | Специализированный motion transfer: DWPose (тело+руки) + face-crops ведут персону с картинки, 1:1 повтор движения и мимики | `run_wanimate.py` |
| **SCAIL-2 14B** (Wan2.1-based, fp8) | Motion transfer без скелета — сырые RGB-кадры + цветные SAM3-маски идентичностей; свободнее по кадрированию | `run_scail2.py` |
| **Z-Image** (+FaceDetailer) | Генерация персоны в нужных нарядах/сценах, фикс. лицо через описание+seed | `gen_reel_outfits.py`, `zimage_faceoff.py` |
| **RIFE / ComfyUI-Frame-Interpolation** | Нейросетевая интерполяция кадров (нода `"RIFE VFI"`) | в `run_wanimate_full.py` |
| **TransNetV2** (`transnetv2-pytorch`) | Детекция границ шотов в риле | в `run_wanimate_outfits.py` |

## Раннеры (точки входа)
- **`run_wanimate.py`** — одиночный клип. Флаги `--bg ref|video|image`, `--length`, `--seed`,
  `--mask-mode feather|blockify`, `--bg-hole on|off`, `--prompt`, `--dump`.
- **`run_scail2.py`** — то же на SCAIL-2 (три режима фона).
- **`run_wanimate_full.py`** — ВЕСЬ рил: авто-чанкинг (`continue_motion`+`video_frame_offset`),
  RIFE (`--rife-mult`) или нативные fps (`--native-fps`, `--rife-mult 1`), ретайм `--out-fps` + аудио.
- **`run_wanimate_outfits.py`** — СМЕНА НАРЯДОВ синхронно с рилом: TransNetV2 → шоты →
  посегментные рефы (единый seed) → concat + аудио. `--fps`, `--bg ref|image`, `--outfits`.
- **`gen_reel_outfits.py`** — Z-Image генерит персону в нарядах рила (одна комната → стабильный фон).

## Ключевые механизмы и решения
1. **Режимы фона.** `ref` = фон с реф-картинки персоны (Move mode, без маски).
   `video` = вставка в сцену рила (Replacement + relight LoRA). `image` = отдельный статичный фон.
2. **Wan-Animate vs SCAIL-2.** Wan копирует кадрирование/тайминг рила 1:1 (+ мимика через
   face-crops); SCAIL сам строит full-body композицию, без скелетных артефактов.
3. **Relight LoRA** — только для Replacement (`bg video/image`) у Wan-Animate; НЕ ставить на
   SCAIL-2 (другая база Wan2.1) и не нужен в `bg ref`.
4. **Липсинк.** Wan-Animate переносит движение губ С ВИДЕО (не с аудио). Audio-driven talking
   head = отдельные Wan2.2-S2V / InfiniteTalk.
5. **Полный рил.** Native 16fps модели → чанкинг по 77 кадров; число чанков авто по длине рила;
   стык невидим (нода сшивает по последним 5 кадрам). Обрезка по длине рила убирает статичный хвост.
6. **fps / «замедление».** RIFE ×2 (16→32→30) даёт «плавучее» слоу-мо на быстром движении.
   Правильно — генерить нативно (16→24/30, feed driving в нужном fps, `--rife-mult 1`): все
   кадры реальные, движение чёткое. 24 fps — компромисс скорость/плавность.
7. **Смена образа.** Одним рефом Wan-Animate образ не меняет (identity-lock). Решение: сегментация
   рила по шотам (TransNetV2) + свой реф-наряд на шот + единый seed для лица + concat (рез = переход рила).
8. **Наряды под рил + стабильный фон.** Z-Image с фикс. персоной+seed+FaceDetailer генерит образы
   в ОДНОЙ комнате → `bg ref` (без маски) даёт свои длинные волосы, нет ореола, фон-комната стабильна.
9. **Маска волос (для bg image/video).** `feather` (MaskToImage→ImageBlur→ImageToMask, мягкий край)
   >> `blockify` (грубые 32px, чёрный ореол). Чёрная «дыра» обязательна (иначе персона не рисуется).
   Ограничение: маска берётся из силуэта ТАНЦОРА → персона наследует его причёску → для своих волос bg=ref.

## Тайминги (RTX PRO 6000, 96GB, 6-шаговый distill)
- ~**1с генерации на кадр** + ~40с прогрев на чанк.
- Одиночный клип 49 кадров: ~50–65с. Полный рил 24fps (4 чанка): ~275с.
- Смена нарядов (4 шота) + Z-Image нарядов: ~275с + ~40с.

## Веса (качаются скриптами, в git не коммитятся)
Wan2.2-Animate-14B fp8, SCAIL-2 14B fp8, umt5_xxl fp8, clip_vision_h, wan_2.1_vae,
lightx2v distill LoRA, WanAnimate relight LoRA, SCAIL-2 DPO LoRA, sam3.1_multiplex.
Ноды: `comfyui_controlnet_aux` (DWPose), `ComfyUI-Frame-Interpolation` (RIFE), ядро ComfyUI
(WanAnimateToVideo, WanSCAILToVideo, SAM3, SCAIL2ColoredMask).
