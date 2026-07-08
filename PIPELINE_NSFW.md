# NSFW pipeline на unStable Revolution (Z-Image Turbo) — рабочая связка

> Сессия 2026-07-08, RTX PRO 6000 Blackwell 96GB. Синтетическая 18+ персона (см. `CLAUDE.md`).
> Этот документ — точка входа для продолжения на новом поде. Дополняет `NSFW_ZIMAGE.md` (LoRA-путь)
> реальным рабочим стеком на полном NSFW-чекпоинте + финиш + pose/depth + edit.

## TL;DR — что использовать

- **База NSFW = полный чекпоинт `unStable Revolution ZIT v3 fp16`** (Civitai `2193942`, версия `2852808`),
  а НЕ NSFW-LoRA. LoRA `2279079` (23K) — это rank-4 «тень» этого же чекпоинта; полный чекпоинт
  самодостаточен (кожа + анатомия), NSFW-LoRA поверх него не нужна.
- **Генерация:** `UNETLoader(unstable) → ModelSamplingAuraFlow shift 3 → KSampler 8 шагов cfg 1
  res_multistep` (+ `ConditioningZeroOut` как негатив, cfg1 негатив игнорит). Текст-энк `qwen_3_4b`
  (lumina2), vae `z_image_ae`.
- **Финиш (универсальный, для ЛЮБОЙ картинки): `2K → skin-refine dn0.35 → FaceDetailer`.**
  Один вход-картинка → `finish.py <img>`. Порядок важен: FaceDetailer ПОСЛЕДНИМ на 2K.
- **Правки (добавить объект/акт, сменить одежду/позу руки):** `Qwen-Image-Edit-2511`. После Qwen —
  ОБЯЗАТЕЛЬНО финиш-блок (у Qwen кожа восковая).
- **Поза:** Fun-CN-2.1 (`ModelPatchLoader` → `model_patches/` → `QwenImageDiffsynthControlnet`).
  **strength 0.6 + 16 шагов.** Для explicit-ракурсов — **depth (DepthAnythingV2)**, не openpose.

## Ключевые инсайты (дорого достались — не повторять ошибки)

1. **Полный NSFW-чекпоинт > любая NSFW-LoRA.** unStable Revolution ZIT бьёт все LoRA (сравнение 7×7 в
   `run_lora_compare.py`): лучшая кожа + сильная анатомия одновременно, без стека.
2. **НЕ стекать z-image-base LoRA на unStable.** `Realistic Snapshot v5` / `Radiant Realism v2` обучены на
   базовом `z-image-turbo`; их дельты конфликтуют со смещёнными весами unStable → **галлюцинации**. На
   чистом Z-Image они ок, на unStable — ломают.
3. **Skin enhancer = низко-denoise img2img на unStable.** `VAEEncode(z_image_ae) → KSampler(unStable,
   denoise 0.35, res_multistep) → decode`. Кросс-модельный ре-скин: чинит восковую кожу Qwen и любую
   плоскую кожу. denoise — главный рычаг: 0.2 едва, **0.35 sweet spot**, >0.4 начинает менять сам кадр.
4. **Порядок финиша: 2K → skin-refine → FaceDetailer (FaceDetailer ПОСЛЕДНИМ).** На 2K лицо детейлится
   по большему кропу и не смазывается последующим рефайном → +12% лапласиана лица vs «FaceDetailer до 2K».
   FaceDetailer на 2K: `guide_size 768, max_size 1536, denoise 0.4`.
5. **Pose-control strength.** 1.0 = галлюцинация/вытянутая анатомия (turbo не успевает за 8 шагов). Sweet
   spot **0.6–0.7 + 16 шагов** (низкая сила + больше шагов держит позу без артефактов). 16 vs 24 — почти без разницы.
6. **Тип позы решает, каким контролом брать.** DWPose/openpose (скелет тела) надёжен ТОЛЬКО для
   вертикальных/видимых поз. На виде сзади (doggy) → **пустой скелет** (control не работает); на «ноги в
   камеру» (spread) → **мешанина** → галлюцинация. **Depth (DepthAnythingV2, тот же Fun-CN model_patch)**
   устойчив к форшортенингу/перекрытию/ракурсам сзади — **по умолчанию для NSFW-поз**. Openpose оставить для стоя/портрет.
7. **Qwen-Edit-2511 — точечные правки.** Конкретные глаголы действия + пространственная привязка («in her
   hand», «the tip penetrating», «lips wrapped around the tip») + «Keep everything else identical: same
   face, body, pose, lighting and background» → точная правка с сохранением идентичности/сцены. Кожа Qwen
   восковая → всегда прогонять через финиш-блок. Ноды нативные (TextEncodeQwenImageEditPlus +
   FluxKontextMultiReferenceLatentMethod + CFGNorm), 40 шагов cfg3 euler, shift 3.1.
8. **Civitai enumeration (см. `research_loras.py`).** Browse-API 403-ит без браузерного User-Agent; XXX-модели
   (nsfwLevel≥8) скрыты без `nsfw=true`; надёжный сигнал — поле `nsfwLevel` (флаг `model.nsfw` врёт).
   Итог переписи: ~1190 ZImageTurbo-LoRA, ~847 NSFW (XXX 348 / X-nude 165 / R-soft 334).
9. **Лицензия unStable:** `allowDerivatives: False`, `allowCommercialUse: {RentCivit}`. Локально
   генерировать можно; распространять дериватив — нет; коммерция формально ограничена. Для чистого
   коммерческого пайплайна (OnlyFans/Fanvue) — аргумент за собственную обученную NSFW-DoRA/базу.

## Модели (что качать — см. `dl_nsfw_pipeline.sh`)

| Модель | ID / repo | Файл → папка | Размер |
|---|---|---|---|
| **unStable Revolution ZIT v3 fp16** | Civitai ver `2852808` | `diffusion_models/unstable_revolution_zit_v3_fp16.safetensors` | 12 GB |
| Z-Image text-enc | `Comfy-Org/z_image_turbo` | `text_encoders/qwen_3_4b.safetensors` | 7.5 GB |
| Z-Image VAE | `Comfy-Org/z_image_turbo` | `vae/z_image_ae.safetensors` | 0.3 GB |
| Fun ControlNet Union 2.1 | `alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.1` | `model_patches/zimage_fun_controlnet_union_2.1.safetensors` | 6.7 GB |
| FaceDetailer bbox | `Bingsu/adetailer` | `ultralytics/bbox/face_yolov8m.pt` | 50 MB |
| Upscaler | `Kim2091/UltraSharp` | `upscale_models/4x-UltraSharp.pth` | 64 MB |
| Qwen-Image-Edit-2511 | `Comfy-Org/Qwen-Image-Edit_ComfyUI` | `diffusion_models/qwen_image_edit_2511_fp8mixed.safetensors` | 20 GB |
| Qwen 2.5-VL text-enc | `Comfy-Org/Qwen-Image_ComfyUI` | `text_encoders/qwen_2.5_vl_7b.safetensors` | 8 GB |
| Qwen VAE | `Comfy-Org/Qwen-Image_ComfyUI` | `vae/qwen_image_vae.safetensors` | 0.25 GB |
| DepthAnythingV2 (vitl) | авто через `comfyui_controlnet_aux` | `custom_nodes/.../ckpts/…` | ~1.3 GB |
| _(опц.)_ NSFW-LoRA сравнение | Civitai `2279079/2222911/2205140/2359268/2299623` | `loras/` | — |

## Ноды на поде (сверх базового ComfyUI)
`ComfyUI-Impact-Pack` + `ComfyUI-Impact-Subpack` (FaceDetailer + детектор) · `comfyui_controlnet_aux`
(DWPreprocessor + DepthAnythingV2Preprocessor). Qwen-Edit-ноды — нативные, доп. установка не нужна.
Blackwell: `numpy==2.4.6` + `scipy>=1.13`.

## Workflows (API-формат, для `smoke_submit.py` / раннеров)

| Файл | Что делает |
|---|---|
| `wf_unstable.json` | unStable t2i (base) |
| `wf_unstable_finish.json` | unStable t2i → 2K refine → FaceDetailer (ABCD-абляция) |
| `wf_unstable_ref_face.json` | тот же финиш, порядок 2K→FaceDetailer (утверждённый) |
| `wf_finish_block.json` + `finish.py` | **универсальный финиш**: LoadImage → 2K → skin-refine dn0.35 → FaceDetailer. Работает на любом кадре |
| `wf_pose_unstable.json` | pose-control (openpose/DWPose), strength 0.6 / 16 шагов |
| `wf_depth_unstable.json` | **depth-control (DepthAnythingV2)** — для explicit-поз |
| `wf_qwen_edit_dildo.json` | Qwen-Edit одиночная правка (add object/act) |
| `wf_qwen_edit_finish.json` | Qwen-Edit → 2K → skin-refine → FaceDetailer (всё в одном сабмите) |
| `wf_skin_refine.json` | только skin-refine пасс (демо/тюнинг denoise) |
| `wf_nsfw_var.json` + `run_lora_compare.py` | сравнение NSFW-LoRA вариантов |
| `wf_zimage_finish_nsfw.json` | ранний вариант (Z-Image turbo + NSFW LoRA + finish) — легаси |

Раннеры: `run_unstable_finish.py`, `run_explicit.py`, `run_lora_compare.py`, `run_qe_finish.py`,
`finish.py`, `research_loras.py`. Визуализации: `build_compare.py`, `build_unstable_finish.py`
(генерят самодостаточные HTML в `gallery/artifact/`).

## Setup на свежем поде (после Destroy)
1. `bash provision.sh` (или `prov_zimage.sh` — тримленный: ComfyUI + Impact-Pack/Subpack + numpy-пин)
   и `cd ComfyUI/custom_nodes && git clone Fannovel16/comfyui_controlnet_aux && pip install -r ...`.
2. `bash dl_nsfw_pipeline.sh` (unStable + Fun-CN + Qwen-Edit + face/upscale + z-image стек).
3. ComfyUI в tmux на :10100 (см. `startup.sh`).
4. Смоук: `python smoke_submit.py wf_unstable.json` → картинка в `output/`.

## Тайминги (RTX 6000 96GB)
t2i unStable ~3с (~30GB) · финиш-блок +12–24с (2K) · pose/depth ~14–24с · Qwen-Edit ~40–66с (грузит ~30GB).
Диск после всего стека: ~60–70 GB.

## Осталось (следующий под)
- **char-LoRA (Phase 6) — единственный незакрытый большой пункт.** Лицо сейчас плавает от кадра к кадру.
  Тренировать (ai-toolkit, DoRA rank 32–64) на `Tongyi-MAI/Z-Image` base или на unStable; инференс на
  unStable; вставлять в те же FaceDetailer/KSampler-ноды. Стек с NSFW уже вшит в базу → char-LoRA @~0.8.
- Опц.: своя NSFW-DoRA/база (лицензионно чистая, см. инсайт 9).
- Depth-pose батч; серия Qwen-Edit правок из одного базового кадра для консистентной сцены.
