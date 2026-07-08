# Видео-пайплайн LTX-2.3 — i2v + motion-control (dance transfer)

> Сессия 2026-07-08, RTX PRO 6000 96GB. Дополняет `PIPELINE_NSFW.md` (фото-стек).
> Итог: **i2v и motion-control (перенос танца с driving-видео на персону) работают headless.**

## TL;DR
- **База:** `ltx-2.3-22b-dev-fp8.safetensors` (Lightricks/LTX-2.3-fp8) через `CheckpointLoaderSimple`
  (чекпоинт самодостаточен: DiT+VAE). Текст-энк: `gemma_3_12B_it_clean_fp8` (Pavpif/ltx2-gemma3-text-encoder)
  через `LTXAVTextEncoderLoader`. 121 кадр @ ~512×896, 8-шаговые ManualSigmas, cfg 1, `euler_ancestral`.
- **Motion-control:** union IC-LoRA (`ltx-2.3-22b-ic-lora-union-control-ref0.5`) + DWPose с driving-клипа →
  `LTXAddVideoICLoRAGuide` (guide strength 0.75) + персона-якорь через `LTXVImgToVideoConditionOnly` (bypass=False).
- **Рабочий граф:** `wf_ltx_dance.json` (API-формат) + дебаг-раннер `run_ltx_matrix.py`.

## ⚠️ ГЛАВНЫЙ БАГ (стоил ~10 прогонов — НЕ повторять)
**IC-LoRA `ref0.5` требует гайд на половинном разрешении.** Вход `latent_downscale_factor` ноды
`LTXAddVideoICLoRAGuide` ОБЯЗАН быть **связью из `LTXICLoRALoaderModelOnly` (output slot 1)** — лоадер
сам отдаёт фактор. Литерал `1.0` ломает раскладку attention-токенов → **выход = сама контрол-карта**
(скелет/canny), инвариантно к промпту/сиду/силе/сэмплеру. При этом i2v (контрол off) работает — это
сбивает диагностику. Быстрый детектор коллапса: яркость кадров (f0 ~140, f20+ ~4 = коллапс) — встроен в `run_ltx_matrix.py`.

## Прочие гочи
- **kornia ≥0.8**: в `ComfyUI-LTXVideo/pyramid_blending.py` импорт `pad` из kornia падает → шим `pad = F.pad`
  (патч локальный, теряется при recycle — накатить заново, см. SETUP.md APPENDIX 2 стиль).
- **V3-ноды (`ResizeImageMaskNode`, DynamicCombo)** в API-формате: под-входы = плоские точечные ключи
  (`resize_type` = "scale to multiple", `resize_type.multiple` = 32). UI-экспорт делает это сам; при ручной
  конвертации UI→API виджеты «конвертированные-в-инпут» сохраняют слот в widgets_values (сдвиг индексов).
- **Sampler**: `euler_ancestral` (cfg_pp-вариант не нужен при cfg 1).
- **DWPose** надёжен на чётких full-body позах; driving-клип: одиночный танцор, полный рост, чистый фон.
- Gated `google/gemma` не нужен: чистый single-file энкодер — `Pavpif/ltx2-gemma3-text-encoder` (fp8 13GB).

## Модели (скрипт `dl_ltx.sh` + `dl_ref.sh`)
| Файл | Источник | Папка |
|---|---|---|
| `ltx-2.3-22b-dev-fp8.safetensors` (28G) | Lightricks/LTX-2.3-fp8 | checkpoints/ |
| `gemma_3_12B_it_clean_fp8.safetensors` (13G) | Pavpif/ltx2-gemma3-text-encoder | text_encoders/ |
| `ltx-2.3-22b-ic-lora-union-control-ref0.5` | Lightricks/…-Union-Control | loras/ltxv/ltx2/ |
| `ltx-2.3-22b-distilled-lora-384-1.1` | Lightricks/LTX-2.3 | loras/ltxv/ltx2/ |
| spatial/temporal upscalers | Lightricks/LTX-2.3 | latent_upscale_models/ |
| (опц. GGUF-путь) `ltx-2.3-22b-dev-Q4_K_M.gguf` + LTX23 VAEs | unsloth/LTX-2.3-GGUF | unet/, vae/ |
| (опц.) gemma_fp4 + text_projection (для UI-воркфлоу автора) | inflatebot / onewayomni | text_encoders/ |

## Ноды (сверх фото-стека)
`ComfyUI-LTXVideo` (+kornia-шим) · `comfyui_controlnet_aux` (DWPose) · `ComfyUI_essentials` ·
`ComfyUI-Video-Depth-Anything` · для UI-воркфлоу автора: `rgthree-comfy`, `VideoHelperSuite`, `KJNodes`,
`ComfyMath`, `Custom-Scripts (pysssss)`, `ComfyUI-GGUF`, `Easy-Use`, `mxToolkit`.

## Файлы репо
- `wf_ltx_dance.json` — рабочий motion-control граф (API): LoadVideo(driving)→DWPose→union guide + персона.
- `run_ltx_matrix.py` — матричный дебаг-раннер (конфиги × авто-детект коллапса + mid-кадры).
- `dl_ltx.sh` / `dl_ref.sh` / `inst_nodes.sh` — модели LTX / модели UI-воркфлоу / нод-пакеты.
- Референс: `drive_wf/1. LTX 2.3 All-In-One-1 260606-1.json` (UI, сабграфы) — рабочий, но headless не конвертится; наш API-граф эквивалентен его motion-ветке.

## Результаты
`gallery/ltx_matrix/`: A (union 1.0, guide 0.75) и B (union 0.5) — персона повторяет позу танца
(руки над головой из driving); C (контрол off) — санити i2v. ~25-65с на 121 кадр 512×896 (fp8, 96GB).

## Дальше
- Полный вертикальный клип: длинный driving (Sway Dance 31с) + NSFW-персона + spatial-upscale до 1080p.
- NSFW-danse: та же схема, промпт/персона из unStable-пайплайна (см. PIPELINE_NSFW.md).
- char-LoRA остаётся главным незакрытым пунктом для консистентности лица в видео.

## Липсинк (LipDub IC-LoRA) — итоги 2026-07-08
- **LipDub = редаб**: родной домен — видео УЖЕ говорящего человека (+аудио в дорожке файла). На статичном портрете артикуляция слабая, на говорящем видео — полная.
- **Аудио пересинтезируется** (`LTXVSetAudioRefTokens` = speaker-identity ref-токены, негативные RoPE-позиции; stage1 = EmptyLatentAudio). Точный трек: frozen-латент (slot 2 SetAudioRefTokens) в stage1-concat — но губы тогда слабее.
- **Рабочий рецепт говорящей персоны**: фронтальный i2v-реф + спич вмуксован в реф → official-разводка → в пост-обработке заменить дорожку реальным аудио (`lipsync_FINAL_persona.mp4`). Идеальный audio-driven talking head = Wan S2V (отдельный стек).
- Файлы: `wf_lipdub.json` (+nolora), `ui2api.py` (UI→API конвертер со всеми фиксами: dotted-keys V3, сдвиг виджетов, downscale-линк).
