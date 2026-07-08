# Reel motion-transfer pipeline (multi-shot → персона → склейка)

End-to-end: Instagram-reel → TransNetV2 shot-detection → нарезка driving-шотов →
motion-control генерация каждого шота на нашу персону → concat в вертикальный ролик.

## Шаги (2026-07-08, проверено на DZDXIh3iGPe: 5 шотов)
1. **Shot detection:** `pip install transnetv2-pytorch`;
   `TransNetV2().detect_scenes(video, threshold=0.5)` → список (start,end) кадров.
2. **Нарезка driving-шотов:** ffmpeg `-ss` + `-frames:v 49` (LTX требует 8k+1),
   `scale=512:896:force_original_aspect_ratio=increase,crop=512:896,fps=24`. Короткие/лишние шоты отбросить.
3. **Генерация:** `run_reel.py` — на каждый шот `wf_ltx_dance.json` с:
   LoadVideo=drive_N.mp4, LoadImage=persona (full-body), union 1.0 / guide 0.75,
   **единый noise_seed=42** → консистентная идентичность между шотами. Длина = кадры driving (GetImageSize).
4. **Склейка:** ре-энкод в единый формат (1024×1792, 24fps) → `ffmpeg -f concat` → монтажные cut'ы.

## Ограничения
- Фон/кадрирование одинаковые во всех шотах (persona-якорь один). Для смены локаций:
  свой якорь-кадр на шот (Qwen-Edit персона в кадр локации) ИЛИ depth-ветка union.
- Union@0.75 переносит преимущественно верх тела; сидячие/лежачие позы держатся слабее.

Файлы: `run_reel.py`, `wf_ltx_dance.json`, TransNetV2 (pip). Результат-пример: gallery/reel_out/reel_FINAL.mp4.
