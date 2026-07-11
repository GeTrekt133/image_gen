"""Rebuild face-off page v2: Z-Image vs HiDream (hand prompts / +Refine full / +Refine dev)."""
import base64, io, os
from PIL import Image

SCENES = [
    ("selfie", "Селфи", "хендхелд, золотой вечерний свет, квартира"),
    ("gym", "Зал · спортивный топ", "кэнидид между подходами, дневной свет"),
    ("street", "Улица", "стрит-стайл, европейский город"),
    ("swimsuit", "Купальник", "инфинити-пул, голден-ауэр"),
    ("restaurant", "Ресторан", "вечер, свечи, чёрное сатиновое платье"),
]

COLS = [
    ("zimage_{s}", "Z-Image + финиш", "zi"),
    ("hidream_{s}", "HiDream full · ручной промпт", "hd0"),
    ("hidream_full_ref_{s}", "HiDream full + Refine", "hd"),
    ("hidream_dev_ref_{s}", "HiDream dev-2604 + Refine", "dv"),
]

def b64(path, w=640, q=80):
    im = Image.open(path).convert("RGB")
    im.thumbnail((w, w), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=q, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()

rows = []
for key, title, sub in SCENES:
    figs = []
    for tpl, label, cls in COLS:
        src = b64(f"results/faceoff/{tpl.format(s=key)}.png")
        figs.append(f'<figure><div class="chip chip-{cls}">{label}</div>'
                    f'<img src="{src}" alt="{label} — {title}" loading="lazy"></figure>')
    rows.append(f'''
<section class="scene">
  <div class="scene-head"><h2>{title}</h2><span class="scene-sub">{sub}</span></div>
  <div class="quad">{''.join(figs)}</div>
</section>''')

html = '''<title>HiDream vs Z-Image — раунд 2, с официальным рефайнером</title>
<style>
:root{
  --bg:#131312; --panel:#1c1b19; --line:#2c2a26; --ink:#ece8df; --mut:#9b968c;
  --zi:#5fb3a1; --hd0:#8a7a5c; --hd:#d9a441; --dv:#c96f4e; --maxw:1320px;
}
@media (prefers-color-scheme: light){:root{--bg:#f5f3ee;--panel:#fffdf8;--line:#e2ded4;--ink:#25231f;--mut:#6d685e;}}
:root[data-theme="light"]{--bg:#f5f3ee;--panel:#fffdf8;--line:#e2ded4;--ink:#25231f;--mut:#6d685e;}
:root[data-theme="dark"]{--bg:#131312;--panel:#1c1b19;--line:#2c2a26;--ink:#ece8df;--mut:#9b968c;}
html{background:var(--bg)}
body{margin:0;background:var(--bg);color:var(--ink);
  font:16px/1.55 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;}
.wrap{max-width:var(--maxw);margin:0 auto;padding:40px 24px 80px}
header h1{font-family:Georgia,'Times New Roman',serif;font-weight:600;font-size:clamp(24px,3.6vw,38px);
  margin:0 0 6px;text-wrap:balance}
header .lede{color:var(--mut);max-width:75ch;margin:0}
.finding{margin:22px 0 0;background:var(--panel);border:1px solid var(--line);border-left:3px solid var(--hd);
  border-radius:8px;padding:16px 20px;max-width:85ch}
.finding b{color:var(--hd)}
.meta{display:flex;flex-wrap:wrap;gap:10px;margin:20px 0 8px}
.meta div{background:var(--panel);border:1px solid var(--line);border-radius:6px;
  padding:8px 14px;font-size:13px;color:var(--mut)}
.meta b{color:var(--ink);font-weight:600}
.scene{margin-top:46px}
.scene-head h2{display:inline;font-family:Georgia,serif;font-weight:600;font-size:23px;margin:0 12px 0 0}
.scene-sub{color:var(--mut);font-size:14px}
.quad{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-top:12px}
@media (max-width:1080px){.quad{grid-template-columns:repeat(2,1fr)}}
@media (max-width:560px){.quad{grid-template-columns:1fr}}
figure{margin:0;background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:8px}
figure img{width:100%;height:auto;display:block;border-radius:4px}
.chip{font-size:11px;font-weight:600;letter-spacing:.3px;text-transform:uppercase;
  margin:0 0 7px;display:inline-block;padding:3px 9px;border-radius:99px}
.chip-zi{background:color-mix(in srgb,var(--zi) 18%,transparent);color:var(--zi)}
.chip-hd0{background:color-mix(in srgb,var(--hd0) 22%,transparent);color:var(--hd0)}
.chip-hd{background:color-mix(in srgb,var(--hd) 18%,transparent);color:var(--hd)}
.chip-dv{background:color-mix(in srgb,var(--dv) 18%,transparent);color:var(--dv)}
.verdict{margin-top:56px;background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:26px 28px}
.verdict h2{font-family:Georgia,serif;margin:0 0 14px;font-size:24px}
.verdict table{border-collapse:collapse;width:100%;font-size:14.5px}
.verdict th,.verdict td{text-align:left;padding:9px 12px;border-bottom:1px solid var(--line);vertical-align:top}
.verdict th{color:var(--mut);font-weight:600;font-size:13px;text-transform:uppercase;letter-spacing:.4px}
.verdict td:first-child{color:var(--mut);white-space:nowrap}
.notes{margin-top:18px;color:var(--mut);font-size:14.5px;max-width:80ch}
.notes b{color:var(--ink)}
</style>
<div class="wrap">
<header>
  <h1>HiDream vs Z-Image, раунд 2: недостающим звеном был официальный Prompt-Refine</h1>
  <p class="lede">Ресерч показал: на лидерборде «Dev-2604» — это связка «рефайнер + модель» (в официальном
  Space переписывание промпта включено по умолчанию). Подняли официальный Prompt-Refine (Gemma-4-31B,
  59 GB) на поде, отрефайнили те же 5 брифов и перегенерили. Сид 7, 2048², во всех вариантах.</p>
  <div class="finding">
  <b>Итог раунда 2:</b> рефайнер радикально поднял HiDream — точные вывески, связная география сцены,
  живые фоны. HiDream full + Refine теперь играет на равных с Z-Image-стеком и выигрывает
  у него сцену/контекст; Z-Image сохраняет перевес в полировке лица крупным планом и скорости.</div>
  <div class="meta">
    <div>GPU: <b>RTX PRO 6000 Blackwell 96GB</b></div>
    <div>Z-Image + финиш: <b>~20 c</b></div>
    <div>HiDream full: <b>~41 c</b> + рефайн ~20 c</div>
    <div>HiDream dev-2604: <b>~13 c</b> + рефайн ~20 c</div>
  </div>
</header>
''' + "".join(rows) + '''
<section class="verdict">
<h2>Вердикт после двух раундов</h2>
<table>
<tr><th>Критерий</th><th>Лидер</th><th>Комментарий</th></tr>
<tr><td>Кожа/лицо крупным планом</td><td style="color:var(--zi);font-weight:600">Z-Image + финиш</td>
<td>FaceDetailer-полировка всё ещё вне конкуренции на клоузапах; full+Refine сократил разрыв.</td></tr>
<tr><td>Сцена, контекст, реквизит</td><td style="color:var(--hd);font-weight:600">HiDream full + Refine</td>
<td>Читаемые вывески (CAFÉ DU COIN), связная география улицы/марины, осмысленные фоны без «блобов».</td></tr>
<tr><td>Следование брифу</td><td style="color:var(--hd);font-weight:600">HiDream + Refine</td>
<td>Рефайнер конвертирует бриф в точную пространственную раскладку — модель её исполняет дословно.</td></tr>
<tr><td>Скорость потока</td><td style="color:var(--zi);font-weight:600">Z-Image</td>
<td>20 c против 33–60 c (рефайн+генерация). Для ленты из десятков кадров это существенно.</td></tr>
<tr><td>Лидербордная связка (dev+Refine)</td><td style="color:var(--dv);font-weight:600">достойно</td>
<td>Сильно лучше голого dev, но кожа проще, чем у full — арене хватает, продакшену портретов нет.</td></tr>
</table>
<p class="notes"><b>Почему арена не совпала с нашим первым тестом:</b> (1) лидербордная сущность включает
рефайнер, а мы гоняли голую модель; (2) арена меряет широкий спектр задач (композиция, текст, знания),
а не «инста-портрет крупным планом» — нишу, под которую Z-Image Turbo затюнен специально.
<br><br><b>Рабочая схема дальше:</b> лента — Z-Image-стек как есть; сложные сцены — бриф → Prompt-Refine →
HiDream full → (опц.) финиш-стек Z-Image поверх для полировки лица. Рефайнер держать выгруженным и
поднимать пакетно: 59 GB VRAM не совместимы с одновременной загрузкой генераторов.</p>
</section>
</div>'''

out = "/tmp/claude-0/-workspace/9d8669da-c5e9-4ba0-934e-f8d362fd9f3d/scratchpad/faceoff.html"
open(out, "w").write(html)
print("written", out, f"{os.path.getsize(out)/1e6:.1f} MB")
