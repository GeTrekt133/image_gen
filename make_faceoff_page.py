"""Build the face-off comparison HTML (images embedded as data URIs)."""
import base64, io, os
from PIL import Image

SCENES = [
    ("selfie", "Селфи", "хендхелд, золотой вечерний свет, квартира"),
    ("gym", "Зал · спортивный топ", "кэнидид между подходами, дневной свет"),
    ("street", "Улица", "стрит-стайл, переход, глубокий фокус f/8"),
    ("swimsuit", "Купальник", "инфинити-пул, голден-ауэр, мокрая кожа"),
    ("restaurant", "Ресторан", "вечер, свечи, чёрное сатиновое платье"),
]

def b64(path, w=880, q=82):
    im = Image.open(path).convert("RGB")
    im.thumbnail((w, w), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=q, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()

rows = []
for key, title, sub in SCENES:
    hd = b64(f"results/faceoff/hidream_{key}.png")
    zi = b64(f"results/faceoff/zimage_{key}.png")
    rows.append(f'''
<section class="scene">
  <div class="scene-head"><h2>{title}</h2><span class="scene-sub">{sub}</span></div>
  <div class="pair">
    <figure><div class="chip chip-hd">HiDream-O1 full</div><img src="{hd}" alt="HiDream — {title}" loading="lazy"></figure>
    <figure><div class="chip chip-zi">Z-Image Turbo + финиш</div><img src="{zi}" alt="Z-Image — {title}" loading="lazy"></figure>
  </div>
</section>''')

html = '''<title>HiDream vs Z-Image — 5 инста-сцен</title>
<style>
:root{
  --bg:#131312; --panel:#1c1b19; --line:#2c2a26; --ink:#ece8df; --mut:#9b968c;
  --hd:#d9a441; --zi:#5fb3a1; --maxw:1180px;
}
@media (prefers-color-scheme: light){:root{--bg:#f5f3ee;--panel:#fffdf8;--line:#e2ded4;--ink:#25231f;--mut:#6d685e;}}
:root[data-theme="light"]{--bg:#f5f3ee;--panel:#fffdf8;--line:#e2ded4;--ink:#25231f;--mut:#6d685e;}
:root[data-theme="dark"]{--bg:#131312;--panel:#1c1b19;--line:#2c2a26;--ink:#ece8df;--mut:#9b968c;}
html{background:var(--bg)}
body{margin:0;background:var(--bg);color:var(--ink);
  font:16px/1.55 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;}
.wrap{max-width:var(--maxw);margin:0 auto;padding:40px 24px 80px}
header h1{font-family:Georgia,'Times New Roman',serif;font-weight:600;font-size:clamp(26px,4vw,40px);
  margin:0 0 6px;text-wrap:balance;letter-spacing:.2px}
header .lede{color:var(--mut);max-width:68ch;margin:0}
.meta{display:flex;flex-wrap:wrap;gap:10px;margin:20px 0 8px}
.meta div{background:var(--panel);border:1px solid var(--line);border-radius:6px;
  padding:8px 14px;font-size:13px;color:var(--mut)}
.meta b{color:var(--ink);font-weight:600}
.legend{display:flex;gap:16px;margin:18px 0 4px;font-size:13px;color:var(--mut)}
.dot{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:6px;vertical-align:-1px}
.scene{margin-top:46px}
.scene-head{display:baseline;margin-bottom:12px}
.scene-head h2{display:inline;font-family:Georgia,serif;font-weight:600;font-size:24px;margin:0 12px 0 0}
.scene-sub{color:var(--mut);font-size:14px}
.pair{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-top:12px}
@media (max-width:760px){.pair{grid-template-columns:1fr}}
figure{margin:0;background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:10px}
figure img{width:100%;height:auto;display:block;border-radius:4px}
.chip{font-size:12px;font-weight:600;letter-spacing:.4px;text-transform:uppercase;
  margin:0 0 8px;display:inline-block;padding:3px 10px;border-radius:99px}
.chip-hd{background:color-mix(in srgb,var(--hd) 18%,transparent);color:var(--hd)}
.chip-zi{background:color-mix(in srgb,var(--zi) 18%,transparent);color:var(--zi)}
.verdict{margin-top:56px;background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:26px 28px}
.verdict h2{font-family:Georgia,serif;margin:0 0 14px;font-size:24px}
.verdict table{border-collapse:collapse;width:100%;font-size:14.5px}
.verdict th,.verdict td{text-align:left;padding:9px 12px;border-bottom:1px solid var(--line);vertical-align:top}
.verdict th{color:var(--mut);font-weight:600;font-size:13px;text-transform:uppercase;letter-spacing:.4px}
.verdict td:first-child{color:var(--mut);white-space:nowrap}
.win-hd{color:var(--hd);font-weight:600}.win-zi{color:var(--zi);font-weight:600}
.notes{margin-top:18px;color:var(--mut);font-size:14.5px;max-width:75ch}
.notes b{color:var(--ink)}
</style>
<div class="wrap">
<header>
  <h1>HiDream-O1 full vs Z-Image Turbo: 5 инста-сцен</h1>
  <p class="lede">Честный тест 2026-07-11: каждый движок в своём рабочем флоу. HiDream-O1-Image full
  (fp32, 50 шагов, CFG 5, плотные SCALIST-промпты, 2048²). Z-Image Turbo — флоу из репы:
  wf_zimage (bf16, 8 шагов) → wf_finish_zimage (FaceDetailer + 4x-UltraSharp → 2K + рефайн).
  Одинаковые сцены и сид, промпты — в нативном стиле каждого движка.</p>
  <div class="meta">
    <div>GPU: <b>RTX PRO 6000 Blackwell 96GB</b></div>
    <div>HiDream: <b>~41 c/кадр</b></div>
    <div>Z-Image + финиш: <b>~20 c/кадр</b></div>
    <div>Финальное разрешение: <b>2048² оба</b></div>
  </div>
  <div class="legend">
    <span><span class="dot" style="background:var(--hd)"></span>слева — HiDream-O1 full</span>
    <span><span class="dot" style="background:var(--zi)"></span>справа — Z-Image Turbo + финиш</span>
  </div>
</header>
''' + "".join(rows) + '''
<section class="verdict">
<h2>Вердикт</h2>
<table>
<tr><th>Критерий</th><th>Кто сильнее</th><th>Комментарий</th></tr>
<tr><td>Кожа / лицо крупным планом</td><td class="win-zi">Z-Image</td>
<td>FaceDetailer + финиш дают полированную «модельную» кожу; лицо красивее и стабильнее от сцены к сцене.</td></tr>
<tr><td>Следование промпту</td><td class="win-hd">HiDream</td>
<td>Полотенце на плече, толпа в осенних пальто, брускетта и свеча на столе — HiDream отработал бриф дословно; Z-Image детали теряет.</td></tr>
<tr><td>Контекст сцены / фон</td><td class="win-hd">HiDream</td>
<td>Богатая среда: читаемые улицы, интерьеры, посетители ресторана. У Z-Image фон беднее и более «студийный».</td></tr>
<tr><td>Candid-аутентичность</td><td class="win-hd">HiDream</td>
<td>Позы и свет ближе к живому любительскому фото; Z-Image склонен к editorial-глянцу даже в кэжуал-сценах.</td></tr>
<tr><td>Консистентность персоны</td><td class="win-zi">Z-Image</td>
<td>Без char-LoRA лицо Z-Image держится узнаваемым между сценами; у HiDream персона плавает сильнее.</td></tr>
<tr><td>Скорость</td><td class="win-zi">Z-Image</td>
<td>~20 c против ~41 c на кадр (после прогрузки моделей).</td></tr>
</table>
<p class="notes"><b>Практический вывод:</b> это разные инструменты. Z-Image + финиш остаётся базой
«ленты» — быстрее, красивее лицо, готовая полировка кожи. HiDream full забирает сцены, где решает
композиция и точность брифа: сложные локации, реквизит, много объектов, толпа, текст. Плавающая
персона HiDream — не блокер: идентичность в любом случае фиксируется char-LoRA (Phase 6).
Стоит попробовать гибрид: композиция HiDream → финиш-стек Z-Image (2K + skin + FaceDetailer).</p>
</section>
</div>'''

out = "/tmp/claude-0/-workspace/9d8669da-c5e9-4ba0-934e-f8d362fd9f3d/scratchpad/faceoff.html"
os.makedirs(os.path.dirname(out), exist_ok=True)
open(out, "w").write(html)
print("written", out, f"{os.path.getsize(out)/1e6:.1f} MB")
