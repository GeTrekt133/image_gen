"""Self-contained gallery for results/nsfw (unStable Revolution batch): prompts + images as data URIs.
Output: results/nsfw/gallery.html — grid + lightbox + structured prompt breakdown."""
import base64, html as H, io, os
from PIL import Image

PERSONA = ("beautiful young woman with long honey-blonde wavy hair, hazel eyes, "
           "natural skin texture with visible pores and faint freckles")

# (key, title_ru, shot, scene_after_persona)
SCENES = [
    ("boudoir_topless", "Будуар · топлес", "boudoir photo of a topless {P}",
     "sitting on the edge of a bed in black lace panties, bare breasts, soft window light across her body, silk sheets, warm intimate atmosphere, photorealistic, sharp focus"),
    ("lingerie_mirror", "Бельё · зеркало", "sensual photo of a {P}",
     "in a sheer black lace lingerie set, standing by a full-length mirror in a dim bedroom, string lights, seductive gaze over her shoulder, photorealistic, high detail"),
    ("nude_window", "Ню · окно", "artistic full nude photo of a {P}",
     "standing by a tall window in morning light, fully nude, natural breasts and curves, side-lit body with soft shadows, tasteful sensual pose, photorealistic, sharp focus"),
    ("shower_nude", "Ню · душ", "photo of a fully nude {P}",
     "in a walk-in shower, wet skin with water droplets and streams, wet hair slicked back, steam, glass wall with condensation, hands in her hair, photorealistic, high detail"),
    ("bed_explicit", "Эксплицит · кровать", "explicit photo of a fully nude {P}",
     "lying on a bed with legs parted, nude spread pose, visible vulva, bare breasts, relaxed sensual expression, warm bedside lamp light, silk sheets, photorealistic, sharp focus"),
    ("ass_back", "Ню · со спины", "photo of a fully nude {P}",
     "kneeling on a bed facing away from camera, view from behind, bare back and round ass, looking back over her shoulder with a teasing smile, golden hour window light, photorealistic, high detail"),
]

DIR = "results/nsfw"


def b64(path, w, q):
    im = Image.open(path).convert("RGB")
    im.thumbnail((w, w), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=q, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


cards, boxes = [], []
for i, (key, title, shot, scene) in enumerate(SCENES):
    thumb = b64(f"{DIR}/{key}.png", 720, 82)
    big = b64(f"{DIR}/{key}.png", 1600, 87)
    full = f"{shot.replace('{P}', PERSONA)}, {scene}"
    tags = "".join(f"<span>{H.escape(t.strip())}</span>" for t in scene.split(","))
    shot_html = H.escape(shot).replace("{P}", f'<em class="persona-tok">персона</em>')
    cards.append(f'''
<article class="card">
  <button class="imgwrap" onclick="openbox({i})" aria-label="Открыть {H.escape(title)}">
    <img src="{thumb}" alt="{H.escape(title)}" loading="lazy">
    <span class="zoom">⤢</span>
  </button>
  <div class="body">
    <div class="head">
      <h2>{title}</h2>
      <button class="copy" onclick="copyPrompt(this)" data-p="{H.escape(full, quote=True)}">копировать промпт</button>
    </div>
    <p class="fname">{DIR}/{key}.png · 2048×2048</p>
    <p class="shot">{shot_html}</p>
    <div class="tags">{tags}</div>
  </div>
</article>''')
    boxes.append(f'''
<div class="slide" id="slide{i}" hidden>
  <img src="{big}" alt="{H.escape(title)}">
  <p class="cap"><b>{title}</b> — {H.escape(full)}</p>
</div>''')

n = len(SCENES)
html = f'''<!doctype html><html lang="ru"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>NSFW-батч · unStable Revolution</title>
<style>
:root{{--bg:#131110;--panel:#1d1a18;--line:#332d28;--ink:#efe9e1;--mut:#a2988c;--acc:#d4915c;--acc2:#8f6b4a}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--bg);color:var(--ink);font:16px/1.55 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}}
.wrap{{max-width:1240px;margin:0 auto;padding:44px 24px 90px}}
header h1{{font-family:Georgia,'Times New Roman',serif;font-weight:600;font-size:clamp(24px,3.6vw,38px);margin:0 0 10px;letter-spacing:.01em;text-wrap:balance}}
header h1 b{{color:var(--acc);font-weight:600}}
.lede{{color:var(--mut);max-width:78ch;margin:0}}
.spec{{display:flex;flex-wrap:wrap;gap:9px;margin:22px 0 0}}
.spec div{{background:var(--panel);border:1px solid var(--line);border-radius:7px;padding:7px 14px;font-size:13px;color:var(--mut)}}
.spec b{{color:var(--ink);font-weight:600}}
.persona{{margin:22px 0 0;background:linear-gradient(135deg,var(--panel),#211c17);border:1px solid var(--line);border-left:3px solid var(--acc);border-radius:9px;padding:15px 19px;font-size:14px;color:var(--mut)}}
.persona b{{color:var(--acc)}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(330px,1fr));gap:20px;margin-top:34px}}
.card{{background:var(--panel);border:1px solid var(--line);border-radius:12px;overflow:hidden;display:flex;flex-direction:column;transition:border-color .15s}}
.card:hover{{border-color:var(--acc2)}}
.imgwrap{{position:relative;display:block;width:100%;padding:0;border:0;background:#000;aspect-ratio:1/1;overflow:hidden;cursor:zoom-in}}
.imgwrap img{{width:100%;height:100%;object-fit:cover;display:block;transition:transform .35s ease}}
.imgwrap:hover img{{transform:scale(1.035)}}
.zoom{{position:absolute;right:10px;bottom:10px;background:rgba(0,0,0,.55);color:#fff;border-radius:6px;padding:3px 9px;font-size:15px;opacity:0;transition:opacity .2s}}
.imgwrap:hover .zoom{{opacity:1}}
.body{{padding:15px 17px 17px}}
.head{{display:flex;align-items:baseline;justify-content:space-between;gap:10px}}
.body h2{{font-family:Georgia,serif;font-size:19px;font-weight:600;margin:0}}
.copy{{flex:none;background:none;border:1px solid var(--line);color:var(--acc);border-radius:6px;padding:3px 10px;font-size:11.5px;cursor:pointer}}
.copy:hover{{border-color:var(--acc)}}
.fname{{color:var(--mut);font-size:11.5px;font-family:ui-monospace,SFMono-Regular,monospace;margin:5px 0 12px}}
.shot{{font-size:13.5px;color:var(--ink);margin:0 0 9px;font-style:italic}}
.persona-tok{{font-style:normal;background:rgba(212,145,92,.16);color:var(--acc);border-radius:4px;padding:0 6px;font-size:12.5px}}
.tags{{display:flex;flex-wrap:wrap;gap:6px}}
.tags span{{background:var(--bg);border:1px solid var(--line);color:var(--mut);border-radius:99px;padding:2.5px 10px;font-size:12px}}
dialog{{border:0;padding:0;background:transparent;max-width:none;width:100%;height:100%}}
dialog::backdrop{{background:rgba(8,6,5,.93)}}
.lb{{position:fixed;inset:0;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:12px;padding:34px 70px}}
.slide{{display:flex;flex-direction:column;align-items:center;gap:12px;max-width:min(92vw,900px)}}
.slide img{{max-width:100%;max-height:78vh;border-radius:8px;box-shadow:0 20px 60px rgba(0,0,0,.6)}}
.cap{{color:var(--mut);font-size:12.5px;line-height:1.5;margin:0;max-width:90ch;text-align:center}}
.cap b{{color:var(--ink)}}
.nav{{position:fixed;top:50%;transform:translateY(-50%);background:rgba(255,255,255,.07);border:0;color:#fff;font-size:26px;width:46px;height:64px;border-radius:9px;cursor:pointer}}
.nav:hover{{background:rgba(255,255,255,.16)}}
#prev{{left:12px}}#next{{right:12px}}
#close{{position:fixed;top:14px;right:16px;background:none;border:0;color:#bbb;font-size:30px;cursor:pointer}}
footer{{margin-top:44px;color:var(--mut);font-size:12.5px;border-top:1px solid var(--line);padding-top:16px}}
</style></head><body>
<div class="wrap">
<header>
<h1>NSFW-батч · <b>unStable Revolution</b> + универсальный финиш</h1>
<p class="lede">Шесть сцен по одной синтетической 18+ персоне (см. CLAUDE.md), общий сид — проверка
консистентности лица и качества кожи по всему спектру: будуар → ню → эксплицит.
Пайплайн по PIPELINE_NSFW.md: база unStable Revolution → 2K-апскейл → skin-refine → FaceDetailer.</p>
<div class="spec">
  <div>Чекпоинт: <b>unStable Revolution</b></div>
  <div>Финиш: <b>2K → skin-refine → FaceDetailer</b></div>
  <div>Сид: <b>21</b> (общий)</div>
  <div>Выход: <b>2048×2048 PNG</b></div>
  <div>Сцен: <b>{n}</b></div>
</div>
<div class="persona"><b>Персона — общий префикс всех промптов:</b><br>{H.escape(PERSONA)}</div>
</header>
<div class="grid">{''.join(cards)}</div>
<footer>Файлы: {DIR}/&lt;scene&gt;.png · контрольная сетка: {DIR}/grid_check.jpg · скрипт генерации: nsfw_batch.py</footer>
</div>
<dialog id="box"><div class="lb">
<button id="close" onclick="box.close()" aria-label="Закрыть">✕</button>
<button class="nav" id="prev" onclick="step(-1)" aria-label="Назад">‹</button>
{''.join(boxes)}
<button class="nav" id="next" onclick="step(1)" aria-label="Вперёд">›</button>
</div></dialog>
<script>
const box=document.getElementById('box');let cur=0,N={n};
function show(i){{cur=(i+N)%N;for(let k=0;k<N;k++)document.getElementById('slide'+k).hidden=k!==cur;}}
function openbox(i){{show(i);box.showModal();}}
function step(d){{show(cur+d);}}
box.addEventListener('click',e=>{{if(e.target===box)box.close();}});
document.addEventListener('keydown',e=>{{if(!box.open)return;if(e.key==='ArrowRight')step(1);if(e.key==='ArrowLeft')step(-1);}});
function copyPrompt(btn){{navigator.clipboard.writeText(btn.dataset.p).then(()=>{{const t=btn.textContent;btn.textContent='скопировано ✓';setTimeout(()=>btn.textContent=t,1200);}});}}
</script>
</body></html>'''

out = os.path.join(DIR, "gallery.html")
open(out, "w").write(html)
print("written", out, f"{os.path.getsize(out)/1e6:.1f} MB")
