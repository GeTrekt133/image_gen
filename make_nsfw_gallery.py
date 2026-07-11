"""Local self-contained NSFW gallery: prompts + adult-anchored Z-Image images (data URIs)."""
import base64, io, os
from PIL import Image

PERSONA = ("a 32-year-old mature adult woman, clearly adult grown woman with a full curvy womanly figure, "
           "wide hips, full large bust, thick thighs, mature adult facial features with defined cheekbones "
           "and laugh lines, confident grown-woman presence, long honey-blonde wavy hair, hazel eyes, "
           "natural skin texture with pores and freckles")

SCENES = [
    ("boudoir_topless", "Boudoir · topless",
     "boudoir photo of a topless {P}, sitting on the edge of a bed in black lace panties, bare breasts, soft window light, silk sheets, photorealistic, sharp focus"),
    ("lingerie_mirror", "Lingerie · mirror",
     "sensual photo of {P} in a black lace lingerie set, standing by a full-length mirror in a dim bedroom, string lights, confident gaze over her shoulder, photorealistic, high detail"),
    ("nude_window", "Nude · window",
     "artistic full nude photo of {P}, standing by a tall window in morning light, fully nude, full mature curves, side-lit body with soft shadows, tasteful pose, photorealistic, sharp focus"),
    ("shower_nude", "Nude · shower",
     "photo of a fully nude {P} in a walk-in shower, wet skin with water droplets, wet hair slicked back, steam, hands in her hair, photorealistic, high detail"),
    ("bed_explicit", "Explicit · bed",
     "explicit photo of a fully nude {P} reclining on a bed, full mature curvy body, bare breasts, relaxed confident expression, warm bedside lamp light, silk sheets, photorealistic, sharp focus"),
    ("ass_back", "Nude · from behind",
     "photo of a fully nude {P} kneeling on a bed facing away, view from behind, full round hips and ass, looking back over her shoulder, golden hour light, photorealistic, high detail"),
]

def b64(path, w=760, q=82):
    im = Image.open(path).convert("RGB")
    im.thumbnail((w, w), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=q, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()

cards = []
for key, title, tpl in SCENES:
    img = b64(f"results/nsfw_adult/{key}.png")
    full = tpl.replace("{P}", PERSONA)
    cards.append(f'''
<article class="card">
  <div class="imgwrap"><img src="{img}" alt="{title}" loading="lazy"></div>
  <div class="body">
    <h2>{title}</h2>
    <p class="fname">results/nsfw_adult/{key}.png</p>
    <details><summary>Промпт</summary><p class="prompt">{full}</p></details>
  </div>
</article>''')

html = f'''<!doctype html><html lang="ru"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>NSFW тест · Z-Image Turbo (взрослое заякоривание)</title>
<style>
:root{{--bg:#141312;--panel:#1e1c1a;--line:#302c28;--ink:#ede8e0;--mut:#9a938a;--acc:#c98a5a;}}
@media(prefers-color-scheme:light){{:root{{--bg:#f6f3ee;--panel:#fff;--line:#e4ded4;--ink:#26231f;--mut:#6c665c;}}}}
:root[data-theme="light"]{{--bg:#f6f3ee;--panel:#fff;--line:#e4ded4;--ink:#26231f;--mut:#6c665c;}}
:root[data-theme="dark"]{{--bg:#141312;--panel:#1e1c1a;--line:#302c28;--ink:#ede8e0;--mut:#9a938a;}}
*{{box-sizing:border-box}}html{{background:var(--bg)}}
body{{margin:0;background:var(--bg);color:var(--ink);font:16px/1.55 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}}
.wrap{{max-width:1200px;margin:0 auto;padding:40px 22px 80px}}
h1{{font-family:Georgia,serif;font-weight:600;font-size:clamp(23px,3.4vw,34px);margin:0 0 8px;text-wrap:balance}}
.lede{{color:var(--mut);max-width:80ch;margin:0 0 6px}}
.spec{{display:flex;flex-wrap:wrap;gap:9px;margin:18px 0 4px}}
.spec div{{background:var(--panel);border:1px solid var(--line);border-radius:6px;padding:7px 13px;font-size:13px;color:var(--mut)}}
.spec b{{color:var(--ink)}}
.persona{{margin:20px 0 0;background:var(--panel);border:1px solid var(--line);border-left:3px solid var(--acc);border-radius:8px;padding:14px 18px;font-size:14px;color:var(--mut)}}
.persona b{{color:var(--acc)}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(320px,1fr));gap:18px;margin-top:30px}}
.card{{background:var(--panel);border:1px solid var(--line);border-radius:10px;overflow:hidden;display:flex;flex-direction:column}}
.imgwrap{{background:#000;aspect-ratio:1/1;overflow:hidden}}
.imgwrap img{{width:100%;height:100%;object-fit:cover;display:block}}
.body{{padding:14px 16px 16px}}
.body h2{{font-family:Georgia,serif;font-size:18px;margin:0 0 4px}}
.fname{{color:var(--mut);font-size:12px;font-family:ui-monospace,monospace;margin:0 0 10px}}
details summary{{cursor:pointer;font-size:13px;color:var(--acc);font-weight:600}}
.prompt{{color:var(--mut);font-size:13px;line-height:1.5;margin:8px 0 0}}
</style></head><body>
<div class="wrap">
<h1>NSFW-тест · Z-Image Turbo с взрослым заякориванием</h1>
<p class="lede">Синтетическая 18+ персона (см. CLAUDE.md). Сгенерировано на чистом Z-Image Turbo bf16
(не unStable) с усиленным взрослым заякориванием в позитиве — зрелые черты + оформившаяся фигура.
Финиш: 2K → skin-refine dn0.35 → FaceDetailer.</p>
<div class="spec">
  <div>База: <b>z_image_turbo_bf16</b></div>
  <div>Семплинг: <b>8 шагов · cfg 1 · res_multistep · shift 3</b></div>
  <div>Сид: <b>21</b></div>
  <div>Финиш: <b>2K → skin dn0.35 → FaceDetailer</b></div>
  <div>~<b>20 c/кадр</b></div>
</div>
<div class="persona"><b>Персона (общий префикс промптов):</b><br>{PERSONA}</div>
<div class="grid">{''.join(cards)}</div>
</div>
<script>
// respect stored theme toggle if the host stamps data-theme; no-op otherwise
</script>
</body></html>'''

out = "/workspace/image_gen/results/nsfw_adult/gallery.html"
open(out, "w").write(html)
print("written", out, f"{os.path.getsize(out)/1e6:.1f} MB")
