# TRAINING — LTX-2.3 fine-tuning plan (dance motion + hands + persona)

> Context: `CLAUDE.md`. Base = `ltx-2.3-22b-dev-fp8`. GPU = RTX 6000 96GB.
> This doc is the plan for improving dance motion-transfer, hand robustness on fast frames,
> and persona identity via training. Sources at the bottom.

## Problem
6-finger / anatomy artifacts appear on **dynamic (fast-motion) frames**. Closed models (Kling/Veo/Seedance)
almost never do this — but that's **scale + no-distillation + higher resolution + curated data + RLHF +
an internal detect-and-fix product layer**, none of which we fully match on one open 22B. We can **narrow**
the gap with (a) targeted training and (b) more compute-per-frame on hero shots.

## Why train (not just a repair pass)
Per-frame hand-repair (detect→inpaint→re-encode) has **temporal flicker**; optical-flow warp over a wrong
hand often looks *worse*. Training shifts the base **prior** → more robust in-domain. (Repair stays a valid
*complement* for hero frames, and its automatable version — finger-count reward — feeds DPO, see below.)

## Three adapters, three jobs (don't conflate)
| Adapter | Job | Method |
|---|---|---|
| **char-LoRA** | persona identity (face/body) | LTX Trainer, likeness LoRA |
| **IC Pose-Control** | *pose adherence* for dance | **warm-start from union** (variant A) |
| **Base fine-tune** | *hand anatomy + dance style* | **LoKr (+PiSSA+rsLoRA)** or fallback |

**Key separation:** control ≠ hands. Pose-control improves *following the skeleton*; the 6-finger lives in
**base generation** (DiT + VAE). A pose/control adapter (any) will NOT fix hands — only a base FT does.

---

## Method choice — adapters vs full-FT (why LoKr)
Quality vs full-FT (numbers from LLM benches; video-gen = extrapolation, no direct study):

| Method | ≈ full-FT | Note |
|---|---|---|
| **GaLore** | ≈100% (it *is* full-param, low-mem) | slow (~13× vs LoRA), outputs a full model, tight on 96GB |
| **MoRA** | ~90-97% (square-matrix high-rank) | less battle-tested in diffusion |
| **LoKr** | ~90-95% for style/structure | proven in diffusion, cheap, mergeable ← **pick** |
| **DoRA** | ~90-98% but only at **low-mid rank**; ≈LoRA at high rank | edge washes out ≥ r128 |
| **rsLoRA** | ~85-95%, rises with rank | `α/√r` scaling *unlocks* high rank |
| **PiSSA** | > LoRA (SVD init) | init trick, stack on any low-rank |
| **OFT/BOFT** | = or > full-FT on *controllability/preservation*; < on big domain shift | rotates, minimal forgetting |

**Decision:** **LoKr + PiSSA-init + rsLoRA-scaling** for the base (≈90-95% of full-FT for style/structure,
cheap, mergeable, and on **small data often ≥ naive full-FT** because full-FT overfits/forgets — see
"Illusion of Equivalence"). Escalate to **GaLore full-FT** only if hands plateau.

**Two axes:** *adapter* (where params go) × *objective* (what you optimize). The genuinely-new lever for
hands is **DPO / reward fine-tuning** (Diffusion-DPO/D3PO fix hands in images; DenseDPO/Diffusion-APO/DRF for
video 2025-26). For hands the reward is **semi-automatable** (hand-keypoint finger-count → win/lose pairs) —
deferred, but it's the strongest anatomy lever and worth a second pass.

---

## Data

### The fast-motion-blur insight (important)
**Motion-blurred real hands are GOOD training targets** — real people have 5 fingers even when blurred, so the
model learns "fast hand = coherent blur with correct topology" → stops hallucinating a 6th finger in motion.
Do **NOT** filter by sharpness. Filter by **"is this a real, temporally-recoverable hand"**:
- ✅ keep: sharp 5-finger AND coherently-blurred real hands. **Oversample fast-motion** (the target regime).
- ❌ drop only: indistinct blobs / self-occluded mush / cut-off, **compression garbage**, and 🔴
  **AI-generated or heavily-AI-filtered clips** (common on IG/TikTok — they carry the artifact → poison).

### Funnel (one ingest → two subsets)
```
ingest ~5-15k dance clips (dance classifier, highest-res)
  → TransNetV2 shot-split (no splices)
  → common: single-dancer · full-body · no-subtitle · native-fps · dedup(pHash)
  → MOTION-POOL (IC Pose):   + motion/angle diversity (hands any-quality)
  │     → DWPose → pairs (pose-guide → clip)
  → REAL-HAND-POOL (LoKr):   + AI-video drop · hand-prominence · highest-res · keep real blur
        → VLM captions (appearance, NO motion words)
  → both: LTX Trainer preprocess (cache latents+embeddings), 8n+1 windows, native fps
```
Reuses `prep_drive_new.py` (shots/subtitles/fps/8n+1/masks) — extend with AI-filter + hand-recoverability
check + captions + pose-pairing.

### Amounts
| Training | Keep | Ingest | ~hrs video |
|---|---|---|---|
| **IC Pose LoRA** (warm-start) | 500-1500 pairs (motion diversity) | ~2-5k | ~2-4 |
| **LoKr base** (hands/style) | 1000-3000 (real hands, 30-50% hand-prominent) | ~5-15k | ~4-8 |

**Start MVP ~600 (300 motion + 300 real-hand) → train both → eval → scale.** Don't scrape 15k before a signal.

### Tooling
`yt-dlp` / `gallery-dl` (`-f bestvideo`), dance-hashtags + creators (high hit-rate), pHash dedup,
AI-video filter (AI-tool watermarks, over-smooth skin, temporal artifacts).

### Legal / ethics (per CLAUDE.md)
Use scraped real video only for **abstract pose (skeleton) + SFW clothed-dance** (motion + hand lesson) —
not identity. **NSFW + persona identity are SEPARATE inference-time layers** (char-LoRA + NSFW checkpoint),
never in training data. No NSFW training on real people.

---

## IC Pose-Control — warm-start (Variant A) ✅ supported
ltx-trainer has **`load_checkpoint`** (ModelConfig): *"path to resume training from a checkpoint file or
directory."* → point it at the union IC-LoRA and continue. **Why A over B:** you don't need union's
canny/depth/general-pose generality, so directly refining its weights specializes *tighter* to dance.
(Its depth/canny modes will erode — fine; the NSFW-still depth-CN is a *different* image model, unaffected.)

**Verify on pod (2 things):** (1) your `rank` + `target_modules` **match union's** (read them from the
union `.safetensors`); (2) `load_checkpoint` accepts an **external** adapter as init (not only its own
checkpoints) — if not, fallback **B** = merge union into base, train a residual LoRA.

**🔴 The gotcha that kills A:** too-high LR **wipes** the warm-start → becomes from-scratch (weak on 500 clips).
Use **low LR + early-stop**. Eval vs union baseline (adherence ↑, no new artifacts).

**Starter config** (verify exact YAML nesting vs shipped `configs/*.yaml`):
```yaml
model:
  load_checkpoint: .../loras/ltxv/ltx2/ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors
lora:
  rank: <= union's rank            # must match union
  alpha: <= rank
  target_modules: [<union's>]      # e.g. attn1.to_q/to_k/to_v/to_out.0
  dropout: 0.0
optimization:
  learning_rate: 2.0e-5            # LOW — refine, not relearn
  steps: 800                       # start ~500, early-stop on eval
  batch_size: 1
  optimizer_type: adamw8bit
  scheduler_type: constant
training_strategy:
  name: flexible
  conditions:
    - type: reference              # pose-guide = IC-LoRA reference
      downscale_factor: 2          # =ref0.5 like union (1 = tighter, but shifts distribution)
      temporal_scale_factor: 1
dataset:
  latents_dir: /workspace/data/dance_pairs/latents
```
**Skip this entirely if union's pose adherence is already good enough** — then put the whole data budget into
LoKr (hands are the real pain, not adherence).

---

## Base fine-tune — LoKr (hands + dance style)
**🔴 ltx-trainer does NOT do LoKr** — it supports LoRA / full-FT / IC-LoRA only (LoRA config: `rank` 8-128,
`alpha`, `dropout`, `target_modules`; optim `learning_rate` 1e-5..1e-3, `optimizer_type` adamw/adamw8bit).
So for LoKr:
1. **Preferred:** **ai-toolkit** (has LTX + LyCORIS/LoKr) — verify it supports **LTX-2.3-22b LoKr**.
2. **Fallback if no LoKr anywhere:** high-rank **LoRA/DoRA** (rank ≤128 — ltx-trainer caps 128) **+ rsLoRA
   scaling + PiSSA init**, or **full-FT (GaLore-style)** via ltx-trainer's full-FT mode.
3. **If mechanisms truly absent → hand-write LoKr** (operator's call).

Rank note: **ltx-trainer caps LoRA rank ~128** (revise earlier "128-256" down). "High rank" on LTX = ≤128.
Targets for a base FT: attention **+ MLP/ff** modules (not just attn — hands/anatomy need the feed-forward too).
LR: low (refinement risk of forgetting base video priors). PiSSA init + rsLoRA scaling to make the rank count.

---

## Eval (decide when to add data)
- **Hands:** % of fast-motion frames with a correct 5-finger hand (auto: hand-keypoint detector + finger-count
  / anatomy heuristic). This is also the DPO reward signal.
- **Pose adherence:** MPJPE(output pose vs driving pose).
- **Identity:** face-embedding consistency across frames (for char-LoRA).
- Always compare **vs the union/base baseline**. Plateau → add data; regression → LR too high / overtrained.

## Compute (RTX 6000 96GB)
- IC-LoRA warm-start + LoRA/DoRA/LoKr (rank ≤128): fits (base fp8/bf16 + adamw8bit + activation checkpointing).
- Full-FT 22B: **not** single-card → GaLore (tight, slow) or multi-GPU (FSDP/ZeRO).

## Open items / next steps
1. **Inspect union `.safetensors`** → read `rank` + `target_modules` (to match in Variant A).
2. **Verify** `load_checkpoint` accepts the external union adapter (A) — else fallback B.
3. **Verify ai-toolkit LoKr for LTX-2.3** — else fallback (high-rank DoRA+rsLoRA / full-FT) or hand-write LoKr.
4. Build **MVP data ~600** → first warm-start pose LoRA + base LoKr → **eval on hand metric** → scale.
5. Later: **DPO/reward pass** on finger-count win/lose pairs (strongest anatomy lever, semi-automatable).

## ✅ Verification (deep-research, 2026-07-10, primary sources)

**CONFIRMED by Lightricks docs / arXiv:**
- **Infra (Claim 1)** — ltx-trainer LoRA/full/IC-LoRA + all cited keys. Concrete **defaults**: `rank 32`,
  `alpha 32`, `learning_rate 1e-4`, adamw/adamw8bit, targets `to_k/to_q/to_v/to_out.0`. IC-LoRA pairs:
  reference latents concatenated, **clean (timestep=0), no noise, excluded from loss**. `load_checkpoint` exists.
- **Control landscape (Claim 2)** — no 22b pose-only; only Union (Canny+Depth+Pose, ref0.5, ~654MB);
  dedicated pose only on 19b; 22b also has **Motion-Track-Control**.
- **DPO fixes hands (Claim 7)** — **D3PO** (arXiv 2311.13231, CVPR 2024, code available) reduces hand
  deformity + "correct number of fingers" **without a reward model**; DenseDPO fixes video-DPO low-motion bias.

**🔴 CRITICAL — we missed:** IC-LoRA control INFERENCE runs **only on the distilled 22b checkpoint**
(`ltx-2.3-22b-distilled.safetensors`), NOT dev — official: *"Do not use it with the dev checkpoint"*
(ltx.io/blog/using-lora-adapters, `ic_lora.py`). Our pipeline is on **`dev-fp8`**; in ComfyUI dev+union runs,
but the *official* control path is distilled → the distilled (lower-detail) model is what hurts hands on
controlled generation. **Verify dev+union isn't degraded; the sanctioned control path is distilled.**

**Corrections:**
- **Full-FT 22B = 4-8× H100 + FSDP**, NOT one 96GB → **GaLore-full-FT-on-one-card is OFF**; 96GB = PEFT/LoRA
  family only (32GB min / 80GB rec). Confirms LoKr/LoRA is the path.
- **Rank NOT hard-capped at 128** — 8-128 is a "typical range", default 32; higher is allowed (not enforced).
- **6-finger ≠ specifically caused by distillation** — that causal claim was REFUTED (bad source). It's
  base-generation + resolution + fast-motion (distilled control path likely contributes, not proven as THE cause).

**🆕 New lever — HandCraft** (WACV 2025, arXiv 2411.04332): training-free post-hoc hand restoration —
YOLOv8 hand-detect → auto-mask → **MANO 3D-hand depth-conditioning → ControlNet inpaint**. Better than naive
Qwen-Edit (MANO = correct 3D-hand prior). Lineage: HandRefiner'23 → HandCraft'24 → 3D-mesh-guided (2506.12680)'25.
**IMAGE method → per-frame on video = temporal-flicker risk**; complements, doesn't replace base-FT.

**🟡 STILL UNVERIFIED (our reasoning — treat as hypotheses to test on pod):**
- Warm-start (Variant A) low-LR specialization of union without forgetting (`load_checkpoint` exists; external-
  adapter-init + no-forgetting unconfirmed).
- "Motion-blur = good data" + data amounts (500-1500 / 1000-3000) — plausible, no source.
- 🔴 **LoKr/LyCORIS for LTX-2.3 (ai-toolkit) — STILL OPEN** (unconfirmed). PiSSA/rsLoRA for LTX unconfirmed.
- No community LTX-2.3 dance/hand LoRA or published hyperparams found (beyond the defaults above).

## Sources
- LTX Trainer docs: `configuration-reference.md`, `training-guide.md`, `dataset-preparation.md`, `training-modes.md`
  (github.com/Lightricks/LTX-2, packages/ltx-trainer).
- DoRA — arXiv 2402.09353 (rank ablation: gap huge at low rank, ~+2-3 at r32-64, washes out higher; half-rank effect).
- GaLore / Q-GaLore (≈full-FT, <0.65 gap, ~13× slower). LoRA-vs-FT "Illusion of Equivalence" — arXiv 2410.21228.
- LoKr/OFT/BOFT/PiSSA/rsLoRA/MoRA — PEFT surveys 2026.
- DPO for diffusion: Diffusion-DPO 2311.12908, D3PO (hands, no reward model); video: Diffusion-APO 2605.07503,
  DenseDPO 2506.03517, Diffusion-DRF 2601.04153.
