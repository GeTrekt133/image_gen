"""In-process batch for the FULL model: load once, generate cafe/beach x 5 seeds each."""
import os, sys, time
os.environ.setdefault("FA_VERSION", "auto")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "HiDream-O1-Image"))

import torch
from transformers import AutoProcessor
from models.qwen3_vl_transformers import Qwen3VLForConditionalGeneration
from models.pipeline import generate_image
from inference import add_special_tokens, get_tokenizer

MODEL = "models/hidream-o1-dev-2604" if "--dev" in sys.argv else "models/hidream-o1-full"

P_CAFE = "An ultra-photorealistic candid lifestyle photograph of a young woman in her mid-twenties with loosely pinned honey-blonde wavy hair, warm hazel eyes and naturally textured skin showing faint freckles and visible pores, sitting at a scuffed oak table beside a tall cafe window in soft directional morning light. She wears an oversized cream cable-knit sweater with chunky ribbed cuffs pushed up her forearms, both hands wrapped around a ceramic latte cup with delicate rosetta latte art, caught mid-laugh looking out the window. Composition is a waist-up shot from a slight three-quarter angle at eye level, subject placed on the right third of the frame, window and a blurred cobblestone street with pedestrians filling the left, a potted trailing plant and chalkboard menu softly out of focus in the background. Warm golden window light rakes across her face creating soft catchlights in her eyes and gentle shadow modelling on the knit texture, dust motes visible in the light shaft. Shot on a full-frame camera with an 85mm f/1.8 lens, shallow depth of field, crisp focus on her eyes and the cup rim, creamy bokeh, subtle Kodak Portra 400 film grain, true-to-life color grading with warm amber tones, authentic instagram lifestyle aesthetic."

P_BEACH = "An ultra-photorealistic golden-hour travel photograph of a young woman in her mid-twenties with sun-kissed tanned skin, wind-blown salty chestnut-blonde hair and a genuine open-mouthed laugh, walking barefoot along the wet shoreline of a tropical beach where clear turquoise water breaks into thin white foam around her ankles. She wears an unbuttoned oversized white linen shirt with rolled sleeves fluttering in the breeze over a terracotta floral-print triangle bikini, tiny water droplets and a light sheen of salt spray visible on her collarbone and stomach, a thin gold anklet on her left ankle. Full-body composition from a low frontal angle with her centered slightly left, footprints trailing behind her in the smooth wet sand, dense green palm trees and a rocky headland softly blurred in the background under scattered golden clouds. Late afternoon sun low on the horizon backlights her hair with a warm rim light while soft reflected light from the sand fills her face, glistening specular highlights on the water. Shot on a full-frame camera with a 50mm f/2 lens, fast shutter freezing the splashing foam, crisp focus on her face, natural skin texture with visible pores, subtle film grain, vibrant but true-to-life color grading, authentic travel influencer instagram aesthetic."

SEEDS = [101, 202, 303, 404, 505]

def main():
    print(f"[batch] loading {MODEL} (fp32)…", flush=True)
    t0 = time.time()
    processor = AutoProcessor.from_pretrained(MODEL)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL, torch_dtype=torch.float32, device_map="cuda").eval()
    tokenizer = get_tokenizer(processor)
    add_special_tokens(tokenizer)
    print(f"[batch] loaded in {time.time()-t0:.0f}s", flush=True)

    os.makedirs("results/batch10", exist_ok=True)
    jobs = [("cafe", P_CAFE), ("beach", P_BEACH)]
    for name, prompt in jobs:
        for seed in SEEDS:
            out = f"results/batch10/{name}_s{seed}.png"
            t = time.time()
            image = generate_image(
                model=model, processor=processor, prompt=prompt,
                ref_image_paths=[], height=2048, width=2048,
                num_inference_steps=50, guidance_scale=5.0, shift=3.0,
                timesteps_list=None, scheduler_name="default", seed=seed,
                keep_original_aspect=False, layout_bboxes=None)
            image.save(out)
            print(f"[batch] {out} in {time.time()-t:.0f}s", flush=True)
    print("[batch] all done", flush=True)

if __name__ == "__main__":
    main()
