"""Test: is the heavy background blur prompt-driven? Deep-focus rewrite of the cafe scene."""
import os, sys, time
os.environ.setdefault("FA_VERSION", "auto")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "HiDream-O1-Image"))

import torch
from transformers import AutoProcessor
from models.qwen3_vl_transformers import Qwen3VLForConditionalGeneration
from models.pipeline import generate_image
from inference import add_special_tokens, get_tokenizer

P_CAFE_DEEP = "An ultra-photorealistic candid lifestyle photograph of a young woman in her mid-twenties with loosely pinned honey-blonde wavy hair, warm hazel eyes and naturally textured skin showing faint freckles and visible pores, sitting at a scuffed oak table beside a tall cafe window in soft directional morning light. She wears an oversized cream cable-knit sweater with chunky ribbed cuffs pushed up her forearms, both hands wrapped around a ceramic latte cup with delicate rosetta latte art, caught mid-laugh looking out the window. Composition is a waist-up shot from a slight three-quarter angle at eye level, subject placed on the right third of the frame. Through the window on the left, a cobblestone street with pedestrians in autumn coats, parked bicycles and storefronts with painted wooden signs is rendered in sharp focus with clearly readable architectural detail; inside, a chalkboard menu with legible handwritten items and a potted trailing plant on a brick wall are also crisply in focus. Warm golden window light rakes across her face creating soft catchlights in her eyes and gentle shadow modelling on the knit texture. Shot on a full-frame camera with a 35mm lens stopped down to f/8, deep depth of field keeping both the subject and the entire background sharp from front to back, high edge-to-edge sharpness, subtle Kodak Portra 400 film grain, true-to-life color grading with warm amber tones, authentic documentary street-photography aesthetic."

def main():
    MODEL = "models/hidream-o1-full"
    print(f"[df] loading {MODEL}…", flush=True)
    processor = AutoProcessor.from_pretrained(MODEL)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL, torch_dtype=torch.float32, device_map="cuda").eval()
    tokenizer = get_tokenizer(processor)
    add_special_tokens(tokenizer)

    os.makedirs("results/deepfocus", exist_ok=True)
    runs = [
        ("cafe_deep_s101_cfg5", 101, 5.0),
        ("cafe_deep_s202_cfg5", 202, 5.0),
        ("cafe_deep_s101_cfg4", 101, 4.0),
    ]
    for name, seed, cfg in runs:
        t = time.time()
        image = generate_image(
            model=model, processor=processor, prompt=P_CAFE_DEEP,
            ref_image_paths=[], height=2048, width=2048,
            num_inference_steps=50, guidance_scale=cfg, shift=3.0,
            timesteps_list=None, scheduler_name="default", seed=seed,
            keep_original_aspect=False, layout_bboxes=None)
        out = f"results/deepfocus/{name}.png"
        image.save(out)
        print(f"[df] {out} in {time.time()-t:.0f}s", flush=True)
    print("[df] done", flush=True)

if __name__ == "__main__":
    main()
