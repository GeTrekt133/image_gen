"""Generate 5 scenes with OFFICIAL refined prompts on full and dev-2604."""
import json, os, sys, time
os.environ.setdefault("FA_VERSION", "auto")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "HiDream-O1-Image"))

import torch
from transformers import AutoProcessor
from models.qwen3_vl_transformers import Qwen3VLForConditionalGeneration
from models.pipeline import generate_image, DEFAULT_TIMESTEPS
from inference import add_special_tokens, get_tokenizer

PROMPTS = json.load(open("results/refined_prompts.json"))
SEED = 7

CONFIGS = [
    # (tag, model_path, kwargs for generate_image)
    ("full_ref", "models/hidream-o1-full",
     dict(num_inference_steps=50, guidance_scale=5.0, shift=3.0,
          timesteps_list=None, scheduler_name="default")),
    ("dev_ref", "models/hidream-o1-dev-2604",
     dict(num_inference_steps=28, guidance_scale=0.0, shift=1.0,
          timesteps_list=DEFAULT_TIMESTEPS, scheduler_name="flash",
          noise_scale_start=8.0, noise_scale_end=8.0, noise_clip_std=8.0)),
]

def main():
    os.makedirs("results/faceoff", exist_ok=True)
    for tag, mp, kw in CONFIGS:
        print(f"[ref] loading {mp}…", flush=True)
        processor = AutoProcessor.from_pretrained(mp)
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            mp, torch_dtype=torch.float32, device_map="cuda").eval()
        add_special_tokens(get_tokenizer(processor))
        for name, prompt in PROMPTS.items():
            t = time.time()
            image = generate_image(
                model=model, processor=processor, prompt=prompt,
                ref_image_paths=[], height=2048, width=2048, seed=SEED,
                keep_original_aspect=False, layout_bboxes=None, **kw)
            out = f"results/faceoff/hidream_{tag}_{name}.png"
            image.save(out)
            print(f"[ref] {out} in {time.time()-t:.0f}s", flush=True)
        del model
        torch.cuda.empty_cache()
    print("[ref] done", flush=True)

if __name__ == "__main__":
    main()
