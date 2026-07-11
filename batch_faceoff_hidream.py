"""Face-off: 5 insta scenes on HiDream FULL (fp32, 50 steps, CFG 5)."""
import os, sys, time
os.environ.setdefault("FA_VERSION", "auto")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "HiDream-O1-Image"))

import torch
from transformers import AutoProcessor
from models.qwen3_vl_transformers import Qwen3VLForConditionalGeneration
from models.pipeline import generate_image
from inference import add_special_tokens, get_tokenizer

PERSONA = "a young woman in her mid-twenties with long honey-blonde wavy hair, warm hazel eyes and naturally textured skin with faint freckles and visible pores"

SCENES = {
    "selfie": f"An ultra-photorealistic casual handheld selfie of {PERSONA}, holding the phone slightly above eye level with her right arm partially visible at the frame edge, genuine relaxed smile with slightly squinted eyes, golden late-afternoon sunlight from a window to her left creating soft catchlights. She wears a simple white ribbed tank top and a thin gold necklace. Behind her, a cozy apartment living room with a beige linen sofa, a leafy potted monstera and framed posters on a warm white wall, everything moderately sharp and recognizable. Slight wide-angle phone-lens perspective with mild distortion at the edges, natural handheld framing tilted a few degrees, realistic phone-camera look with natural skin texture, subtle noise in the shadows, true-to-life colors, authentic instagram selfie aesthetic.",
    "gym": f"An ultra-photorealistic candid fitness photo of {PERSONA}, hair in a high ponytail with loose strands, standing in a modern gym between exercises, wearing a dusty-rose sports bra and high-waisted black leggings, a light genuine sheen of sweat on her collarbone and stomach, holding a stainless steel water bottle in one hand with a white towel over her shoulder. Three-quarter body composition from eye level, she stands slightly right of center; behind her rows of dumbbells on a rack, a cable machine and large windows with daylight are clearly visible and moderately sharp. Bright even gym lighting mixed with window daylight, crisp focus on her face and torso, defined but natural muscle tone, realistic fabric texture with slight compression wrinkles, authentic fitness influencer instagram aesthetic, shot on a full-frame camera with a 35mm lens at f/5.6.",
    "street": f"An ultra-photorealistic street style photograph of {PERSONA}, walking towards the camera across a European city crosswalk mid-stride, wearing an oversized camel-colored blazer over a white t-shirt, straight-leg blue jeans and white leather sneakers, a small black shoulder bag, hair lifted slightly by wind. Full-body composition from a low eye level, she is centered; behind her a row of old stone buildings with shops, parked bicycles and a few pedestrians in autumn coats, all rendered sharp with readable architectural detail under a bright overcast sky. Soft diffused daylight, natural motion in her hair and blazer, crisp focus from front to back with deep depth of field at f/8 on a 35mm lens, subtle film grain, true-to-life muted color grading, authentic candid editorial instagram aesthetic.",
    "swimsuit": f"An ultra-photorealistic golden-hour photo of {PERSONA} with slightly damp beach-wavy hair, standing waist-deep at the edge of an infinity pool overlooking the sea, wearing a matte sage-green one-piece swimsuit with a modest scooped neckline, water droplets on her shoulders and arms, laughing naturally with her head tilted back slightly. Composition from chest height a few meters away, she is on the left third; the pool edge, turquoise sea, distant coastline and a warm sunset sky with scattered clouds fill the rest of the frame in sharp focus. Low warm sun creates a golden rim light on her hair and shoulders with soft reflected fill from the water, glistening specular highlights, crisp focus on her face, natural wet-skin texture, subtle film grain, vibrant but true-to-life colors, authentic travel influencer instagram aesthetic, 50mm lens at f/5.6.",
    "restaurant": f"An ultra-photorealistic candid evening photo of {PERSONA} with softly curled hair, sitting at a table in an upscale but cozy restaurant, wearing an elegant black satin slip dress with thin straps and small gold earrings, leaning slightly forward with her chin resting on one hand, warm genuine smile looking just off-camera. On the white tablecloth in front of her a glass of red wine, a lit candle in a glass holder and a small plate of bruschetta are clearly visible and sharp. Behind her the restaurant interior with warm pendant lights, a bar with glowing bottle shelves and a few other guests remains recognizable and only gently softened. Warm candlelight and tungsten ambience modelling her face with soft flattering shadows, crisp focus on her eyes, natural skin texture, subtle film grain, rich warm color grading, authentic instagram dinner-date aesthetic, 50mm lens at f/4.",
}

SEED = 7

def main():
    MODEL = "models/hidream-o1-full"
    print(f"[faceoff] loading {MODEL}…", flush=True)
    processor = AutoProcessor.from_pretrained(MODEL)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL, torch_dtype=torch.float32, device_map="cuda").eval()
    tokenizer = get_tokenizer(processor)
    add_special_tokens(tokenizer)

    os.makedirs("results/faceoff", exist_ok=True)
    for name, prompt in SCENES.items():
        t = time.time()
        image = generate_image(
            model=model, processor=processor, prompt=prompt,
            ref_image_paths=[], height=2048, width=2048,
            num_inference_steps=50, guidance_scale=5.0, shift=3.0,
            timesteps_list=None, scheduler_name="default", seed=SEED,
            keep_original_aspect=False, layout_bboxes=None)
        out = f"results/faceoff/hidream_{name}.png"
        image.save(out)
        print(f"[faceoff] {out} in {time.time()-t:.0f}s", flush=True)
    print("[faceoff] HiDream side done", flush=True)

if __name__ == "__main__":
    main()
