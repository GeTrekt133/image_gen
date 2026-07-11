"""Run the official Prompt-Refine (Gemma-4-31B finetune) locally on 5 short scene briefs."""
import json, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "HiDream-O1-Image"))
from prompt_agent_v2 import REWRITE_SYSTEM_PROMPT

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

BRIEFS = {
    "selfie": "instagram selfie of a young woman with honey-blonde wavy hair in a white ribbed tank top, cozy apartment, golden afternoon light",
    "gym": "candid photo of a young woman with honey-blonde hair in a dusty-rose sports bra at a modern gym, holding a water bottle",
    "street": "street style photo of a young woman with honey-blonde hair in an oversized camel blazer and jeans crossing a european street",
    "swimsuit": "golden hour photo of a young woman with honey-blonde hair in a sage-green one-piece swimsuit at an infinity pool by the sea",
    "restaurant": "evening photo of a young woman with honey-blonde hair in a black satin slip dress at a cozy upscale restaurant, wine and candlelight",
}

def main():
    mp = "models/prompt-refine"
    print("[refine] loading tokenizer/model…", flush=True)
    tok = AutoTokenizer.from_pretrained(mp)
    model = AutoModelForCausalLM.from_pretrained(mp, torch_dtype=torch.bfloat16, device_map="cuda").eval()
    out = {}
    for name, brief in BRIEFS.items():
        msgs = [{"role": "system", "content": REWRITE_SYSTEM_PROMPT},
                {"role": "user", "content": brief}]
        enc = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt",
                                      return_dict=True).to("cuda")
        t = time.time()
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=1024, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
        text = tok.decode(gen[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        out[name] = text
        print(f"[refine] {name} ({time.time()-t:.0f}s): {text[:140]}…", flush=True)
    json.dump(out, open("results/refined_prompts.json", "w"), ensure_ascii=False, indent=1)
    print("[refine] saved results/refined_prompts.json", flush=True)

if __name__ == "__main__":
    main()
