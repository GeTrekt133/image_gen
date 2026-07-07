# Workflows — pose-control + edit per base (July 2026)

All three bases HAVE trained ControlNets — **reuse the official templates** (ship with ComfyUI
`Browse Templates`, or in the model HF repos), don't hand-write JSON. Base t2i workflows
(`wf_zimage.json`, `wf_flux2.json`, `wf_qwen.json`) are already verified against the official
templates (same model files). Below: the exact ControlNet model + loader gotcha + template per base.

## ControlNet models (all support pose via a DWPose preprocessor, strength 0.8–1.0)

| Base | ControlNet model (HF) | Loader node → folder | ⚠️ gotcha |
|---|---|---|---|
| **Qwen-Image** | `InstantX/Qwen-Image-ControlNet-Union` → `diffusion_pytorch_model.safetensors` (3.54 GB) | **native** `ControlNetLoader` → `models/controlnet/` | standard path; canny/softedge/depth/**pose** |
| **FLUX.2 Dev** | `alibaba-pai/FLUX.2-dev-Fun-Controlnet-Union` | custom node **`bryanmcguire/comfyui-flux2fun-controlnet`** | needs that node; canny/hed/depth/**pose**/mlsd + inpaint. (Klein alt: Civitai `2578291`) |
| **Z-Image Turbo** | `alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.0` (use **2.1**) | **`ModelPatchLoader` → `models/model_patches/`** (NOT controlnet/!) | different mechanism; official `Z_Image_Turbo_ControlNet_Union.json` in the HF repo |

**Pose preprocessor** (all): `DWPreprocessor` from `comfyui_controlnet_aux` — LoadImage(ref) → DWPreprocessor → the base's control apply.

## Qwen-Image-Edit (image editing) — models

| Component | File | HF repo path |
|---|---|---|
| diffusion | `qwen_image_edit_fp8_e4m3fn.safetensors` | `Comfy-Org/Qwen-Image-Edit_ComfyUI` /split_files/diffusion_models/ |
| text-enc | `qwen_2.5_vl_7b_fp8_scaled.safetensors` | `Comfy-Org/Qwen-Image_ComfyUI` /split_files/text_encoders/ |
| vae | `qwen_image_vae.safetensors` | `Comfy-Org/Qwen-Image_ComfyUI` /split_files/vae/ |
| LoRA (4-step, opt.) | `Qwen-Image-Lightning-4steps-V1.0.safetensors` | `lightx2v/Qwen-Image-Lightning` |

Nodes: `LoadImage → ImageScaleToTotalPixels → VAEEncode → TextEncodeQwenImageEdit(prompt) → KSampler`
(+ `LoraLoaderModelOnly` for the Lightning LoRA). Official template: docs.comfy.org/tutorials/image/qwen/qwen-image-edit.

## Official templates to reuse (same models as above)
- Qwen t2i / ControlNet: `Comfy-Org/workflow_templates` → `image_qwen_Image_2512.json` and the `..._controlnet` variant.
- Qwen-Edit: ComfyUI `Browse Templates → Image → Qwen-Image-Edit`.
- Z-Image ControlNet: `Z_Image_Turbo_ControlNet_Union.json` in the alibaba-pai repo.
- FLUX.2 ControlNet: example workflow in the `bryanmcguire/comfyui-flux2fun-controlnet` node repo.

## Finalize on the pod (claude task — guarantees they load)
1. Download the ControlNet + edit models (see `download_models.sh`, section 4).
2. Install the extra node for FLUX.2: clone `bryanmcguire/comfyui-flux2fun-controlnet` into `custom_nodes/`, restart ComfyUI.
3. For each: open the OFFICIAL template (Browse Templates / repo JSON) in the ComfyUI UI, confirm it
   references the downloaded model files (fix names if needed), add `LoadImage → DWPreprocessor` for the
   pose reference, run once to confirm it generates.
4. **Export API format** (Settings → dev mode → Save (API Format)) → save as
   `wf_pose_qwen.json`, `wf_pose_flux2.json`, `wf_pose_zimage.json`, `wf_qwen_edit.json` so
   `smoke_submit.py`/`generate.py` can drive them headless.
5. Smoke each: a pose reference image → generated image following that pose; and an input image → Qwen-Edit result.

## Recommendation
Per the research, **Qwen-Image is the pose/control champion** (native, multi-reference face+clothes+pose) —
make it your primary pose base. Z-Image for fast t2i (+ its Fun-ControlNet when you need pose cheaply),
FLUX.2 for max-detail hero frames. Qwen-Edit for "change outfit / pose / relight" on an existing shot.
