# neuro-avatar — pod scripts & ComfyUI workflows

Provisioning scripts and ComfyUI workflow/template JSONs for a Vast.ai GPU pod used for
neuro-avatar (AI-persona) image generation and character-LoRA training.

## Scripts
- `provision.sh` — pod provisioning.
- `startup.sh` — session startup.
- `env.sh` — environment entry point (activates venv, sources `secrets.env`, sets HF vars).
- `download_models.sh` / `dl_zimage.sh` — model download helpers.
- `smoke_submit.py` — smoke test that submits a workflow to ComfyUI.

## ComfyUI workflows / templates
- `wf_*.json` — API-format workflows (flux2, sdxl/biglust, hidream, zimage, 2604).
- `tpl_*.json` — full workflow templates (hidream, zimage).

## Not included (by design)
- `secrets.env` and any tokens — never committed.
- Third-party repos (`ComfyUI/`, `ai-toolkit/`, `kohya_ss/`) and model weights — see `.gitignore`.

## Usage
`env.sh` expects a `secrets.env` next to it defining `HF_TOKEN` and `CIVITAI_TOKEN`.
Create it locally; it is gitignored.
