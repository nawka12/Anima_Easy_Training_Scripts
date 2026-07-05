# Anima Easy Training Scripts

A PySide6 trainer UI focused on producing LoRAs for the **Anima** model — a DiT with a Qwen3 text encoder and the Qwen-Image VAE.

This is a fork of [67372a/LoRA_Easy_Training_Scripts](https://github.com/67372a/LoRA_Easy_Training_Scripts). The UI keeps the familiar layout, but the **training backend has been replaced**: instead of a kohya/sd_scripts fork, it now drives [**Raelina-Rae/diffusion-pipe**](https://github.com/Raelina-Rae/diffusion-pipe), a DeepSpeed-based framework with first-class Anima support (LoRA, LoKr, and full fine-tune, plus in-training sampling and contrastive flow matching).

If you want a general-purpose LoRA trainer, use the upstream. Use this one when Anima is what you're training.

> **⚠️ Linux only.** diffusion-pipe runs on DeepSpeed, which does not run natively on Windows. Train on Linux (or WSL2 with a working CUDA setup). The `.bat` launchers are left in place but are **not supported** for the diffusion-pipe backend.

> **⚠️ Preview weights.** Anima is still training. A LoRA trained against the current **preview** weights may not transfer to the final release — treat preview LoRAs as throwaway and plan to retrain. If you upload one, say it was trained on preview.

## What's different from upstream

The frontend↔backend HTTP contract is unchanged, but the argument vocabulary and everything downstream of it were rewritten for diffusion-pipe:

- **Adapters:** the Network panel offers **LoRA**, **LoKr**, or **Full fine-tune** (omitting the adapter). The LyCORIS algo zoo, block weights, DoRA, LoRA-FA, and `network_alpha` are gone — diffusion-pipe forces `alpha = rank`. LoKr exposes `factor`, `use_tucker`, and rank/module dropout.
- **Model inputs:** **DiT model** (`transformer_path`), **Qwen3** (`llm_path`), **VAE** (`vae_path`). The T5 tokenizer path and token-length knobs are gone (dp's Anima is Qwen3-only).
- **Timestep sampling:** `logit_normal` or `uniform` only, with `sigmoid_scale`. Contrastive flow matching is a single `contrastive_flow_lambda`.
- **Optimizer / scheduler:** optimizer names map to dp equivalents (`AdamW`→`adamw_optimi`, `AdamW8bit`→`AdamW8bitKahan`, …) with a free-text escape hatch to the `pytorch_optimizer` library. Schedulers collapse to **constant / linear / cosine** + warmup. The rex/restart/custom-scheduler zoo, min-SNR, noise-offset, and other epsilon-prediction knobs are removed (Anima is flow matching).
- **Bucketing:** aspect-ratio buckets (`enable_ar_bucket`, `num_ar_buckets`) instead of resolution-step buckets. Multi-resolution is a single list — see below.
- **Sampling:** the Sample panel is a prompt list with **per-prompt negative prompts** plus width/height/steps/CFG/seed, written to dp's `sample.toml`.
- **Masked training:** per-directory, via a mask folder on each subset (`mask_path`).
- **New knobs:** `blocks_to_swap` (low-VRAM), `pipeline_stages`, `compile`, `llm_adapter_lr`, `caching_batch_size`.

**Gained vs. the old backend:** full fine-tune, LoKr, in-training sampling with negatives, contrastive flow matching, native multi-resolution, per-directory masks, pipeline parallelism + block swap.

**Lost:** the LyCORIS algo zoo beyond LoKr, block weights, custom LR schedulers, regularization images, per-subset caption/augmentation knobs, and native Windows.

### Old configs

Configs saved with the old sd_scripts backend load fine — the UI auto-detects the old format, converts it, and pops up a report of every dropped or remapped setting. You can also convert offline:

```bash
python tools/convert_sdscripts_toml.py old_config.toml [new_config.toml]
```

## Installation

### Linux (Python 3.11)

```bash
git clone --recurse-submodules https://github.com/nawka12/Anima_Easy_Training_Scripts.git -b refresh
cd Anima_Easy_Training_Scripts
git submodule update --init backend
./install311.sh
```

The backend installer pulls the `diffusion_pipe` submodule (not its heavy sub-submodules — Anima doesn't need them), creates a backend-level venv at `backend/venv`, and installs DeepSpeed + diffusion-pipe's requirements. This needs a working CUDA toolchain.

When the installer asks "Are you using this locally? (y/n):" answer `y` if you're training on this machine. Otherwise the backend won't install.

### Windows

Not supported — DeepSpeed does not run natively on Windows. Use **WSL2** with a CUDA-enabled setup and follow the Linux instructions inside it.

## Running

```bash
./run.sh           # Linux
```

The UI launches the backend in the background and the args UI in the foreground. Point it at your Anima DiT + Qwen3 + VAE, fill in your datasets, hit **Start Training**.

> On the first run, diffusion-pipe pre-caches latents and text embeddings before training starts, so the first epoch is slow to begin. Trained files land in a timestamped run directory inside your **Output Folder** (which is also where TensorBoard should point).

## TOML format

Saved configs are still sectioned by widget (`[group.args]` / `[group.dataset_args]`), but the keys inside now use the diffusion-pipe vocabulary. At training time the backend translates these into diffusion-pipe's three config files (`main.toml`, `dataset.toml`, `sample.toml`). A simple single-resolution LoKr config:

```toml
[[subsets]]
image_dir = "/path/to/dataset"
num_repeats = 1
name = "concept"
# mask_path = "/path/to/masks"   # optional, per-directory masked training

[general_args.args]
mixed_precision = "bf16"
gradient_checkpointing = true
max_train_epochs = 10
# blocks_to_swap = 20            # optional, lowers VRAM

[general_args.dataset_args]
resolution = 1024
batch_size = 4

[bucket_args.dataset_args]
enable_ar_bucket = true
num_ar_buckets = 7

[anima_args.args]
pretrained_model_name_or_path = "/path/to/anima_dit.safetensors"
qwen3 = "/path/to/qwen3_0.6b.safetensors"
vae = "/path/to/qwen_image_vae.safetensors"
timestep_sampling = "logit_normal"
sigmoid_scale = 1.0
llm_adapter_lr = 0            # 0 freezes the Qwen3→DiT adapter (more stable on small sets)
# contrastive_flow_lambda = 0.05   # optional contrastive flow matching

[network_args.args]
type = "lokr"                 # "lora" | "lokr" | "none" (full fine-tune)
rank = 8
factor = 4
# note: no alpha — diffusion-pipe forces alpha = rank

[optimizer_args.args]
optimizer_type = "AdamW"      # mapped to adamw_optimi
lr_scheduler = "cosine"       # constant | linear | cosine
learning_rate = 2e-5

[saving_args.args]
output_dir = "/path/to/output"
save_precision = "bf16"
output_name = "my_anima_lora"
save_every_n_epochs = 1

[sample_args.args]
sample_every_n_epochs = 1
sample_at_first = true
width = 1024
height = 1024
num_inference_steps = 32
guidance_scale = 4.0
seed = 42
prompts = [
  { prompt = "1girl, solo, masterpiece", negative_prompt = "worst quality, low quality" },
]
```

For a **full fine-tune**, set `type = "none"` (the adapter is omitted). For a plain **LoRA**, use `type = "lora"` with just a `rank`.

### Multi-resolution training

diffusion-pipe trains multiple resolutions natively: the dataset is duplicated across each area in a `resolutions` list. Add rows under **Additional Resolutions (Mixed-Res)** in the UI; at training time they collapse into a single list, e.g. `resolutions = [768, 1024]`. No per-block bucket configuration is needed.

### Captions (`.txt` + `.caption`)

diffusion-pipe's dataset reads captions from `<image>.txt` only (there is no configurable caption extension). If your dataset pairs **booru tags in `.txt`** with a **natural-language caption in `.caption`**, enable **Captions → "Train on both .txt tags and .caption NLP"**. On validation, the backend writes a `captions.json` into each dataset folder (next to the images) listing each image's captions, and turns on `online_captions` (with random-caption off).

diffusion-pipe then trains **one example per caption**, so:

- image with only `.txt` → trained **once** per epoch (tags)
- image with `.txt` + `.caption` → trained **twice** per epoch (once on the tags, once on the NL caption)

i.e. a 1k-image set where every image has both files becomes 2k effective samples per epoch. Mixed folders work too (1-caption and 2-caption images coexist). Without this option, only `.txt` is used and `.caption` files are ignored (they also produce harmless "could not open" warnings, since diffusion-pipe scans them as would-be images).

## Credit

- [Raelina-Rae/diffusion-pipe](https://github.com/Raelina-Rae/diffusion-pipe) — the training backend (a fork of tdrussell/diffusion-pipe with Anima support).
- [67372a/LoRA_Easy_Training_Scripts](https://github.com/67372a/LoRA_Easy_Training_Scripts) — direct parent of this fork; source of the PySide6 UI.
- [derrian-distro/LoRA_Easy_Training_Scripts](https://github.com/derrian-distro/LoRA_Easy_Training_Scripts) — original project the upstream descends from.
- [qt-material](https://github.com/UN-GCPDS/qt-material) — UI theming.

Training is run by diffusion-pipe's `train.py`, launched via DeepSpeed from the bundled `diffusion_pipe` submodule in the backend.
