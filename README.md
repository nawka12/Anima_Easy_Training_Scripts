# Anima Easy Training Scripts

A PySide6 trainer UI focused on producing LoRAs for the **Anima** model — a DiT with a Qwen3 + T5 text encoder pair and the Qwen-Image VAE.

This is a fork of [67372a/LoRA_Easy_Training_Scripts](https://github.com/67372a/LoRA_Easy_Training_Scripts) (which itself descends from derrian-distro's original). The upstream project supports the full SD1.x / SD2.x / SDXL / Flux family and Textual Inversion; this fork strips all of that out so the UI only exposes the controls Anima actually uses.

If you want a general-purpose LoRA trainer, use the upstream. Use this one when Anima is what you're training.

## What's different from upstream

- General Args is rewritten around Anima: **DiT model**, **Qwen3**, **VAE**, **T5 tokenizer** (optional) are the four model inputs. SDXL / V2 / V-param / V-pred / FP8 / clip_skip / CLIP max_token_length are gone.
- New **Anima Sampling** section: `timestep_sampling` (sigma / uniform / sigmoid / shift / flux_shift), `discrete_flow_shift`, `sigmoid_scale`, Qwen3 / T5 max token lengths.
- New **Anima Memory / Attention** section: `vae_chunk_size`, `blocks_to_swap`, `vae_disable_cache`, `flash_attn`, `split_attn`, `unsloth_offload_checkpointing`. xFormers automatically locks `split_attn` on.
- New **Additional Resolutions** widget supports sd-scripts' multi-resolution `[[datasets]]` shape — add a row per extra resolution (each with its own `skip_image_resolution`, `batch_size`, bucket settings) and save round-trips the multi-`[[datasets]]` toml.
- Flux, EDM² loss weighting, ExperimentalArgs (flow / CFM / debiased estimation), NoiseOffset, and Textual Inversion widgets are removed.
- Train Mode menu is removed; the backend is always invoked with `anima=True` → `anima_train_network.py`.

Everything else from upstream (network args, optimizer args, queue, sampling, logging, accelerate, custom optimizers via `LoraEasyCustomOptimizer`, etc.) is unchanged.

## Installation

### Linux (Python 3.11)

```bash
git clone --recurse-submodules https://github.com/nawka12/Anima_Easy_Training_Scripts.git -b refresh
cd Anima_Easy_Training_Scripts
git submodule update --init --recursive
./install311.sh
```

If `install311.sh` doesn't work for your environment, fall back to the manual recipe from [upstream's README](https://github.com/67372a/LoRA_Easy_Training_Scripts#linux) — the dependency setup is identical, only the model surface changes.

### Windows

```
git clone https://github.com/nawka12/Anima_Easy_Training_Scripts.git -b refresh
cd Anima_Easy_Training_Scripts
install.bat
```

When the installer asks "Are you using this locally? (y/n):" answer `y` if you're training on this machine. Otherwise the backend won't install.

## Running

```bash
./run.sh           # Linux
run.bat            # Windows
```

The UI launches the backend in the background and the args UI in the foreground. Point it at your Anima DiT + Qwen3 + VAE, fill in your datasets, hit **Start Training**.

## TOML format

Configs are sectioned by widget. Anima-specific keys live under `[anima_args.args]`. The simplest single-resolution config looks like this:

```toml
[[subsets]]
caption_extension = ".txt"
image_dir = "/path/to/dataset"
num_repeats = 1
name = "concept"

[general_args.args]
seed = 42
mixed_precision = "bf16"
gradient_checkpointing = true
max_train_epochs = 10
cache_latents = true
cache_latents_to_disk = true
sdpa = true

[general_args.dataset_args]
resolution = 1024
batch_size = 4

[bucket_args.dataset_args]
enable_bucket = true
min_bucket_reso = 512
max_bucket_reso = 1536
bucket_reso_steps = 64

[anima_args.args]
pretrained_model_name_or_path = "/path/to/anima_dit.safetensors"
qwen3 = "/path/to/qwen3_0.6b.safetensors"
vae = "/path/to/qwen_image_vae.safetensors"
qwen3_max_token_length = 512
t5_max_token_length = 512
timestep_sampling = "sigmoid"
discrete_flow_shift = 3.0
sigmoid_scale = 1.0
attn_mode = "flash"

[network_args.args]
network_dim = 32
network_alpha = 16

[optimizer_args.args]
optimizer_type = "AdamW"
lr_scheduler = "cosine"
learning_rate = 1e-4

[saving_args.args]
output_dir = "/path/to/output"
save_precision = "bf16"
save_model_as = "safetensors"
output_name = "my_anima_lora"
save_every_n_epochs = 1
```

### Multi-resolution training

To train at multiple resolutions simultaneously, add rows under **Additional Resolutions (Mixed-Res)** in the UI. The toml is then saved using sd-scripts' `[[datasets]]` shape:

```toml
[[datasets]]
resolution = 512
batch_size = 8
enable_bucket = true
min_bucket_reso = 512
max_bucket_reso = 1536
bucket_reso_steps = 64

  [[datasets.subsets]]
  caption_extension = ".txt"
  image_dir = "/path/to/dataset"
  num_repeats = 1

[[datasets]]
resolution = 1024
skip_image_resolution = 512   # skip images smaller than 512 for this bucket
batch_size = 8
enable_bucket = true
min_bucket_reso = 512
max_bucket_reso = 1536
bucket_reso_steps = 64

  [[datasets.subsets]]
  caption_extension = ".txt"
  image_dir = "/path/to/dataset"
  num_repeats = 1
```

The first `[[datasets]]` block is built from your General Args + Bucket Args + subset list; each additional resolution row in the UI becomes another block. All blocks share the same subset list. Loading a multi-resolution toml back into the UI restores the rows automatically.

## Credit

- [67372a/LoRA_Easy_Training_Scripts](https://github.com/67372a/LoRA_Easy_Training_Scripts) — direct parent of this fork. Maintains the broader trainer, the extended `LoraEasyCustomOptimizer` set, RamTorch integration, and the backend.
- [derrian-distro/LoRA_Easy_Training_Scripts](https://github.com/derrian-distro/LoRA_Easy_Training_Scripts) — original project the upstream descends from.
- [kohya-ss/sd-scripts](https://github.com/kohya-ss/sd-scripts) — the training scripts at the bottom of the stack.
- [qt-material](https://github.com/UN-GCPDS/qt-material) — UI theming.

The Anima training script (`anima_train_network.py`) lives in the bundled `sd_scripts` submodule.
