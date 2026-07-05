# BACK-MIGRATE: Move the training backend from sd_scripts to diffusion-pipe

**Status:** planning document — no code has been written yet.
**Target backend:** https://github.com/Raelina-Rae/diffusion-pipe (fork of `tdrussell/diffusion-pipe`)
**Audience:** an implementing agent (Opus/Sonnet). Everything you need is either in this file, in this repo, or in the two repos linked below. When this document says *verify*, it means the claim was not confirmed during planning — check the source before building on it.

---

## 1. Context and goal

This project ("Anima Easy Training Scripts") is a PySide6 desktop frontend for training **Anima** LoRAs. It currently drives a kohya-style **sd_scripts** fork through a small HTTP backend. We are replacing that trainer with **diffusion-pipe**, a DeepSpeed-based training framework whose Raelina-Rae fork has first-class Anima support (LoRA + LoKr + full fine-tune, in-training sampling, multi-caption, contrastive flow matching).

**Guiding principle:** the frontend ↔ backend HTTP contract survives; the *vocabulary* of the args payload and everything downstream of `/validate` is rewritten. This is not a submodule swap — it is a rewrite of the translation layer plus a trimming/extension of the UI panels.

### Repos involved

| Repo | Role | Branch |
|---|---|---|
| `nawka12/Anima_Easy_Training_Scripts` (this repo) | PySide6 frontend | `refresh` — create work branch `diffusion-pipe` |
| `nawka12/Anima_Easy_Training_scripts_Backend` | Starlette HTTP server wrapping the trainer (git submodule at `backend/`) | `refresh` — create work branch `diffusion-pipe` |
| `Raelina-Rae/diffusion-pipe` | New trainer. Replaces the `sd_scripts/` submodule inside the backend repo | `main` |

The `backend/` submodule is **not checked out** in this working copy. First step of any implementation session:
`git submodule update --init backend` (and inside the backend repo, diffusion-pipe itself has submodules: `git submodule update --init --recursive`).

### Hard constraints (decided; do not relitigate)

1. **Linux-only runtime.** DeepSpeed does not run natively on Windows. The `.bat` installers/launchers either get removed or become thin WSL wrappers — decide with the user before deleting, default is to leave them but add a README warning.
2. **Anima-only.** The frontend already hardcodes `anima=True` (`main_ui_files/MainUI.py:295-301`). Do not port SDXL/Flux/SD1.5 paths.
3. **Keep the HTTP contract shape.** Same endpoints, same polling model. Remote/Colab usage via the tunnel service must keep working.
4. **Old saved TOMLs will not load.** Provide a converter script (Phase 5) rather than dual-format support in the UI.

---

## 2. Current architecture (as-is)

```
PySide6 UI (this repo)                Backend repo (submodule at backend/)
┌───────────────────────┐   HTTP     ┌──────────────────────────────────────┐
│ main_ui_files/*.py    │ ────────►  │ main.py (Starlette, port 8000)       │
│ builds nested args    │  /validate │  ├ utils/validation.py (449 lines)   │
│ dict keyed to         │  /train    │  ├ utils/process.py    (76 lines)    │
│ sd_scripts arg names  │  /is_train │  │   → writes runtime_store/         │
│                       │  /stop_*   │  │     config.toml + dataset.toml    │
│ queue_store/*.toml    │  /check_p  │  └ subprocess.Popen(                 │
│ (saved job configs)   │  /resize   │      sd_scripts/anima_train_network  │
└───────────────────────┘            │      .py --config_file --dataset_cfg)│
                                     └──────────────────────────────────────┘
```

### The `/validate` payload the UI sends today

```jsonc
{
  "args": {
    "general_args":   { "seed": 23, "max_train_epochs": 10, "mixed_precision": "bf16", ... },
    "anima_args":     { "pretrained_model_name_or_path": ..., "qwen3": ..., "vae": ...,
                        "t5_tokenizer_path": ..., "qwen3_max_token_length": ...,
                        "timestep_sampling": ..., "split_attn": ..., ... },
    "network_args":   { "network_dim": 32, "network_alpha": 16, "network_args": {"algo": "lokr", ...}, ... },
    "optimizer_args": { "optimizer_type": ..., "learning_rate": ..., "lr_scheduler": ...,
                        "optimizer_args": {...}, "lr_scheduler_args": {...}, ... },
    "saving_args":    { "output_dir": ..., "output_name": ..., "save_every_n_epochs": ...,
                        "save_toml": ..., "tag_occurrence": ..., ... },
    "sample_args":    { ... prompt text + sampler + cadence ... },
    "logging_args":   { "logging_dir": ..., wandb keys ... },
    "extra_args":     { free-form key/value flags }
  },
  "dataset": {
    "general_args": { "resolution": ..., "batch_size": ... },
    "bucket_args":  { "enable_bucket": ..., "min_bucket_reso": ..., "max_bucket_reso": ...,
                      "bucket_reso_steps": ..., "bucket_no_upscale": ... },
    "subsets": [ { "image_dir": ..., "num_repeats": ..., "caption_extension": ...,
                   "shuffle_caption": ..., "keep_tokens": ..., ... }, ... ]
  },
  "accelerate": { "enabled": bool, "num_processes": int, "main_process_port": int }
}
```

Group names come from each widget's `self.name` (see `main_ui_files/ArgsListUI.py`, and `self.name = "..."` in each `main_ui_files/*UI.py`). Multi-resolution training sends `dataset = {"datasets": [...]}` instead (see `MainUI.py:247-258`).

### Backend endpoints (all must keep working)

| Endpoint | Method | Behavior today |
|---|---|---|
| `/validate` | POST | Validates payload, writes `runtime_store/config.toml` + `dataset.toml`, returns processed args + tag counts |
| `/train` | GET | Query params `train_mode/sdxl/flux/anima/accelerate_*`; picks train script; `subprocess.Popen` |
| `/is_training` | GET | `{"training": bool, "errored": bool}` from `Popen.poll()` |
| `/stop_training` | GET | `terminate()`, or `kill()` with `?force=true` |
| `/check_path` | POST | Path/extension existence check for drag-drop highlighting |
| `/tokenize_text` | GET | CLIP token count (used by SampleUI) |
| `/resize` (`start_resize`) | POST | Runs `utils/resize_lora.py` on an existing LoRA file |
| `/stop_server`, tunnel start/kill | GET | Lifecycle + cloudflared/remote tunnel |

---

## 3. Target architecture (to-be)

diffusion-pipe has **no API and no server** — it is launched as:

```bash
NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 deepspeed --num_gpus=N train.py --config /path/to/main.toml
```

and is configured by **three TOML files**:

1. **Main config** — training loop, `[model]`, `[adapter]`, `[optimizer]`, `[monitoring]`. References the other two by path (`dataset = '...'`, `sample = '...'`).
2. **`dataset.toml`** — resolutions, AR buckets, caption handling, `[[directory]]` entries.
3. **`sample.toml`** — sampling resolution/steps/cfg/seed + `[[prompts]]` list.

The backend keeps its Starlette server and endpoints; `/validate` now emits `runtime_store/main.toml`, `runtime_store/dataset.toml`, `runtime_store/sample.toml`, and `/train` launches deepspeed.

Reference configs to read before implementing (in the diffusion-pipe repo):
- `examples/anima_examples.toml` — Anima main config (canonical starting point)
- `examples/main_example.toml` — every top-level key, annotated
- `examples/dataset.toml` — every dataset key, annotated
- `examples/sample.toml` — sampling format
- `docs/supported_models.md` § "Anima" — model paths, `llm_adapter_lr` guidance, output format (ComfyUI/LyCORIS)
- `train.py` — adapter types (`lora`, `lokr` only; anything else raises `NotImplementedError` ~line 145), optimizer resolution (explicit types + `pytorch_optimizer` passthrough ~line 686)

---

## 4. Argument mapping (the core of the work)

This section is the spec for the new `utils/validation.py` + `utils/process.py` in the backend, and drives the UI changes in Phase 4. Legend: ✅ direct map, 🔧 needs conversion, ➕ new (no UI today), ❌ dropped (no equivalent).

### 4.1 `general_args` + dataset `general_args` → main config top level

| UI arg (sd_scripts) | diffusion-pipe | Notes |
|---|---|---|
| `max_train_epochs` | `epochs` | ✅ |
| `gradient_accumulation_steps` | `gradient_accumulation_steps` | ✅ |
| `gradient_checkpointing` | `activation_checkpointing` | ✅ rename |
| `mixed_precision` | `[model].dtype` + `save_dtype` | 🔧 `bf16`→`bfloat16`, `fp16`→`float16` |
| `seed` | — | ❌ **verify**: grep diffusion-pipe for a seed option; if absent, drop from UI (sample.toml has its own `seed`) |
| `max_data_loader_n_workers` | `map_num_proc` (caching only) | 🔧 approximate; expose as "caching workers" |
| `persistent_data_loader_workers` | — | ❌ |
| dataset `batch_size` | `micro_batch_size_per_gpu` | ✅ rename |
| dataset `resolution` | dataset.toml `resolutions = [N]` | 🔧 list-valued; multi-res becomes multiple entries (see 4.8) |
| — | `pipeline_stages`, `partition_method`, `blocks_to_swap`, `compile`, `steps_per_print`, `caching_batch_size` | ➕ new controls (Phase 4) |

### 4.2 `anima_args` → `[model]`

| UI arg | diffusion-pipe `[model]` | Notes |
|---|---|---|
| `pretrained_model_name_or_path` | `transformer_path` | ✅ |
| `qwen3` | `llm_path` | ✅ |
| `vae` | `vae_path` | ✅ |
| `timestep_sampling` | `timestep_sample_method` | 🔧 verify value names match (`uniform`, `logit_normal`, ...) |
| `t5_tokenizer_path`, `t5_max_token_length` | — | ❌ diffusion-pipe's Anima uses Qwen3 only |
| `qwen3_max_token_length` | — | ❌ **verify** — check `models/` in diffusion-pipe for a token-length knob |
| `vae_disable_cache`, `split_attn`, `unsloth_offload_checkpointing` | — | ❌ (activation checkpointing + `blocks_to_swap` are the equivalents) |
| — | `llm_adapter_lr` (default `0`), `sigmoid_scale`, `flux_shift`, `multiscale_loss_weight` | ➕ new controls |

`type = 'anima'` is emitted unconditionally by the backend.

### 4.3 `network_args` → `[adapter]`

Only **`lora`** and **`lokr`** exist; omitting `[adapter]` entirely = full fine-tune.

| UI arg | diffusion-pipe | Notes |
|---|---|---|
| algo selector (LoRA/LoCon/LoHa/LoKr/DyLoRA/IA3/BOFT/GLoRA/GoRA/RaLoRA…) | `type = 'lora' \| 'lokr'` (+ "Full fine-tune" option) | 🔧 collapse to 3 choices |
| `network_dim` | `rank` | ✅ |
| `network_alpha` | — for lokr (alpha forced = rank per docs); **verify** for lora whether peft `lora_alpha` is exposed | 🔧 |
| LoKr `factor` (network_args) | `factor` | ✅ (`-1` = auto) |
| — (lokr extras) | `decompose_both`, `use_tucker`, `include_conv`, `rank_dropout`, `module_dropout`, `rank_dropout_scale` | ➕ per `docs/supported_models.md` §Cosmos-Predict2 (same LoKr backend) |
| conv_dim/conv_alpha, network_dropout, block weights, `train_norm`, LoRA-FA | — | ❌ delete; `block_weight_presets/` becomes dead — remove dir + `modules/BlockWeightWidgets.py` wiring |
| — | `init_from_existing` (resume-from-LoRA) | ➕ maps loosely to today's `network_weights` — **verify** semantics in train.py |

### 4.4 `optimizer_args` → `[optimizer]` + top-level LR keys

| UI arg | diffusion-pipe | Notes |
|---|---|---|
| `optimizer_type` | `type` | 🔧 name map: `AdamW`→`adamw_optimi`, `AdamW8bit`→`AdamW8bitKahan`, `Prodigy`→`Prodigy`, `Came`→`CAME`; unknown names pass through to the `pytorch_optimizer` library (train.py ~686), so keep a free-text escape hatch |
| `learning_rate` / `unet_lr` | `lr` | ✅ (single LR for Anima) |
| `text_encoder_lr` | `[model].llm_adapter_lr` | 🔧 closest analogue; not a text-encoder LR |
| `optimizer_args` dict | inlined into `[optimizer]` (`betas`, `weight_decay`, `eps`, …) | ✅ |
| `max_grad_norm` | `gradient_clipping` | ✅ rename |
| `lr_scheduler` + `lr_scheduler_args` + custom_scheduler (rex, cosine restarts…) | `warmup_steps`, optional `lr_scheduler = 'linear'`, `force_constant_lr` | 🔧 **major loss** — scheduler UI collapses to warmup + constant/linear |
| `warmup_ratio` | `warmup_steps` | 🔧 backend computes steps from dataset size (mirror the logic in current `validation.py: validate_warmup_ratio`) |
| `min_snr_gamma`, `scale_weight_norms`, noise offset family, `ip_noise_gamma`, `zero_terminal_snr` | — | ❌ epsilon-prediction concepts; Anima is flow matching. Delete from UI (Anima path) |
| `loss_type` huber | `pseudo_huber_c` | 🔧 verify semantics in train.py before mapping; else drop |
| masked loss checkbox | dataset `[[directory]].mask_path` | 🔧 moves from optimizer UI to subset UI |

### 4.5 `saving_args` → main config top level

| UI arg | diffusion-pipe | Notes |
|---|---|---|
| `output_dir` / `output_name` | `output_dir` / `output_name` | ✅ note: dp creates a timestamped run dir inside `output_dir` — **verify** and surface the real path in logs |
| `save_every_n_epochs` / `save_every_n_steps` | same names | ✅ (+ `save_every_n_examples` ➕) |
| `save_precision` | `save_dtype` | ✅ rename |
| `save_state` / `resume` | `checkpoint_every_n_epochs` / `checkpoint_every_n_minutes` + `--resume_from_checkpoint` CLI flag | 🔧 resume becomes a `/train` query param that appends the CLI flag — **verify** exact flag name in train.py |
| `save_toml`, `tag_occurrence`, `tag_file_location` | unchanged | ✅ frontend-side features, no backend involvement |
| `save_last_n_epochs`, huggingface upload | — | ❌ |

Output format: Anima LoRA/LoKr saved in **ComfyUI/LyCORIS format** (docs). Consequence: **verify** `utils/resize_lora.py` in the backend still works on these files; if not, disable the resize UI for dp-produced files.

### 4.6 `bucket_args` + subsets → `dataset.toml`

Semantics change: sd_scripts buckets by **resolution steps**; diffusion-pipe buckets by **aspect ratio** around fixed pixel areas.

| UI arg | diffusion-pipe (dataset.toml) | Notes |
|---|---|---|
| `enable_bucket` | `enable_ar_bucket` | ✅ |
| `min_bucket_reso` / `max_bucket_reso` / `bucket_reso_steps` | `min_ar` / `max_ar` / `num_ar_buckets` | 🔧 **different semantics** — replace UI fields, don't convert values. Sensible defaults: `min_ar=0.5, max_ar=2.0, num_ar_buckets=7` |
| `bucket_no_upscale` | — | ❌ |
| — | `frame_buckets` | emit `[1]` always (image training) |
| subset `image_dir` | `[[directory]].path` | ✅ |
| subset `num_repeats` | `[[directory]].num_repeats` | ✅ same semantics (fork docs say so explicitly) |
| subset `shuffle_caption` | top-level `cache_shuffle_num` (+ `cache_shuffle_delimiter`) | 🔧 shuffling happens at **caching time**, N pre-shuffled variants; per-subset → global |
| subset `keep_tokens` / `keep_tokens_separator` | top-level `keep_tokens` / `keep_tokens_separator` | 🔧 global, not per-subset. If subsets disagree, validation error |
| subset `caption_extension` | — (`.txt` fixed, or `captions.json` via `online_captions`) | ❌ **verify** — the fork added multi-caption support; check dataset code for extension config |
| subset regularization (`is_reg`) | — | ❌ **feature loss**, warn in UI removal notes |
| subset `flip_aug`/`color_aug`/`random_crop`/caption dropout | — | ❌ **verify** flip aug — check dataset.py; upstream had none |
| masked training | `[[directory]].mask_path` | ➕ per-directory mask folder |
| — | `enable_random_caption`, `skip_empty_caption`, `online_captions` | ➕ fork features worth exposing |

### 4.7 `sample_args` → `sample.toml` + main-config cadence keys

Today the backend writes an sd_scripts prompt txt; now emit:

```toml
# runtime_store/sample.toml
width = 1024
height = 1024
num_inference_steps = 32
guidance_scale = 4
seed = 24
[[prompts]]
prompt = "..."
negative_prompt = "..."
```

Main config gets `sample = 'runtime_store/sample.toml'` (absolute path), plus `sample_every_n_epochs` / `sample_every_n_steps` / `sample_at_first`. The SampleUI's sampler selector (euler_a etc.) has no equivalent — ❌ remove. `/tokenize_text` (CLIP counter) is misleading for Qwen3 captions; keep endpoint, consider hiding the UI counter.

### 4.8 Multi-resolution (`additional_resolutions`) 

diffusion-pipe does this natively: `resolutions = [768, 1024]` duplicates the dataset across areas. The whole `AdditionalResolutionsUI` + Format-2 `[[datasets]]` machinery collapses into a list-valued resolution field. Simplify aggressively.

### 4.9 `accelerate_args` → deepspeed launch

| UI arg | New | Notes |
|---|---|---|
| `enabled` + `num_processes` | `deepspeed --num_gpus=N` | N=1 when disabled |
| `main_process_port` | `--master_port` | ✅ |

Also: `pipeline_stages` must divide into num_gpus sensibly — add validation.

### 4.10 `logging_args` → `[monitoring]`

wandb keys map to `enable_wandb`/`wandb_api_key`/`wandb_tracker_name`/`wandb_run_name`. TensorBoard: **verify** — upstream dp writes TB events into the run dir; if so, `logging_dir` UI field just changes its help text. The frontend's TB auto-launch (`MainWindow.py:107`) then points at `output_dir`.

### 4.11 `extra_args` (free-form)

Keep as the escape hatch. New semantics: `key = value` pairs are TOML-injected into the main config top level (parse value as TOML literal, fall back to string). Document this in the UI tooltip.

---

## 5. Implementation phases

Work through these in order; each phase ends in a verifiable state.

### Phase 0 — Setup and reconnaissance (½ day)
1. Branch both repos (`diffusion-pipe` off `refresh`). In the backend repo, replace the `sd_scripts` submodule with `Raelina-Rae/diffusion-pipe` at `backend/diffusion_pipe/` (keep `custom_scheduler/` for now; delete once Phase 2 lands).
2. Update backend `installer.py`/`install.sh`: dp's `requirements.txt`, `deepspeed`, recursive submodule init. dp needs its own venv — reuse the `backend/sd_scripts/venv` path convention or update `main.py:15-23` in **this** repo to the new venv location (`backend/diffusion_pipe/venv` or a backend-level venv — prefer backend-level `backend/venv`).
3. Resolve every **verify** marker in §4 by reading dp source (`train.py`, `models/anima.py` or equivalent, dataset code). Update this file with findings before writing the translator.
4. Smoke-test dp standalone: hand-write the three TOMLs from `examples/anima_examples.toml`, run 1 epoch on ~10 images, confirm a ComfyUI-loadable safetensors appears. **Nothing else proceeds until this works.**

### Phase 1 — Backend translation layer (2–3 days)
Rewrite in the backend repo:
- `utils/validation.py` → validate the **new** payload schema (§4 right-hand columns). Keep the same function signature style: `validate(body) -> (passed, errors, ...)`. Path checks (model files exist, image dirs exist, output dir writable) carry over almost verbatim; tag-counting for `tag_occurrence` carries over (reads caption files — trainer-agnostic).
- `utils/process.py` → emit `runtime_store/main.toml`, `dataset.toml`, `sample.toml` with **absolute paths** cross-referencing each other.
- `main.py` `start_training`: replace the `match` block with:
  ```python
  cmd = ["deepspeed", f"--num_gpus={n}", f"--master_port={port}",
         str(Path("diffusion_pipe/train.py").resolve()),
         "--config", str(Path("runtime_store/main.toml").resolve())]
  env = {**os.environ, "NCCL_P2P_DISABLE": "1", "NCCL_IB_DISABLE": "1"}
  app.state.TRAINING_THREAD = subprocess.Popen(cmd, env=env, preexec_fn=os.setsid)
  ```
  **Critical:** deepspeed spawns worker processes. `terminate()` on the launcher orphans them. Use `preexec_fn=os.setsid` + `os.killpg(os.getpgid(p.pid), SIGTERM)` in `stop_training` (SIGKILL for `force`). Test stop/restart cycles explicitly — GPU memory must be freed.
- `/train` query params: accept and ignore legacy `sdxl/flux/anima`; honor `train_mode` values `lora|lokr|fft` (or read adapter type from validated args — simpler, prefer that); keep `accelerate_*` params, reinterpret as gpu count/port; add optional `resume=true`.
- `/is_training`, `/check_path`, `/tokenize_text`, tunnel, `/stop_server`: unchanged.
- `/resize`: keep; gate on the Phase-0 finding about LyCORIS-format compatibility.

**Verification:** `curl` the endpoints by hand with a JSON payload; confirm the three TOMLs match a known-good hand-written set (diff them); confirm train/stop/is_training lifecycle on the smoke-test dataset.

### Phase 2 — Frontend payload changes, minimal UI (2–3 days)
Rework panels to emit the new vocabulary. Suggested per-file order (self-contained commits):
1. `GeneralUI.py` — renames (§4.1), drop dead fields, add `pipeline_stages`/`compile`/`blocks_to_swap` under an "Advanced" collapsible. `anima_args` → new `[model]` keys (§4.2).
2. `NetworkUI.py` — collapse algo selector to LoRA/LoKr/Full fine-tune; LoKr sub-options; delete block-weight UI (`modules/BlockWeightWidgets.py` usage, `block_weight_presets/`).
3. `OptimizerUI.py` — optimizer name map + free-text type; scheduler section shrinks to warmup/constant/linear; move masked-loss to SubsetUI; delete noise/SNR fields.
4. `SavingUI.py` — renames; save/checkpoint/resume rework.
5. `BucketUI.py` — AR-bucket fields replace reso-bucket fields.
6. `SubsetUI.py` / `SubsetListUI.py` — per-directory keys (§4.6); move shuffle/keep_tokens to a global "Captions" group (put it in BucketUI or GeneralUI); add `mask_path`.
7. `SampleUI.py` — prompt list + negative per prompt; remove sampler selector.
8. `LoggingUI.py` — `[monitoring]` keys.
9. `AdditionalResolutionsUI.py` — replace with multi-value resolutions field (§4.8); delete Format-2 plumbing in `MainUI.process_toml`/`save_toml`.
10. `MainUI.py` — `train_params` becomes `{"train_mode": <lora|lokr|fft>, "num_gpus": ..., "master_port": ...}`.

Keep each widget's `self.name` group keys **unchanged** where possible (`general_args`, `optimizer_args`, …) so TomlFunctions save/load and the queue keep working untouched; only the keys inside groups change.

### Phase 3 — End-to-end integration (1–2 days)
- Full run from the UI: queue two jobs (one LoRA, one LoKr), confirm sequential execution, sampling images appear, `/stop_training` mid-run leaves the GPU clean, second job still starts.
- Full fine-tune path: `[adapter]` omitted; confirm output.
- Multi-GPU if hardware allows; otherwise verify the generated deepspeed command line by inspection.
- Failure UX: bad model path → `/validate` 400 with readable error; OOM mid-train → `is_training.errored=true` surfaces in UI.

### Phase 4 — Config converter + cleanup (1 day)
- `tools/convert_sdscripts_toml.py` (frontend repo): old saved TOML → new schema, printing warnings for every dropped arg (reg images, block weights, schedulers, noise args…). Wire into `TomlFunctions.load_toml` as a fallback: on unknown-schema detection, offer conversion.
- Delete dead code: `block_weight_presets/`, LoRA-FA/DyLoRA branches, `custom_scheduler` references, flux/sdxl remnants in `MainUI.train_helper`.
- README + window title updates; Windows/WSL warning.

### Phase 5 — Docs & release (½ day)
- Update `install.sh`/`update.sh` in both repos; pin the dp submodule commit.
- Note in README: LoRAs train against Anima **preview** weights — retraining will be needed for final weights (upstream docs warning).

**Total estimate: 7–10 focused days.**

---

## 6. Feature delta (communicate to users in release notes)

**Gained:** full fine-tune, LoKr for Anima, in-training sampling with negative prompts, eval datasets/eval loss (optional, ➕ UI later), pipeline parallelism + block swap (low-VRAM), multi-caption random-per-epoch, contrastive flow matching (fork), masked training per directory, native multi-resolution.

**Lost:** LyCORIS algo zoo beyond LoKr, block weights, custom LR schedulers (rex, restarts), regularization images, per-subset caption knobs, caption dropout, noise-offset/min-SNR family (mostly N/A for flow matching anyway), native Windows.

---

## 7. Risks and open questions

1. **DeepSpeed process management** — the #1 source of subtle bugs (orphaned workers holding VRAM). Mitigation in Phase 1; test kill paths early.
2. **Every §4 "verify" marker** — resolve in Phase 0 against dp source, not docs.
3. **Anima preview-weight churn** — upstream warns preview-trained LoRAs won't transfer to final weights. Not our problem to fix; document it.
4. **resize_lora.py vs LyCORIS output format** — may need lycoris-lora's own resize, or drop the feature.
5. **Progress reporting** — sd_scripts prints tqdm the UI never parsed anyway (`is_training` polling only), so no regression, but `steps_per_print=1` makes backend stdout useful; consider tailing it into the UI later (out of scope).
6. **Caching step** — dp pre-caches latents/text embeddings on first run (`caching_batch_size`, `map_num_proc`); first epoch start is slow. Set user expectations in UI status text.

## 8. Acceptance criteria

- [ ] From a clean clone + `install.sh`, a LoRA trains end-to-end from the UI on Linux and loads in ComfyUI against Anima.
- [ ] LoKr and full fine-tune paths produce loadable outputs.
- [ ] Queue of ≥2 jobs runs sequentially; stop mid-job frees the GPU and the next queued job can start.
- [ ] `/validate` rejects bad paths/args with actionable messages; UI shows them.
- [ ] Old-format TOML triggers the converter with a dropped-args report.
- [ ] Remote backend via tunnel still works (`/check_path`, train, poll).
- [ ] `grep -rn "sd_scripts\|sdxl\|flux" --include="*.py"` in both repos returns only intentional remnants (converter, comments).

---

## 9. Phase 0 findings (resolved `verify` markers)

Read against `Raelina-Rae/diffusion-pipe` @ `7240554` (main, cloned 2026-07-05). Anima is **not** its own model file — it is handled by `models/cosmos_predict2.py` (`CosmosPredict2Pipeline`, `self.name='anima'` when `llm_path` points at a Qwen3 `.safetensors`). `train.py` dispatches `model_type in ('cosmos_predict2','anima')` to it (train.py:356). This is why LoKr options mirror the Cosmos-Predict2 docs.

### Launch contract
- Command: `deepspeed --num_gpus=N --master_port=PORT train.py --config <main.toml>`. `train.py` also parses its **own** `--master_port` (default 29500) inside `distributed_init`; pass it too to be safe, but the deepspeed launcher's is what matters. Env: `NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1`.
- Resume: **`--resume_from_checkpoint`** (train.py:46, `nargs='?'`, `const=True`). Bare flag → resume most-recent run dir; `--resume_from_checkpoint <name>` → that subdir of `output_dir`. Requires a prior `checkpoint_every_n_*` to have written state.
- Useful extra CLI flags: `--regenerate_cache`, `--cache_only`, `--trust_cache`, `--reset_dataloader`, `--reset_optimizer`.
- Run dir: dp makes a **timestamped** subdir `output_dir/YYYYMMDD_HH-MM-SS/` (train.py:543) and copies the config + dataset toml into it. TB events (`SummaryWriter(log_dir=run_dir)`, train.py:924) and samples/checkpoints all land there. **Surface this real path in the UI/logs.** Frontend TB auto-launch should point at `output_dir` (parent) so it picks up the newest run.
- `save_every_n_epochs` **or** `save_every_n_steps` **or** `save_every_n_examples` is **required** (assert in `set_config_defaults`, train.py:95).

### [model] (§4.2) — from `cosmos_predict2.py`
- `type='anima'`, `transformer_path` (= UI `pretrained_model_name_or_path`), `vae_path` (= `vae`), `llm_path` (= `qwen3`). If `llm_path` is a **file** → Qwen3-0.6b Anima (uses bundled `configs/qwen3_06b`); if a **dir** → generic Transformers LLM. Emit the file path.
- `dtype` (from `mixed_precision`), optional `transformer_dtype`.
- `timestep_sample_method`: **ONLY** `logit_normal` (default) or `uniform` — anything else raises `NotImplementedError` (cosmos_predict2.py:390). UI `timestep_sampling` must collapse to these two.
- `sigmoid_scale` (default 1.0, scales the logit-normal), `flux_shift` (bool) **or** `shift` (float) for timestep shifting, `multiscale_loss_weight` (default none), `contrastive_flow_lambda` (default 0 — the fork's contrastive flow matching).
- Per-group LRs: `llm_adapter_lr` (default = base lr; docs recommend **0** to freeze the Qwen3→DiT adapter — more stable for small datasets), plus `self_attn_lr`/`cross_attn_lr`/`mlp_lr`/`mod_lr`. **A group LR of 0 freezes that param group** (cosmos_predict2.py:498).
- `cache_text_embeddings` (default true). `llm_adapter_path` optional external adapter weights.
- **DROPPED:** `qwen3_max_token_length` (tokenizer hardcoded `max_length=512`), `t5_tokenizer_path`/`t5_max_token_length` (Anima has no T5 path — the `t5_*` config keys are for the separate Cosmos-Predict2 flavor).

### [adapter] (§4.3) — from `train.py:116-145`
- `type` ∈ {`lora`, `lokr`}; **omit `[adapter]` entirely = full fine-tune**. Any other type → `NotImplementedError`.
- **`alpha` is FORBIDDEN** for both — dp forces `alpha=rank` and raises `NotImplementedError` if you set it. So **drop `network_alpha` from the Anima UI entirely.**
- `rank` (= `network_dim`), `dropout` (default 0.0), `dtype` (default model dtype).
- LoKr extras (all optional, defaults shown): `factor=-1` (auto), `use_tucker=false`, `decompose_both=false`, `rank_dropout=0.0`, `module_dropout=0.0`, `rank_dropout_scale=false`, `include_conv=false`.
- `init_from_existing` (adapter dir path, e.g. `.../epoch50`) → `load_adapter_weights` (base.py:281/601). This is the resume-from-LoRA analogue of `network_weights`.
- **Block swapping requires an adapter** (`assert 'adapter' in config`, train.py:574) and `pipeline_stages=1`.

### [optimizer] (§4.4) — from `train.py:636-686`
- Explicit types (matched **lowercased**): `adamw`, `adamw8bit`, `adamw_optimi` (recommended default), `stableadamw`, `sgd`, `adamw8bitkahan`, `offload`, `automagic`, `genericoptim`. Any other name → dynamically loaded from the `pytorch_optimizer` library **case-sensitively** (`getattr(pytorch_optimizer, type)`, e.g. `Prodigy`, `CAME`, `Tiger`). **Keep a free-text type escape hatch.**
- Name map to use: `AdamW`→`adamw_optimi`, `AdamW8bit`→`AdamW8bitKahan`, `Prodigy`→`Prodigy`, `Came`→`CAME`.
- Optimizer sub-args (`betas`, `weight_decay`, `eps`, …) are inlined into `[optimizer]`.
- `max_grad_norm` → top-level `gradient_clipping`. `learning_rate`/`unet_lr` → `[optimizer].lr`. `text_encoder_lr` → `[model].llm_adapter_lr` (closest analogue).
- **Scheduler** (train.py:869): top-level `lr_scheduler` ∈ {`constant` (default), `linear`, `cosine`} — **cosine IS available** (plan said only constant/linear). Plus `warmup_steps` (int) and `force_constant_lr` (float, overrides everything even on resume). No rex/restarts/custom schedulers. `warmup_ratio` → compute `warmup_steps` from dataset size (reuse existing logic).
- **Huber loss**: Anima's custom loss reads top-level **`huber_delta`** or **`smooth_l1_beta`** (cosmos_predict2.py:513-516), NOT `pseudo_huber_c` (that only applies to the default loss fn, which Anima replaces). Map `loss_type=huber` → `huber_delta` if we expose it at all; otherwise drop.
- **DROPPED (flow-matching / epsilon-pred concepts):** `min_snr_gamma`, `scale_weight_norms`, noise-offset family, `ip_noise_gamma`, `zero_terminal_snr`.

### dataset.toml (§4.6) — from `examples/dataset.toml` + `utils/dataset.py`
- `resolutions=[N]` (list; multi-res = multiple entries, dataset duplicated per area). AR bucketing: `enable_ar_bucket`, `min_ar` (0.5), `max_ar` (2.0), `num_ar_buckets` (7). `frame_buckets=[1]` for images.
- `[[directory]]`: `path` (= `image_dir`), `num_repeats` (same semantics as sd-scripts), optional `mask_path` (per-directory mask folder; R-channel → loss weight), and — importantly — **caption knobs CAN be per-directory OR top-level** (directory overrides top-level; dataset.py:553-598). So `keep_tokens`/`keep_tokens_separator`/`cache_shuffle_num`/`cache_shuffle_delimiter`/`enable_random_caption`/`skip_empty_caption`/`online_captions` may stay per-subset if desired, but plan's "global" approach is simplest.
- Caption shuffle: sd-scripts `shuffle_caption` (bool) → **`cache_shuffle_num`** (int N = number of pre-shuffled variants cached; 0 = off). `keep_tokens` → `keep_tokens` (int), `keep_tokens_separator` → same.
- **DROPPED:** `caption_extension` (**`.txt` fixed**, or `captions.json` via `online_captions=true`), all augmentation (`flip_aug`/`color_aug`/`random_crop`/caption dropout — **dataset.py has none**), regularization/`is_reg` (no equivalent), `bucket_no_upscale`.
- New fork features worth exposing: `enable_random_caption` (random caption variant per epoch), `skip_empty_caption` (default **true**), `online_captions`.

### sample.toml (§4.7) — from `utils/sampling.py:41-57`
- Top level: `width`, `height`, `num_inference_steps` (or `steps`), `guidance_scale` (or `cfg`), `seed`. Per prompt: `[[prompts]]` with `prompt` + `negative_prompt`.
- Cadence keys live in the **main** config: `sample_every_n_epochs`, `sample_every_n_steps`, `sample_at_first` (sampling.py:81-91). Main config also needs `sample = '<abs path to sample.toml>'`.
- No sampler selector (euler_a etc.) — **remove** from SampleUI.

### [monitoring] (§4.10) & misc
- wandb: `enable_wandb`/`wandb_api_key`/`wandb_tracker_name`/`wandb_run_name`. TensorBoard is automatic into the run dir (no toggle) — `logging_dir` UI field just becomes informational.
- `seed`: dp has **no global training-seed knob** (only per-rank eval seed + sample.toml `seed`). **Drop `seed` from the General UI** (or keep it only as the sample seed).
- `save_dtype` (from `save_precision`), `caching_batch_size`, `map_num_proc` (caching workers, from `max_data_loader_n_workers`), `compile`, `steps_per_print`, `blocks_to_swap`, `pipeline_stages`, `partition_method`, `gradient_accumulation_steps`, `activation_checkpointing` (from `gradient_checkpointing`), `epochs` (from `max_train_epochs`), `micro_batch_size_per_gpu` (from dataset `batch_size`).
- **Output format:** Anima LoRA saved as a **PEFT dir** (`adapter_model.safetensors`+`adapter_config.json`), LoKr as **`lokr.safetensors`** in **LyCORIS/ComfyUI format** (cosmos_predict2.py:321-328). Neither is a kohya-style single-file LoRA → the existing `utils/resize_lora.py` almost certainly won't apply; **gate/disable the resize UI** for dp outputs.

### New payload schema (what the frontend must send post-migration)
Keep the outer envelope (`{"args": {...}, "dataset": {...}, "accelerate": {...}}`) and the group keys (`general_args`, `anima_args`→model, `network_args`, `optimizer_args`, `saving_args`, `sample_args`, `logging_args`, `bucket_args`, subsets) so TomlFunctions save/load + queue keep working; only the **keys inside** each group change to the right-hand-column names above.

---

## 10. Phase 2 progress (frontend)

Verified headless (`QT_QPA_PLATFORM=offscreen`) by driving the real widgets → assembling the payload → running it through the backend `validate()` (round-trip harness in the scratchpad). The backend is deliberately tolerant (aliases `pretrained_model_name_or_path`/`qwen3`/`vae`, `image_dir`, `network_dim`, `max_data_loader_n_workers`; ignores unknown keys), so only a few widgets emitted *breaking* values.

**Done (correctness-critical + a couple of features):**
- `NetworkUI.py` — **rewritten.** Algo selector collapsed to LoRA / LoKr / Full fine-tune; emits a **flat** `network_args = {type, rank, ...}`; `alpha` removed entirely (dp errors on it); LoKr extras (`factor` via the repurposed DyLoRA spin, `use_tucker`, `rank_dropout`, `module_dropout`, `dropout`). Old LyCORIS/block-weight controls hidden. *(Was a silent-FFT + alpha-error breaker.)*
- `GeneralUI.py` — `timestep_sampling` combo → `logit_normal`/`uniform` only (was `sigma`/`sigmoid`… → `NotImplementedError`); `sigmoid_scale` now tied to `logit_normal`; contrastive flow → `[model].contrastive_flow_lambda`; `blocks_to_swap` moved from `anima_args` to `general_args` (dp top-level key). Model paths already alias through the backend.
- `OptimizerUI.py` — scheduler combo → `constant`/`linear`/`cosine`; custom rex/restart/polynomial branches removed (were emitting invalid `lr_scheduler` values). `optimizer_type` (mapped), `learning_rate`, inlined `optimizer_args`, `max_grad_norm`, `warmup_ratio` all flow through unchanged.
- `BucketUI.py` — **rewritten** to AR bucketing: group toggle → `enable_ar_bucket`, steps spin → `num_ar_buckets` (min/max_ar keep dp defaults; reso controls hidden).
- `MainUI.py` — multi-resolution now collapses to `dataset.general_args.resolutions = [base, …extras]` instead of the sd_scripts `{"datasets": [...]}` Format-2 the new backend can't parse. (Format-2 TOML *persistence* is frontend-internal and still round-trips.)
- `SavingUI.py`, `AccelerateUI.py` — **no change needed**; already emit compatible keys (`output_dir`/`output_name`/`save_every_n_epochs`/`save_precision`; accelerate→`num_gpus`/`master_port`).
- `SampleUI.py` — **rewritten** with custom content: an enable toggle, cadence (every-N epochs/steps + sample-at-first), width/height/steps/CFG/seed, and a **prompt list with per-prompt negatives**. Emits `sample_args` → backend writes `sample.toml` + main-config cadence keys. Sampler selector gone. Verified end-to-end (prompts → sample.toml).
- `LoggingUI.py` — log-system selector now emits `enable_wandb`; tracker/run-name inputs emit `wandb_tracker_name`/`wandb_run_name`. TensorBoard stays automatic in dp. Verified → `[monitoring]`.
- `SubsetUI.py` — the existing mask-folder input now emits per-directory **`mask_path`** (was sd_scripts `conditioning_data_dir`); gated by the OptimizerUI masked-loss toggle via the existing signal. Verified → `[[directory]].mask_path`.

**Remaining nice-to-haves (non-blocking):**
- Move `shuffle_caption`/`keep_tokens` to a global Captions group (dp reads them top-level; per-subset values are currently ignored). Drop dead per-subset aug/reg keys.
- `min_ar`/`max_ar` float controls, an `llm_adapter_lr` control, visual decluttering of the hidden controls.

---

## 11. Phase 4 + 5 progress (converter, cleanup, docs)

**Converter (`tools/convert_sdscripts_toml.py`):** pure-dict `convert(loaded) -> (new, warnings)` + `is_old_format(loaded)` detector, plus a CLI (`python tools/convert_sdscripts_toml.py old.toml [new.toml]`). Wired into `MainUI.process_toml` as an auto-fallback: old-format configs are detected, converted, and the dropped/remapped-args report is both printed and shown in a warning dialog on the UI load path. Verified with a pure-dict test (timestep/adapter/optimizer/bucket/sample/logging/subset mappings + all drops) **and** a full round-trip (old TOML → convert → load into new widgets → payload → backend `validate()`).

**Dead-code deletion:**
- Deleted `modules/BlockWeightWidgets.py` + `block_weight_presets/` (unreferenced after the NetworkUI rewrite).
- `MainUI.train_helper`: dropped the `sdxl`/`flux`/`anima`/`train_mode` query params (backend reads adapter type from the validated config); `/train` now sends only accelerate params.
- Removed the `toggle_sdxl` no-op from NetworkUI + its call in ArgsListUI.
- `MainWindow`: TensorBoard exe path and log dir → `backend/venv` + `output_dir` (were `backend/sd_scripts/venv` + `logging_dir` — a real bug).
- `backend/updater.py`: init `diffusion_pipe` (not `--recursive`), backend-level venv, no `chdir("sd_scripts")`.

**Resize feature (decided with user):** *gate, keep submodule.* `MainWindow.run_resize` now shows a "not supported for diffusion-pipe Anima output" dialog instead of launching the resize popup (dp saves LyCORIS/PEFT format that `resize_lora.py` can't process). The `sd_scripts` submodule is **kept physically** for now (its git removal deferred to a deliberate later step); `resize_lora.py`/`resize` endpoint remain but are unreachable from the UI.

**Docs (Phase 5):** README rewritten (backend, feature delta, Linux-only + preview-weights warnings, install/run, new TOML examples for LoRA/LoKr/FFT, multi-res, converter usage, credits). Window title updated. `installer.py` already points at `diffusion_pipe`/`backend/venv` from Phase 0.

**Acceptance grep:** remaining `sd_scripts`/`sdxl`/`flux` hits in `.py` are all intentional — converter module, doc comments, the kept-but-gated `resize_lora.py`, and a dead string in `LoraResizePopupUi.py` (unreachable). Generated `ui_files/*.py` still contain old widget names (harmless; the logic layer hides/repurposes them).

**Still open:** Phase 3 real-GPU integration run (can't do here — no GPU); optional physical `sd_scripts` submodule removal; the non-blocking UI nice-to-haves above.
