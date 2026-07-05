"""Convert an old sd_scripts-vocabulary saved TOML to the diffusion-pipe schema.

Saved configs use the ``{group: {"args": {...}, "dataset_args": {...}}, "subsets":
[...], "datasets": [...]}`` shape produced by ``MainUI.save_toml``. This module
translates the *keys inside* each group to the diffusion-pipe vocabulary (see
BACK-MIGRATE.md §4) and reports every dropped / remapped argument so the user
knows what was lost.

Use it two ways:

* Library:  ``new_dict, warnings = convert(loaded_dict)`` (pure dict in/out;
  ``is_old_format(loaded_dict)`` detects whether conversion is needed). Wired
  into ``MainUI.process_toml`` as an automatic fallback.
* CLI:      ``python tools/convert_sdscripts_toml.py old.toml [new.toml]``.
"""
from __future__ import annotations

import copy

# --- timestep sampling: dp Anima only knows logit_normal / uniform ----------
_TIMESTEP_MAP = {
    "sigmoid": "logit_normal",
    "sigma": "logit_normal",
    "shift": "logit_normal",
    "flux_shift": "logit_normal",
    "logit_normal": "logit_normal",
    "uniform": "uniform",
}

# anima_args keys with no diffusion-pipe equivalent
_DROPPED_ANIMA = (
    "t5_tokenizer_path", "qwen3_max_token_length", "t5_max_token_length",
    "vae_disable_cache", "split_attn", "attn_mode", "unsloth_offload_checkpointing",
    "discrete_flow_shift", "vae_chunk_size", "flow_use_ot",
)

_VALID_SCHEDULERS = {"constant", "linear", "cosine"}

# optimizer_args keys that are epsilon-prediction / sd_scripts-only concepts
_DROPPED_OPT = (
    "min_snr_gamma", "scale_weight_norms", "zero_terminal_snr", "ip_noise_gamma",
    "lr_scheduler_num_cycles", "lr_scheduler_power", "lr_scheduler_args",
    "huber_schedule", "text_encoder_lr", "masked_loss",
)

# per-subset keys with no diffusion-pipe equivalent
_DROPPED_SUBSET = (
    "caption_extension", "is_reg", "is_val", "flip_aug", "color_aug", "random_crop",
    "random_crop_padding_percent", "caption_dropout_rate",
    "caption_dropout_every_n_epochs", "caption_tag_dropout_rate", "gamma_aug",
    "gamma_aug_range", "gamma_aug_rate", "face_crop_aug_range", "token_warmup_min",
    "token_warmup_step", "shuffle_caption_sigma", "protected_tags_file",
    "target_image_dir",
)

# per-subset caption keys diffusion-pipe reads dataset-wide -> caption_args
_PROMOTED_SUBSET = ("shuffle_caption", "keep_tokens", "keep_tokens_separator")

# nested network_args (LyCORIS) keys we keep for LoKr
_KEPT_LOKR = ("factor", "use_tucker", "rank_dropout", "module_dropout", "dropout")


def is_old_format(loaded: dict) -> bool:
    """Heuristically detect an sd_scripts-vocabulary config.

    The strongest signal: the new NetworkUI always emits ``network_args.args``
    with a ``type`` key, while the old one emitted ``network_dim`` / a nested
    ``network_args`` table instead.
    """
    if not isinstance(loaded, dict):
        return False
    net_args = loaded.get("network_args", {})
    net_args = net_args.get("args") if isinstance(net_args, dict) else None
    if isinstance(net_args, dict):
        # The new NetworkUI always emits `type`, so its presence is definitive:
        # never convert such a config, even if stray legacy keys (old anima
        # fields, [[datasets]] persistence) appear elsewhere in the file.
        if "type" in net_args:
            return False
        if net_args:
            return True
    anima = _group(loaded, "anima_args", "args")
    if any(k in anima for k in ("t5_tokenizer_path", "qwen3_max_token_length")):
        return True
    ts = anima.get("timestep_sampling")
    if ts and ts not in ("logit_normal", "uniform"):
        return True
    if _group(loaded, "sample_args", "args").get("sample_sampler"):
        return True
    if "datasets" in loaded:
        return True
    return False


def _group(loaded: dict, group: str, sub: str) -> dict:
    body = loaded.get(group, {})
    if not isinstance(body, dict):
        return {}
    inner = body.get(sub, {})
    return inner if isinstance(inner, dict) else {}


def _convert_subset(subset: dict, warnings: set, caption: dict) -> dict:
    out = dict(subset)
    if "conditioning_data_dir" in out:
        out["mask_path"] = out.pop("conditioning_data_dir")
    for key in _DROPPED_SUBSET:
        if key in out:
            out.pop(key)
            warnings.add(f"subset: dropped '{key}'")
    # dp reads caption knobs dataset-wide; promote the strongest per-subset
    # values into the global Captions group.
    if out.pop("shuffle_caption", False):
        caption["shuffle_caption"] = True
        warnings.add("captions: per-subset shuffle_caption promoted to global Captions")
    keep = out.pop("keep_tokens", 0)
    if keep and keep > caption.get("keep_tokens", 0):
        caption["keep_tokens"] = keep
        warnings.add("captions: per-subset keep_tokens promoted to global Captions")
    separator = out.pop("keep_tokens_separator", "")
    if separator and not caption.get("keep_tokens_separator"):
        caption["keep_tokens_separator"] = separator
        warnings.add("captions: per-subset keep_tokens_separator promoted to global Captions")
    return out


def convert(loaded: dict) -> tuple[dict, list[str]]:
    """Return (new_config, warnings). ``loaded`` is not mutated."""
    new = copy.deepcopy(loaded)
    warnings: set[str] = set()

    def ensure(group: str, sub: str) -> dict:
        body = new.setdefault(group, {})
        if not isinstance(body, dict):
            body = new[group] = {}
        return body.setdefault(sub, {})

    # ---- anima_args -> [model] ----
    anima = ensure("anima_args", "args") if "anima_args" in new else {}
    if anima:
        ts = anima.get("timestep_sampling")
        if ts and ts not in ("logit_normal", "uniform"):
            anima["timestep_sampling"] = _TIMESTEP_MAP.get(str(ts).lower(), "logit_normal")
            warnings.add(f"timestep_sampling '{ts}' -> '{anima['timestep_sampling']}'")
        for key in _DROPPED_ANIMA:
            if key in anima:
                anima.pop(key)
                warnings.add(f"model: dropped '{key}'")
        if anima.pop("contrastive_flow_matching", False):
            anima["contrastive_flow_lambda"] = anima.pop("cfm_lambda", 0.02)
        else:
            anima.pop("cfm_lambda", None)
        if "blocks_to_swap" in anima:
            ensure("general_args", "args")["blocks_to_swap"] = anima.pop("blocks_to_swap")

    # ---- network_args -> [adapter] (flat) ----
    net = _group(loaded, "network_args", "args")
    if net:
        nested = net.get("network_args", {}) or {}
        algo = str(nested.get("algo", "lora")).lower()
        if algo == "full":
            adapter_type = "none"
        elif algo == "lokr":
            adapter_type = "lokr"
        elif algo in ("lora", "locon", ""):
            adapter_type = "lora"
        else:
            adapter_type = "lora"
            warnings.add(f"adapter algo '{algo}' unsupported -> LoRA")
        new_net: dict = {"type": adapter_type}
        if adapter_type != "none":
            new_net["rank"] = net.get("network_dim", 32)
            if adapter_type == "lokr":
                for key in _KEPT_LOKR:
                    if key in nested:
                        new_net[key] = nested[key]
            elif "dropout" in nested:
                new_net["dropout"] = nested["dropout"]
        if "network_alpha" in net:
            warnings.add("adapter: dropped 'network_alpha' (dp forces alpha = rank)")
        if net.get("fa"):
            warnings.add("adapter: dropped LoRA-FA (unsupported)")
        for key in ("min_timestep", "max_timestep"):
            if key in net:
                warnings.add(f"adapter: dropped '{key}'")
        for key in nested:
            if key not in ("algo",) + _KEPT_LOKR:
                warnings.add(f"adapter: dropped LyCORIS arg '{key}'")
        new["network_args"]["args"] = new_net

    # ---- optimizer_args -> [optimizer] + scheduler ----
    opt = ensure("optimizer_args", "args") if "optimizer_args" in new else {}
    if opt:
        opt_type = opt.get("optimizer_type")
        if isinstance(opt_type, str) and "." in opt_type:
            opt["optimizer_type"] = opt_type.rsplit(".", 1)[-1]
            warnings.add(f"optimizer_type '{opt_type}' -> '{opt['optimizer_type']}'")
        sched = str(opt.get("lr_scheduler", "")).lower().replace(" ", "_")
        if "lr_scheduler_type" in opt:
            opt.pop("lr_scheduler_type")
            opt["lr_scheduler"] = "constant"
            warnings.add("scheduler: custom lr_scheduler_type -> constant (no dp equivalent)")
        elif sched and sched not in _VALID_SCHEDULERS:
            mapped = "cosine" if "cosine" in sched else "constant"
            opt["lr_scheduler"] = mapped
            warnings.add(f"scheduler '{sched}' -> '{mapped}'")
        for key in _DROPPED_OPT:
            if key in opt:
                opt.pop(key)
                warnings.add(f"optimizer: dropped '{key}'")

    # ---- bucket_args -> AR bucketing ----
    bucket = ensure("bucket_args", "dataset_args") if "bucket_args" in new else {}
    if bucket:
        if "enable_bucket" in bucket:
            bucket["enable_ar_bucket"] = bucket.pop("enable_bucket")
        bucket.setdefault("num_ar_buckets", 7)
        for key in ("min_bucket_reso", "max_bucket_reso", "bucket_reso_steps",
                    "bucket_no_upscale", "multires_training"):
            if key in bucket:
                bucket.pop(key)
                warnings.add(f"bucket: dropped '{key}' (reso-step -> aspect-ratio bucketing)")

    # ---- sample_args ----
    smp = ensure("sample_args", "args") if "sample_args" in new else {}
    if smp:
        if smp.pop("sample_sampler", None) is not None:
            warnings.add("sample: dropped sampler selector (no dp equivalent)")
        if "sample_prompts" in smp:
            smp.pop("sample_prompts")
            warnings.add("sample: dropped prompt .txt path — re-enter prompts in the Sample panel")

    # ---- logging_args -> [monitoring] ----
    log = ensure("logging_args", "args") if "logging_args" in new else {}
    if log:
        if "log_tracker_name" in log:
            log["wandb_tracker_name"] = log.pop("log_tracker_name")
        if "run_name" in log:
            log["wandb_run_name"] = log.pop("run_name")

    # ---- subsets (+ Format-2 datasets) ----
    caption: dict = {}
    if isinstance(new.get("subsets"), list):
        new["subsets"] = [_convert_subset(s, warnings, caption) for s in new["subsets"] if isinstance(s, dict)]
    if isinstance(new.get("datasets"), list):
        warnings.add("multi-resolution [[datasets]] preserved (collapses to a resolutions list at train time)")
        for dataset in new["datasets"]:
            if isinstance(dataset, dict) and isinstance(dataset.get("subsets"), list):
                dataset["subsets"] = [_convert_subset(s, warnings, caption) for s in dataset["subsets"] if isinstance(s, dict)]

    # ---- global caption knobs -> caption_args ----
    general = ensure("general_args", "args") if "general_args" in new else {}
    if general.pop("seed", None) is not None:
        warnings.add("general: dropped 'seed' (dp has no training seed; the sample seed lives in the Sample panel)")
    separator = general.pop("keep_tokens_separator", "")
    if separator and not caption.get("keep_tokens_separator"):
        caption["keep_tokens_separator"] = separator
        warnings.add("captions: keep_tokens_separator moved to the global Captions group")
    if caption:
        ensure("caption_args", "dataset_args").update(caption)

    return new, sorted(warnings)


def _main() -> int:
    import sys
    from pathlib import Path
    import toml

    if len(sys.argv) < 2:
        print("usage: python tools/convert_sdscripts_toml.py <old.toml> [new.toml]")
        return 2
    src = Path(sys.argv[1])
    dst = Path(sys.argv[2]) if len(sys.argv) > 2 else src.with_stem(src.stem + "_diffusion_pipe")
    loaded = toml.loads(src.read_text())
    if not is_old_format(loaded):
        print(f"{src} does not look like an old sd_scripts config (nothing to convert).")
        return 0
    new, warnings = convert(loaded)
    dst.write_text(toml.dumps(new))
    print(f"Converted {src} -> {dst}")
    if warnings:
        print(f"\n{len(warnings)} arg(s) dropped or remapped:")
        for warning in warnings:
            print(f"  - {warning}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
