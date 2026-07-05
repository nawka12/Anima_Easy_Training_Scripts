from pathlib import Path

from PySide6.QtCore import Signal
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QPushButton, QWidget

from modules.BaseWidget import BaseWidget
from modules.DragDropLineEdit import DragDropLineEdit
from ui_files.BaseUI import Ui_base_args_ui


# Anima-specific keys live under [anima_args] in the saved toml.
ANIMA_KEYS: tuple[str, ...] = (
    "pretrained_model_name_or_path",
    "qwen3",
    "vae",
    "t5_tokenizer_path",
    "qwen3_max_token_length",
    "t5_max_token_length",
    "timestep_sampling",
    "discrete_flow_shift",
    "sigmoid_scale",
    "vae_chunk_size",
    "vae_disable_cache",
    "blocks_to_swap",
    "attn_mode",
    "split_attn",
    "unsloth_offload_checkpointing",
    "flow_use_ot",
    "contrastive_flow_matching",
    "cfm_lambda",
)


class GeneralWidget(BaseWidget):
    cacheLatentsChecked = Signal(bool)
    keepTokensSepChecked = Signal(bool)

    def __init__(self, parent: QWidget = None) -> None:
        super().__init__(parent)
        self.colap.set_title("General Args")
        self.widget = Ui_base_args_ui()

        self.name = "general_args"
        self.args = {}
        self.dataset_args = {}
        self.anima_args = {}

        self.setup_widget()
        self.setup_connections()

    # ---------------- helpers ----------------

    def edit_anima_args(self, name: str, value: object, optional: bool = False) -> None:
        if name in self.anima_args:
            del self.anima_args[name]
        if optional and (value is None or value is False or value == "" or value == 0):
            return
        self.anima_args[name] = value

    # ---------------- setup ----------------

    def setup_widget(self) -> None:
        super().setup_widget()
        self.widget.setupUi(self.content)

        # diffusion-pipe's Anima only supports these two timestep sample methods.
        self.widget.timestep_sampling_selector.blockSignals(True)
        self.widget.timestep_sampling_selector.clear()
        self.widget.timestep_sampling_selector.addItems(["logit_normal", "uniform"])
        self.widget.timestep_sampling_selector.setCurrentText("logit_normal")
        self.widget.timestep_sampling_selector.blockSignals(False)

        more_icon = QIcon(str(Path("icons/more-horizontal.svg")))

        def setup_file(elem: DragDropLineEdit, selector: QPushButton, exts: list[str], allow_empty: bool = False):
            elem.setMode("file", exts)
            elem.highlight = True
            elem.allow_empty = allow_empty
            selector.setIcon(more_icon)

        def setup_folder(elem: DragDropLineEdit, selector: QPushButton, allow_empty: bool = True):
            elem.setMode("folder")
            elem.highlight = True
            elem.allow_empty = allow_empty
            selector.setIcon(more_icon)

        model_exts = [".ckpt", ".pt", ".safetensors", ".sft", ".pth"]
        setup_file(self.widget.base_model_input, self.widget.base_model_selector, model_exts)
        setup_file(self.widget.qwen3_model_input, self.widget.qwen3_model_selector, model_exts)
        setup_file(self.widget.vae_input, self.widget.vae_selector, model_exts)
        setup_folder(self.widget.t5_tokenizer_input, self.widget.t5_tokenizer_selector)

        self.widget.global_protected_tags_file_input.setMode("file", [".txt"])
        self.widget.global_protected_tags_file_input.highlight = True
        self.widget.global_protected_tags_file_input.allow_empty = True
        self.widget.global_protected_tags_file_selector.setIcon(more_icon)

        # Seed initial general args
        self.args["seed"] = int(self.widget.seed_input.value())
        self.args["max_train_epochs"] = self.widget.max_train_input.value()
        self.args["max_data_loader_n_workers"] = self.widget.max_data_loader_n_workers_input.value()
        self.args["persistent_data_loader_workers"] = True
        mixed_prec_text = self.widget.mixed_precision_selector.currentText()
        self.args["mixed_precision"] = mixed_prec_text if mixed_prec_text != "float" else "no"

        self.dataset_args["resolution"] = self.widget.width_input.value()
        self.dataset_args["batch_size"] = self.widget.batch_size_input.value()

        # Seed initial Anima args
        self.edit_anima_args("qwen3_max_token_length", self.widget.qwen3_max_token_input.value())
        self.edit_anima_args("t5_max_token_length", self.widget.t5_max_token_input.value())
        self.edit_anima_args("timestep_sampling", self.widget.timestep_sampling_selector.currentText())
        self._sync_sigmoid_scale()
        self._sync_discrete_flow_shift()
        self._sync_vae_chunk()
        self._sync_blocks_to_swap()
        self._sync_flash_attn()
        self._sync_flow_use_ot()
        self._sync_contrastive_flow_matching()

    def setup_connections(self) -> None:
        # Anima model files
        self.widget.base_model_input.textChanged.connect(
            lambda x: self.edit_anima_args("pretrained_model_name_or_path", x)
        )
        self.widget.base_model_selector.clicked.connect(
            lambda: self.set_file_from_dialog(self.widget.base_model_input, "Anima DiT Model", "Model file")
        )
        self.widget.qwen3_model_input.textChanged.connect(
            lambda x: self.edit_anima_args("qwen3", x)
        )
        self.widget.qwen3_model_selector.clicked.connect(
            lambda: self.set_file_from_dialog(self.widget.qwen3_model_input, "Qwen3 Model", "Model file")
        )
        self.widget.vae_input.textChanged.connect(
            lambda x: self.edit_anima_args("vae", x)
        )
        self.widget.vae_selector.clicked.connect(
            lambda: self.set_file_from_dialog(self.widget.vae_input, "VAE Model", "Model file")
        )
        self.widget.t5_tokenizer_input.textChanged.connect(
            lambda x: self.edit_anima_args("t5_tokenizer_path", x, optional=True)
        )
        self.widget.t5_tokenizer_selector.clicked.connect(
            lambda: self.set_folder_from_dialog(self.widget.t5_tokenizer_input, "T5 Tokenizer Folder")
        )

        # Global args
        self.widget.global_protected_tags_file_input.textChanged.connect(
            lambda x: self.edit_args("protected_tags_file", x, optional=True)
        )
        self.widget.global_protected_tags_file_selector.clicked.connect(
            lambda: self.set_file_from_dialog(
                self.widget.global_protected_tags_file_input, "Protected Tags File (Global)", "Text file"
            )
        )
        self.widget.global_protected_tags_file_enable.clicked.connect(self.enable_disable_global_protected_tags)

        # Precision flags
        self.widget.no_half_vae_enable.clicked.connect(lambda x: self.edit_args("no_half_vae", x, True))
        self.widget.low_ram_enable.clicked.connect(lambda x: self.edit_args("lowram", x, True))
        self.widget.high_vram_enable.clicked.connect(lambda x: self.edit_args("highvram", x, True))
        self.widget.FP16_enable.clicked.connect(lambda x: self.change_full_type(x, False))
        self.widget.BF16_enable.clicked.connect(lambda x: self.change_full_type(False, x))

        # Resolution
        self.widget.width_input.valueChanged.connect(self.change_resolution)
        self.widget.height_enable.clicked.connect(self.change_resolution)
        self.widget.height_input.valueChanged.connect(self.change_resolution)

        # Gradient
        self.widget.grad_checkpointing_enable.clicked.connect(
            lambda x: self.edit_args("gradient_checkpointing", x, True)
        )
        self.widget.grad_accumulation_enable.clicked.connect(self.enable_disable_grad_acc)
        self.widget.grad_accumulation_input.valueChanged.connect(
            lambda x: self.edit_args("gradient_accumulation_steps", x, True)
        )

        # Core
        self.widget.max_data_loader_n_workers_input.valueChanged.connect(
            lambda x: self.edit_args("max_data_loader_n_workers", x)
        )
        self.widget.seed_input.valueChanged.connect(lambda x: self.edit_args("seed", int(x)))
        self.widget.batch_size_input.valueChanged.connect(lambda x: self.edit_dataset_args("batch_size", x))
        self.widget.mixed_precision_selector.currentTextChanged.connect(
            lambda x: self.edit_args("mixed_precision", x if x != "float" else "no")
        )
        self.widget.xformers_enable.clicked.connect(lambda x: self.change_optim_type(x, False))
        self.widget.sdpa_enable.clicked.connect(lambda x: self.change_optim_type(False, x))
        self.widget.max_train_selector.currentIndexChanged.connect(self.change_max_mode)
        self.widget.max_train_input.valueChanged.connect(
            lambda: self.change_max_mode(self.widget.max_train_selector.currentIndex())
        )

        # Cache latents / keep tokens / comment
        self.widget.cache_latents_enable.clicked.connect(self.enable_disable_cache_latents)
        self.widget.cache_latents_to_disk_enable.clicked.connect(
            lambda x: self.edit_args("cache_latents_to_disk", x, True)
        )
        self.widget.keep_tokens_seperator_enable.clicked.connect(self.enable_disable_keep_tokens_sep)
        self.widget.keep_tokens_seperator_input.textChanged.connect(
            lambda x: self.edit_args("keep_tokens_separator", x, optional=True)
        )
        self.widget.comment_enable.clicked.connect(self.enable_disable_comment)
        self.widget.comment_input.textChanged.connect(
            lambda: self.edit_args("training_comment", self.widget.comment_input.toPlainText(), True)
        )

        # Anima sampling
        self.widget.timestep_sampling_selector.currentTextChanged.connect(self.change_timestep_sampling)
        self.widget.discrete_flow_shift_input.valueChanged.connect(
            lambda _: self._sync_discrete_flow_shift()
        )
        self.widget.sigmoid_scale_input.valueChanged.connect(lambda _: self._sync_sigmoid_scale())
        self.widget.qwen3_max_token_input.valueChanged.connect(
            lambda x: self.edit_anima_args("qwen3_max_token_length", x)
        )
        self.widget.t5_max_token_input.valueChanged.connect(
            lambda x: self.edit_anima_args("t5_max_token_length", x)
        )

        # Anima memory / attention
        self.widget.vae_chunk_size_input.valueChanged.connect(lambda _: self._sync_vae_chunk())
        self.widget.vae_disable_cache_enable.clicked.connect(
            lambda x: self.edit_anima_args("vae_disable_cache", x, optional=True)
        )
        self.widget.blocks_to_swap_input.valueChanged.connect(lambda _: self._sync_blocks_to_swap())
        self.widget.flash_attn_enable.clicked.connect(lambda _: self._sync_flash_attn())
        self.widget.split_attn_enable.clicked.connect(
            lambda x: self.edit_anima_args("split_attn", x, optional=True)
        )
        self.widget.unsloth_offload_checkpointing.clicked.connect(
            lambda x: self.edit_anima_args("unsloth_offload_checkpointing", x, optional=True)
        )

        # Anima flow matching
        self.widget.flow_use_ot_enable.clicked.connect(lambda _: self._sync_flow_use_ot())
        self.widget.contrastive_flow_matching_enable.clicked.connect(
            lambda _: self._sync_contrastive_flow_matching()
        )
        self.widget.cfm_lambda_input.valueChanged.connect(lambda _: self._sync_contrastive_flow_matching())

    # ---------------- handlers ----------------

    def change_full_type(self, is_fp: bool, is_bf: bool) -> None:
        for arg in ["full_fp16", "full_bf16", "mixed_precision"]:
            if arg in self.args:
                del self.args[arg]
        self.widget.FP16_enable.setEnabled(not is_bf)
        self.widget.BF16_enable.setEnabled(not is_fp)
        self.widget.mixed_precision_selector.setEnabled(not is_bf and not is_fp)

        self.edit_args("full_fp16", is_fp, True)
        self.edit_args("full_bf16", is_bf, True)
        text = self.widget.mixed_precision_selector.currentText()
        self.edit_args(
            "mixed_precision",
            "fp16" if is_fp else "bf16" if is_bf else text if text != "float" else "no",
        )

    def change_resolution(self) -> None:
        if "resolution" in self.dataset_args:
            del self.dataset_args["resolution"]
        if not self.widget.height_enable.isChecked():
            self.widget.height_input.setEnabled(False)
            self.edit_dataset_args("resolution", self.widget.width_input.value())
            return
        self.widget.height_input.setEnabled(True)
        self.edit_dataset_args(
            "resolution",
            [self.widget.width_input.value(), self.widget.height_input.value()],
        )

    def change_optim_type(self, is_xformers: bool, is_sdpa: bool) -> None:
        for arg in ["xformers", "sdpa"]:
            if arg in self.args:
                del self.args[arg]
        self.widget.xformers_enable.setEnabled(not is_sdpa)
        self.widget.sdpa_enable.setEnabled(not is_xformers)
        self.edit_args("xformers", is_xformers, True)
        self.edit_args("sdpa", is_sdpa, True)
        # When xFormers is selected, Anima requires split_attn to be on.
        if is_xformers:
            self.widget.split_attn_enable.setChecked(True)
            self.widget.split_attn_enable.setEnabled(False)
            self.edit_anima_args("split_attn", True, optional=True)
        else:
            self.widget.split_attn_enable.setEnabled(True)
            self.edit_anima_args(
                "split_attn", self.widget.split_attn_enable.isChecked(), optional=True
            )

    def change_max_mode(self, index: int) -> None:
        args = ["max_train_epochs", "max_train_steps"]
        for arg in args:
            if arg in self.args:
                del self.args[arg]
        self.edit_args(args[index], self.widget.max_train_input.value())

    def enable_disable_grad_acc(self, checked: bool) -> None:
        if "gradient_accumulation_steps" in self.args:
            del self.args["gradient_accumulation_steps"]
        self.widget.grad_accumulation_input.setEnabled(checked)
        if checked:
            self.edit_args(
                "gradient_accumulation_steps",
                self.widget.grad_accumulation_input.value(),
                True,
            )

    def enable_disable_cache_latents(self, checked: bool) -> None:
        for arg in ["cache_latents", "cache_latents_to_disk"]:
            if arg in self.args:
                del self.args[arg]
        self.widget.cache_latents_to_disk_enable.setEnabled(checked)
        self.edit_args("cache_latents", checked, True)
        self.edit_args(
            "cache_latents_to_disk",
            self.widget.cache_latents_to_disk_enable.isChecked() and checked,
            True,
        )
        self.cacheLatentsChecked.emit(checked)

    def enable_disable_keep_tokens_sep(self, checked: bool) -> None:
        if "keep_tokens_separator" in self.args:
            del self.args["keep_tokens_separator"]
        self.widget.keep_tokens_seperator_input.setEnabled(checked)
        self.keepTokensSepChecked.emit(checked)
        if checked:
            self.edit_args(
                "keep_tokens_separator",
                self.widget.keep_tokens_seperator_input.text(),
                optional=True,
            )

    def enable_disable_comment(self, checked: bool) -> None:
        if "training_comment" in self.args:
            del self.args["training_comment"]
        self.widget.comment_input.setEnabled(checked)
        if checked:
            self.edit_args("training_comment", self.widget.comment_input.toPlainText(), True)

    def enable_disable_global_protected_tags(self, checked: bool) -> None:
        if "protected_tags_file" in self.args:
            del self.args["protected_tags_file"]
        self.widget.global_protected_tags_file_input.setEnabled(checked)
        self.widget.global_protected_tags_file_selector.setEnabled(checked)
        if checked:
            self.edit_args(
                "protected_tags_file", self.widget.global_protected_tags_file_input.text(), optional=True
            )

    def change_timestep_sampling(self, _text: str = "") -> None:
        sampling_type = self.widget.timestep_sampling_selector.currentText()
        self.edit_anima_args("timestep_sampling", sampling_type)
        # sigmoid_scale scales the logit-normal distribution; N/A for uniform.
        self.widget.sigmoid_scale_input.setEnabled(sampling_type == "logit_normal")
        self.widget.discrete_flow_shift_input.setEnabled(False)
        self._sync_sigmoid_scale()
        self._sync_discrete_flow_shift()

    def _sync_sigmoid_scale(self) -> None:
        if self.widget.timestep_sampling_selector.currentText() == "logit_normal":
            self.edit_anima_args("sigmoid_scale", self.widget.sigmoid_scale_input.value())
        elif "sigmoid_scale" in self.anima_args:
            del self.anima_args["sigmoid_scale"]

    def _sync_discrete_flow_shift(self) -> None:
        # diffusion-pipe's Anima has no discrete_flow_shift knob (it uses
        # shift / flux_shift instead); drop the legacy key.
        if "discrete_flow_shift" in self.anima_args:
            del self.anima_args["discrete_flow_shift"]

    def _sync_vae_chunk(self) -> None:
        value = self.widget.vae_chunk_size_input.value()
        if value > 0:
            self.edit_anima_args("vae_chunk_size", value)
        elif "vae_chunk_size" in self.anima_args:
            del self.anima_args["vae_chunk_size"]

    def _sync_blocks_to_swap(self) -> None:
        # blocks_to_swap is a top-level diffusion-pipe key, so it lives in
        # general_args, not [model]. 0 = disabled (omitted).
        if "blocks_to_swap" in self.anima_args:
            del self.anima_args["blocks_to_swap"]
        self.edit_args("blocks_to_swap", self.widget.blocks_to_swap_input.value(), optional=True)

    def _sync_flash_attn(self) -> None:
        if self.widget.flash_attn_enable.isChecked():
            self.edit_anima_args("attn_mode", "flash")
        elif "attn_mode" in self.anima_args:
            del self.anima_args["attn_mode"]

    def _sync_flow_use_ot(self) -> None:
        self.edit_anima_args(
            "flow_use_ot", self.widget.flow_use_ot_enable.isChecked(), optional=True
        )

    def _sync_contrastive_flow_matching(self) -> None:
        # diffusion-pipe exposes contrastive flow matching as a single
        # [model].contrastive_flow_lambda float (0 = off).
        enabled = self.widget.contrastive_flow_matching_enable.isChecked()
        self.widget.cfm_lambda_input.setEnabled(enabled)
        for legacy in ("contrastive_flow_matching", "cfm_lambda"):
            if legacy in self.anima_args:
                del self.anima_args[legacy]
        if enabled:
            self.edit_anima_args("contrastive_flow_lambda", self.widget.cfm_lambda_input.value())
        elif "contrastive_flow_lambda" in self.anima_args:
            del self.anima_args["contrastive_flow_lambda"]

    # ---------------- load/save ----------------

    def get_anima_args(self) -> dict:
        return dict(self.anima_args)

    def load_args(self, args: dict) -> bool:
        all_args = args
        general = all_args.get(self.name, {})
        anima = all_args.get("anima_args", {})

        # General
        self.widget.no_half_vae_enable.setChecked(general.get("no_half_vae", False))
        self.widget.low_ram_enable.setChecked(general.get("lowram", False))
        self.widget.high_vram_enable.setChecked(general.get("highvram", False))
        self.widget.FP16_enable.setChecked(general.get("full_fp16", False))
        self.widget.BF16_enable.setChecked(general.get("full_bf16", False))
        self.widget.grad_checkpointing_enable.setChecked(general.get("gradient_checkpointing", False))
        self.widget.grad_accumulation_enable.setChecked(bool(general.get("gradient_accumulation_steps", False)))
        self.widget.grad_accumulation_input.setValue(general.get("gradient_accumulation_steps", 1))
        self.widget.seed_input.setValue(int(general.get("seed", 42)))
        self.widget.max_data_loader_n_workers_input.setValue(general.get("max_data_loader_n_workers", 1))

        mixed_prec = general.get("mixed_precision", "fp16")
        self.widget.mixed_precision_selector.setCurrentText(mixed_prec if mixed_prec != "no" else "float")
        self.widget.xformers_enable.setChecked(general.get("xformers", False))
        self.widget.sdpa_enable.setChecked(general.get("sdpa", False))
        self.widget.max_train_selector.setCurrentIndex(0 if general.get("max_train_epochs", None) else 1)
        self.widget.max_train_input.setValue(general.get("max_train_epochs", general.get("max_train_steps", 1)))
        self.widget.cache_latents_enable.setChecked(general.get("cache_latents", False))
        self.widget.cache_latents_to_disk_enable.setChecked(general.get("cache_latents_to_disk", False))
        self.widget.keep_tokens_seperator_enable.setChecked(bool(general.get("keep_tokens_separator", False)))
        self.widget.keep_tokens_seperator_input.setText(general.get("keep_tokens_separator", ""))
        self.widget.comment_enable.setChecked(bool(general.get("training_comment", False)))
        self.widget.comment_input.setText(general.get("training_comment", ""))
        self.widget.global_protected_tags_file_enable.setChecked(bool(general.get("protected_tags_file", False)))
        self.widget.global_protected_tags_file_input.setText(general.get("protected_tags_file", ""))

        # Anima-specific (fall back to legacy general keys for older tomls)
        def pick(key: str, default):
            if key in anima:
                return anima[key]
            return general.get(key, default)

        self.widget.base_model_input.setText(pick("pretrained_model_name_or_path", ""))
        self.widget.qwen3_model_input.setText(pick("qwen3", ""))
        self.widget.vae_input.setText(pick("vae", ""))
        self.widget.t5_tokenizer_input.setText(pick("t5_tokenizer_path", ""))
        self.widget.qwen3_max_token_input.setValue(pick("qwen3_max_token_length", 512))
        self.widget.t5_max_token_input.setValue(pick("t5_max_token_length", 512))
        self.widget.timestep_sampling_selector.setCurrentText(pick("timestep_sampling", "logit_normal"))
        self.widget.discrete_flow_shift_input.setValue(pick("discrete_flow_shift", 3.0))
        self.widget.sigmoid_scale_input.setValue(pick("sigmoid_scale", 1.0))
        self.widget.vae_chunk_size_input.setValue(pick("vae_chunk_size", 0))
        self.widget.vae_disable_cache_enable.setChecked(pick("vae_disable_cache", False))
        # blocks_to_swap moved to general_args (top-level dp key); fall back to legacy anima key.
        self.widget.blocks_to_swap_input.setValue(general.get("blocks_to_swap", pick("blocks_to_swap", 0)))
        self.widget.flash_attn_enable.setChecked(pick("attn_mode", "") == "flash")
        self.widget.split_attn_enable.setChecked(pick("split_attn", False))
        self.widget.unsloth_offload_checkpointing.setChecked(pick("unsloth_offload_checkpointing", False))
        self.widget.flow_use_ot_enable.setChecked(pick("flow_use_ot", True))
        cfm_lambda = pick("contrastive_flow_lambda", pick("cfm_lambda", 0.0))
        self.widget.contrastive_flow_matching_enable.setChecked(bool(cfm_lambda))
        self.widget.cfm_lambda_input.setValue(cfm_lambda or 0.02)

        # Re-sync internal args dicts
        self.change_full_type(self.widget.FP16_enable.isChecked(), self.widget.BF16_enable.isChecked())
        self.edit_args("no_half_vae", self.widget.no_half_vae_enable.isChecked(), True)
        self.edit_args("lowram", self.widget.low_ram_enable.isChecked(), True)
        self.edit_args("highvram", self.widget.high_vram_enable.isChecked(), True)
        self.edit_args("gradient_checkpointing", self.widget.grad_checkpointing_enable.isChecked(), True)
        self.enable_disable_grad_acc(self.widget.grad_accumulation_enable.isChecked())
        self.edit_args("seed", int(self.widget.seed_input.value()))
        self.edit_args("max_data_loader_n_workers", self.widget.max_data_loader_n_workers_input.value())
        self.change_optim_type(self.widget.xformers_enable.isChecked(), self.widget.sdpa_enable.isChecked())
        self.change_max_mode(self.widget.max_train_selector.currentIndex())
        self.enable_disable_cache_latents(self.widget.cache_latents_enable.isChecked())
        self.enable_disable_keep_tokens_sep(self.widget.keep_tokens_seperator_enable.isChecked())
        self.enable_disable_comment(self.widget.comment_enable.isChecked())
        self.enable_disable_global_protected_tags(self.widget.global_protected_tags_file_enable.isChecked())

        # Re-sync Anima args
        self.edit_anima_args("pretrained_model_name_or_path", self.widget.base_model_input.text())
        self.edit_anima_args("qwen3", self.widget.qwen3_model_input.text())
        self.edit_anima_args("vae", self.widget.vae_input.text())
        self.edit_anima_args("t5_tokenizer_path", self.widget.t5_tokenizer_input.text(), optional=True)
        self.edit_anima_args("qwen3_max_token_length", self.widget.qwen3_max_token_input.value())
        self.edit_anima_args("t5_max_token_length", self.widget.t5_max_token_input.value())
        self.change_timestep_sampling()
        self._sync_vae_chunk()
        self.edit_anima_args(
            "vae_disable_cache", self.widget.vae_disable_cache_enable.isChecked(), optional=True
        )
        self._sync_blocks_to_swap()
        self._sync_flash_attn()
        self.edit_anima_args(
            "unsloth_offload_checkpointing",
            self.widget.unsloth_offload_checkpointing.isChecked(),
            optional=True,
        )
        self._sync_flow_use_ot()
        self._sync_contrastive_flow_matching()

        return True

    def load_dataset_args(self, dataset_args: dict) -> bool:
        dataset_args = dataset_args.get(self.name, {})
        resolution = dataset_args.get("resolution", 1024)
        self.widget.width_input.setValue(resolution[0] if isinstance(resolution, list) else resolution)
        self.widget.height_enable.setChecked(isinstance(resolution, list))
        self.widget.height_input.setValue(resolution[1] if isinstance(resolution, list) else resolution)
        self.widget.batch_size_input.setValue(dataset_args.get("batch_size", 1))

        self.change_resolution()
        self.edit_dataset_args("batch_size", self.widget.batch_size_input.value())
        return True
