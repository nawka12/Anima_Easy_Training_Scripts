from PySide6.QtWidgets import QWidget
from ui_files.NetworkUI import Ui_network_ui
from modules.BaseWidget import BaseWidget


# diffusion-pipe only supports LoRA and LoKr adapters; omitting the adapter
# entirely is a full fine-tune. The UI collapses the old LyCORIS algo zoo to
# these three choices, and emits a FLAT network_args group:
#   {"type": "lora"|"lokr", "rank": N, ...lokr extras...}
# or {"type": "none"} for full fine-tune. Alpha is intentionally absent:
# diffusion-pipe forces alpha == rank and errors if alpha is provided.
ADAPTER_CHOICES = ["LoRA", "LoKr", "Full fine-tune"]


class NetworkWidget(BaseWidget):
    def __init__(self, parent: QWidget = None) -> None:
        super().__init__(parent)
        self.colap.set_title("Network Args (Adapter)")
        self.widget = Ui_network_ui()

        self.name = "network_args"
        self.args = {"type": "lora", "rank": 32}

        self.setup_widget()
        self.setup_connections()
        self._sync_adapter()

    def setup_widget(self) -> None:
        super().setup_widget()
        self.widget.setupUi(self.content)

        # Collapse the algo selector to the three diffusion-pipe choices.
        self.widget.algo_select.blockSignals(True)
        self.widget.algo_select.clear()
        self.widget.algo_select.addItems(ADAPTER_CHOICES)
        self.widget.algo_select.setCurrentIndex(0)
        self.widget.algo_select.blockSignals(False)

        # Repurpose the DyLoRA unit spinbox as the LoKr "factor" control
        # (-1 = auto). Everything else in the old LyCORIS UI is unused now.
        self.widget.dylora_unit_input.setMinimum(-1)
        self.widget.dylora_unit_input.setValue(-1)
        factor_tip = (
            "<html><body><p>Kronecker factor for LoKr. -1 = auto (most square split, "
            "smallest adapter). With an oversized rank (full-matrix mode) this is what "
            "controls the adapter's capacity/size: smaller factor = larger adapter "
            "(16 is a common starting point, 8 for more capacity).</p></body></html>"
        )
        self.widget.dylora_unit_label.setText("LoKr Factor")
        self.widget.dylora_unit_label.setToolTip(factor_tip)
        self.widget.dylora_unit_input.setToolTip(factor_tip)

        # Remove the tabs that no longer map to anything in diffusion-pipe;
        # hiding the page widget alone leaves an empty, clickable tab behind.
        for tab in (self.widget.block_weight_tab, self.widget.network_args_tab):
            index = self.widget.tabWidget.indexOf(tab)
            if index != -1:
                self.widget.tabWidget.removeTab(index)

        # Hide the controls that no longer map to anything in diffusion-pipe.
        for name in (
            "lycoris_preset_input", "conv_dim_input", "conv_alpha_input",
            "min_timestep_input", "max_timestep_input", "unet_te_both_select",
            "bypass_mode_enable", "train_norm_enable", "dora_enable", "ip_gamma_enable",
            "ip_gamma_input", "rescale_enable", "constrain_enable", "constrain_input",
            "lora_fa_enable", "train_blocks_selector", "network_alpha_input",
            "cache_te_outputs_enable", "cache_te_to_disk_enable",
        ):
            elem = getattr(self.widget, name, None)
            if elem is not None:
                elem.hide()

    def setup_connections(self) -> None:
        self.widget.algo_select.currentTextChanged.connect(lambda _: self._sync_adapter())
        self.widget.network_dim_input.valueChanged.connect(lambda _: self._sync_adapter())
        self.widget.dylora_unit_input.valueChanged.connect(lambda _: self._sync_adapter())
        self.widget.cp_enable.clicked.connect(lambda _: self._sync_adapter())
        self.widget.network_dropout_enable.clicked.connect(lambda _: self._sync_adapter())
        self.widget.network_dropout_input.valueChanged.connect(lambda _: self._sync_adapter())
        self.widget.rank_dropout_enable.clicked.connect(lambda _: self._sync_adapter())
        self.widget.rank_dropout_input.valueChanged.connect(lambda _: self._sync_adapter())
        self.widget.module_dropout_enable.clicked.connect(lambda _: self._sync_adapter())
        self.widget.module_dropout_input.valueChanged.connect(lambda _: self._sync_adapter())

    def _current_type(self) -> str:
        text = self.widget.algo_select.currentText().lower()
        if text.startswith("lokr"):
            return "lokr"
        if text.startswith("full"):
            return "none"
        return "lora"

    def _sync_adapter(self) -> None:
        """Rebuild the flat network_args dict from the current widget state."""
        adapter_type = self._current_type()
        is_adapter = adapter_type in ("lora", "lokr")
        is_lokr = adapter_type == "lokr"

        # Enable/disable the controls that only apply to an adapter / LoKr.
        self.widget.network_dim_input.setEnabled(is_adapter)
        self.widget.network_dropout_enable.setEnabled(is_adapter)
        self.widget.network_dropout_input.setEnabled(
            is_adapter and self.widget.network_dropout_enable.isChecked()
        )
        for elem in (self.widget.cp_enable, self.widget.rank_dropout_enable,
                     self.widget.module_dropout_enable, self.widget.dylora_unit_input):
            elem.setEnabled(is_lokr)
        self.widget.rank_dropout_input.setEnabled(is_lokr and self.widget.rank_dropout_enable.isChecked())
        self.widget.module_dropout_input.setEnabled(is_lokr and self.widget.module_dropout_enable.isChecked())

        self.args = {"type": adapter_type}
        if not is_adapter:
            return

        self.args["rank"] = self.widget.network_dim_input.value()
        if self.widget.network_dropout_enable.isChecked():
            self.args["dropout"] = round(self.widget.network_dropout_input.value(), 4)

        if is_lokr:
            factor = self.widget.dylora_unit_input.value()
            self.args["factor"] = factor if factor != 0 else -1
            if self.widget.cp_enable.isChecked():
                self.args["use_tucker"] = True
            if self.widget.rank_dropout_enable.isChecked():
                self.args["rank_dropout"] = round(self.widget.rank_dropout_input.value(), 4)
            if self.widget.module_dropout_enable.isChecked():
                self.args["module_dropout"] = round(self.widget.module_dropout_input.value(), 4)

    def load_args(self, args: dict) -> bool:
        network = args.get(self.name, {})
        if not isinstance(network, dict):
            network = {}

        adapter_type = str(network.get("type", network.get("algo", "lora"))).lower()
        if adapter_type in ("none", "full", "fft", "full fine-tune", "full_finetune"):
            self.widget.algo_select.setCurrentText("Full fine-tune")
        elif adapter_type == "lokr":
            self.widget.algo_select.setCurrentText("LoKr")
        else:
            self.widget.algo_select.setCurrentText("LoRA")

        self.widget.network_dim_input.setValue(int(network.get("rank", network.get("network_dim", 32))))
        factor = network.get("factor", -1)
        self.widget.dylora_unit_input.setValue(int(factor) if factor is not None else -1)
        self.widget.cp_enable.setChecked(bool(network.get("use_tucker", False)))
        self.widget.rank_dropout_enable.setChecked(bool(network.get("rank_dropout", False)))
        self.widget.rank_dropout_input.setValue(float(network.get("rank_dropout", 0.0)) or 0.1)
        self.widget.module_dropout_enable.setChecked(bool(network.get("module_dropout", False)))
        self.widget.module_dropout_input.setValue(float(network.get("module_dropout", 0.0)) or 0.1)
        self.widget.network_dropout_enable.setChecked(bool(network.get("dropout", False)))
        self.widget.network_dropout_input.setValue(float(network.get("dropout", 0.0)) or 0.1)

        self._sync_adapter()
        return True
