from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from modules.BaseWidget import BaseWidget


class _PromptRow(QFrame):
    """One [[prompts]] entry: a positive prompt + an optional negative prompt."""

    def __init__(self, on_change, on_remove, parent: QWidget = None) -> None:
        super().__init__(parent)
        self.setFrameShape(QFrame.Shape.StyledPanel)
        layout = QGridLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        self.prompt_input = QLineEdit()
        self.prompt_input.setPlaceholderText("prompt (e.g. 1girl, solo, masterpiece, ...)")
        self.negative_input = QLineEdit()
        self.negative_input.setPlaceholderText("negative prompt (optional)")
        self.remove_button = QPushButton("✕")
        self.remove_button.setFixedWidth(28)
        self.remove_button.setToolTip("Remove this prompt")

        layout.addWidget(QLabel("Prompt"), 0, 0)
        layout.addWidget(self.prompt_input, 0, 1)
        layout.addWidget(self.remove_button, 0, 2)
        layout.addWidget(QLabel("Negative"), 1, 0)
        layout.addWidget(self.negative_input, 1, 1)

        self.prompt_input.textChanged.connect(on_change)
        self.negative_input.textChanged.connect(on_change)
        self.remove_button.clicked.connect(lambda: on_remove(self))

    def to_dict(self) -> dict | None:
        prompt = self.prompt_input.text().strip()
        if not prompt:
            return None
        entry = {"prompt": prompt}
        negative = self.negative_input.text().strip()
        if negative:
            entry["negative_prompt"] = negative
        return entry


class SampleWidget(BaseWidget):
    """In-training sampling for diffusion-pipe.

    Emits ``sample_args`` matching the backend's sample.toml + main-config
    cadence keys::

        {sample_every_n_epochs|sample_every_n_steps, sample_at_first,
         width, height, num_inference_steps, guidance_scale, seed,
         prompts: [{prompt, negative_prompt}, ...]}

    When sampling is disabled or there are no prompts, ``sample_args`` is empty
    and the backend writes no sample.toml (no sampling happens).
    """

    def __init__(self, parent: QWidget = None) -> None:
        super().__init__(parent)
        self.colap.set_title("Sample Args")
        self.name = "sample_args"
        self.prompt_rows: list[_PromptRow] = []

        self.setup_widget()
        self.setup_connections()
        self._sync()

    def setup_widget(self) -> None:
        super().setup_widget()
        outer = QVBoxLayout(self.content)
        outer.setContentsMargins(4, 4, 4, 4)
        outer.setSpacing(6)

        self.sample_group = QGroupBox("Enable Sampling")
        self.sample_group.setCheckable(True)
        self.sample_group.setChecked(False)
        group_layout = QVBoxLayout(self.sample_group)

        # --- cadence row ---
        cadence = QHBoxLayout()
        cadence.addWidget(QLabel("Sample every"))
        self.cadence_input = QSpinBox()
        self.cadence_input.setMinimum(1)
        self.cadence_input.setMaximum(1_000_000)
        self.cadence_input.setValue(1)
        cadence.addWidget(self.cadence_input)
        self.cadence_unit = QComboBox()
        self.cadence_unit.addItems(["epochs", "steps"])
        cadence.addWidget(self.cadence_unit)
        self.sample_at_first = QCheckBox("Sample at first step")
        self.sample_at_first.setChecked(True)
        cadence.addWidget(self.sample_at_first)
        cadence.addStretch(1)
        group_layout.addLayout(cadence)

        # --- generation settings row ---
        settings = QGridLayout()
        self.width_input = self._spin(64, 4096, 1024, 64)
        self.height_input = self._spin(64, 4096, 1024, 64)
        self.steps_input = self._spin(1, 200, 32, 1)
        self.cfg_input = QDoubleSpinBox()
        self.cfg_input.setRange(0.0, 30.0)
        self.cfg_input.setSingleStep(0.5)
        self.cfg_input.setValue(4.0)
        self.seed_input = self._spin(0, 2_147_483_647, 42, 1)
        for col, (label, widget) in enumerate([
            ("Width", self.width_input), ("Height", self.height_input),
            ("Steps", self.steps_input), ("CFG", self.cfg_input), ("Seed", self.seed_input),
        ]):
            settings.addWidget(QLabel(label), 0, col)
            settings.addWidget(widget, 1, col)
        group_layout.addLayout(settings)

        # --- prompt list ---
        group_layout.addWidget(QLabel("Prompts"))
        self._prompts_layout = QVBoxLayout()
        self._prompts_layout.setSpacing(2)
        self._prompts_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        group_layout.addLayout(self._prompts_layout)

        self.add_prompt_button = QPushButton("+ Add Prompt")
        group_layout.addWidget(self.add_prompt_button)

        outer.addWidget(self.sample_group)
        self.add_prompt_row()

    @staticmethod
    def _spin(minimum: int, maximum: int, value: int, step: int) -> QSpinBox:
        sb = QSpinBox()
        sb.setMinimum(minimum)
        sb.setMaximum(maximum)
        sb.setSingleStep(step)
        sb.setValue(value)
        return sb

    def setup_connections(self) -> None:
        self.sample_group.toggled.connect(lambda _: self._sync())
        self.cadence_input.valueChanged.connect(lambda _: self._sync())
        self.cadence_unit.currentIndexChanged.connect(lambda _: self._sync())
        self.sample_at_first.toggled.connect(lambda _: self._sync())
        self.width_input.valueChanged.connect(lambda _: self._sync())
        self.height_input.valueChanged.connect(lambda _: self._sync())
        self.steps_input.valueChanged.connect(lambda _: self._sync())
        self.cfg_input.valueChanged.connect(lambda _: self._sync())
        self.seed_input.valueChanged.connect(lambda _: self._sync())
        self.add_prompt_button.clicked.connect(lambda: (self.add_prompt_row(), self._sync()))

    def add_prompt_row(self, prompt: str = "", negative: str = "") -> _PromptRow:
        row = _PromptRow(on_change=self._sync, on_remove=self._remove_prompt_row)
        row.prompt_input.setText(prompt)
        row.negative_input.setText(negative)
        self._prompts_layout.addWidget(row)
        self.prompt_rows.append(row)
        return row

    def _remove_prompt_row(self, row: _PromptRow) -> None:
        if row not in self.prompt_rows:
            return
        self.prompt_rows.remove(row)
        self._prompts_layout.removeWidget(row)
        row.deleteLater()
        self._sync()

    def _sync(self) -> None:
        enabled = self.sample_group.isChecked()
        for child in (self.cadence_input, self.cadence_unit, self.sample_at_first,
                      self.width_input, self.height_input, self.steps_input,
                      self.cfg_input, self.seed_input, self.add_prompt_button):
            child.setEnabled(enabled)

        if not enabled:
            self.args = {}
            return

        args: dict = {}
        if self.cadence_unit.currentText() == "steps":
            args["sample_every_n_steps"] = self.cadence_input.value()
        else:
            args["sample_every_n_epochs"] = self.cadence_input.value()
        if self.sample_at_first.isChecked():
            args["sample_at_first"] = True
        args["width"] = self.width_input.value()
        args["height"] = self.height_input.value()
        args["num_inference_steps"] = self.steps_input.value()
        args["guidance_scale"] = round(self.cfg_input.value(), 3)
        args["seed"] = self.seed_input.value()

        prompts = [entry for row in self.prompt_rows if (entry := row.to_dict())]
        if prompts:
            args["prompts"] = prompts
        self.args = args

    def load_args(self, args: dict) -> bool:
        sample: dict = args.get(self.name, {})
        has_prompts = bool(sample.get("prompts"))
        self.sample_group.setChecked(bool(sample) and has_prompts)

        if "sample_every_n_steps" in sample:
            self.cadence_unit.setCurrentText("steps")
            self.cadence_input.setValue(int(sample["sample_every_n_steps"]))
        else:
            self.cadence_unit.setCurrentText("epochs")
            self.cadence_input.setValue(int(sample.get("sample_every_n_epochs", 1)))
        self.sample_at_first.setChecked(bool(sample.get("sample_at_first", True)))
        self.width_input.setValue(int(sample.get("width", 1024)))
        self.height_input.setValue(int(sample.get("height", 1024)))
        self.steps_input.setValue(int(sample.get("num_inference_steps", sample.get("steps", 32))))
        self.cfg_input.setValue(float(sample.get("guidance_scale", sample.get("cfg", 4.0))))
        self.seed_input.setValue(int(sample.get("seed", 42)))

        while self.prompt_rows:
            self._remove_prompt_row(self.prompt_rows[0])
        for entry in sample.get("prompts", []):
            if isinstance(entry, dict):
                self.add_prompt_row(entry.get("prompt", ""), entry.get("negative_prompt", ""))
            elif isinstance(entry, str):
                self.add_prompt_row(entry)
        if not self.prompt_rows:
            self.add_prompt_row()

        self._sync()
        return True
