from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from modules.BaseWidget import BaseWidget
from modules.ScrollOnSelect import SpinBox


class _ResolutionRow(QFrame):
    """One [[datasets]] entry: resolution + bucket + batch settings."""

    def __init__(self, parent: QWidget = None) -> None:
        super().__init__(parent)
        self.setFrameShape(QFrame.Shape.StyledPanel)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(6)

        def _spin(minimum: int, maximum: int, value: int, step: int = 1) -> SpinBox:
            sb = SpinBox()
            sb.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
            sb.setMinimum(minimum)
            sb.setMaximum(maximum)
            sb.setSingleStep(step)
            sb.setValue(value)
            sb.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
            return sb

        self.resolution_input = _spin(1, 16777215, 1024, 64)
        self.skip_input = _spin(0, 16777215, 0, 64)
        self.batch_input = _spin(1, 1024, 1)
        self.min_bucket_input = _spin(1, 16777215, 256, 64)
        self.max_bucket_input = _spin(1, 16777215, 3072, 64)
        self.steps_input = _spin(1, 16777215, 64, 8)

        self.no_upscale_enable = QCheckBox("No Upscale")

        self.remove_button = QPushButton("✕")
        self.remove_button.setToolTip("Remove this resolution")
        self.remove_button.setFixedWidth(28)

        for label_text, widget in [
            ("Res", self.resolution_input),
            ("Skip ≤", self.skip_input),
            ("Batch", self.batch_input),
            ("Min Bucket", self.min_bucket_input),
            ("Max Bucket", self.max_bucket_input),
            ("Step", self.steps_input),
        ]:
            layout.addWidget(QLabel(label_text))
            layout.addWidget(widget)
        layout.addWidget(self.no_upscale_enable)
        layout.addStretch(1)
        layout.addWidget(self.remove_button)

        self.resolution_input.setToolTip("Target resolution for this dataset block.")
        self.skip_input.setToolTip(
            "skip_image_resolution: images smaller than this are skipped for this block. 0 = no skip."
        )
        self.batch_input.setToolTip("batch_size for this dataset block.")
        self.min_bucket_input.setToolTip("min_bucket_reso for this dataset block.")
        self.max_bucket_input.setToolTip("max_bucket_reso for this dataset block.")
        self.steps_input.setToolTip("bucket_reso_steps for this dataset block.")

    def to_dict(self) -> dict:
        d = {
            "resolution": self.resolution_input.value(),
            "enable_bucket": True,
            "min_bucket_reso": self.min_bucket_input.value(),
            "max_bucket_reso": self.max_bucket_input.value(),
            "bucket_reso_steps": self.steps_input.value(),
            "batch_size": self.batch_input.value(),
        }
        if self.skip_input.value() > 0:
            d["skip_image_resolution"] = self.skip_input.value()
        if self.no_upscale_enable.isChecked():
            d["bucket_no_upscale"] = True
        return d

    def load_from(self, data: dict) -> None:
        self.resolution_input.setValue(int(data.get("resolution", 1024)))
        self.skip_input.setValue(int(data.get("skip_image_resolution", 0)))
        self.batch_input.setValue(int(data.get("batch_size", 1)))
        self.min_bucket_input.setValue(int(data.get("min_bucket_reso", 256)))
        self.max_bucket_input.setValue(int(data.get("max_bucket_reso", 3072)))
        self.steps_input.setValue(int(data.get("bucket_reso_steps", 64)))
        self.no_upscale_enable.setChecked(bool(data.get("bucket_no_upscale", False)))


class AdditionalResolutionsWidget(BaseWidget):
    """Optional list of additional [[datasets]] resolution blocks for mixed-res training.

    When empty, the toml is saved in single-resolution Format 1 ([[subsets]] at top).
    When populated, save emits sd-scripts' multi-resolution Format 2 ([[datasets]]
    with the General Args + Bucket Args settings as the first dataset and these
    rows providing the rest). The same subset list is replayed under every dataset.
    """

    def __init__(self, parent: QWidget = None) -> None:
        super().__init__(parent)
        self.colap.set_title("Additional Resolutions (Mixed-Res)")
        self.name = "additional_resolutions"

        self.rows: list[_ResolutionRow] = []
        self.setup_widget()

    def setup_widget(self) -> None:
        super().setup_widget()
        outer = QVBoxLayout(self.content)
        outer.setContentsMargins(4, 4, 4, 4)
        outer.setSpacing(4)

        hint = QLabel(
            "Add one row per extra resolution. The base resolution / bucket / batch "
            "from General Args + Bucket Args becomes the first dataset; each row "
            "below becomes another [[datasets]] block sharing the same subsets."
        )
        hint.setWordWrap(True)
        outer.addWidget(hint)

        self._rows_layout = QVBoxLayout()
        self._rows_layout.setSpacing(2)
        outer.addLayout(self._rows_layout)

        self.add_button = QPushButton("+ Add Resolution")
        self.add_button.clicked.connect(lambda: self.add_row())
        outer.addWidget(self.add_button)

    def setup_connections(self) -> None:
        # Connections are managed per-row in add_row().
        pass

    def add_row(self, data: dict | None = None) -> _ResolutionRow:
        row = _ResolutionRow(self.content)
        row.remove_button.clicked.connect(lambda _=False, r=row: self.remove_row(r))
        self._rows_layout.addWidget(row)
        self.rows.append(row)
        if data is not None:
            row.load_from(data)
        return row

    def remove_row(self, row: _ResolutionRow) -> None:
        if row not in self.rows:
            return
        self.rows.remove(row)
        self._rows_layout.removeWidget(row)
        row.deleteLater()

    def clear(self) -> None:
        while self.rows:
            self.remove_row(self.rows[-1])

    def get_extra_datasets(self) -> list[dict]:
        return [row.to_dict() for row in self.rows]

    def load_extra_datasets(self, datasets: list[dict]) -> None:
        self.clear()
        for d in datasets:
            self.add_row(d)

    def load_args(self, _: dict) -> bool:
        return False

    def load_dataset_args(self, _: dict) -> bool:
        return False
