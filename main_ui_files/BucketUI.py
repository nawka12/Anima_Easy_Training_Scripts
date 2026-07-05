from PySide6.QtWidgets import QWidget
from ui_files.BucketUI import Ui_bucket_ui
from modules.BaseWidget import BaseWidget


class BucketWidget(BaseWidget):
    """Aspect-ratio bucketing for diffusion-pipe.

    diffusion-pipe buckets by aspect ratio around fixed pixel areas, not by
    resolution steps like sd_scripts. This widget emits::

        enable_ar_bucket, num_ar_buckets   (min_ar / max_ar keep dp defaults)

    reusing the old reso-bucket controls: the group toggle -> enable_ar_bucket
    and the "steps" spinbox -> num_ar_buckets. The reso-specific controls are
    hidden (they have no aspect-ratio equivalent).
    """

    def __init__(self, parent: QWidget = None) -> None:
        super().__init__(parent)
        self.colap.set_title("Bucket Args (Aspect Ratio)")
        self.widget = Ui_bucket_ui()

        self.name = "bucket_args"
        self.dataset_args = {}

        self.setup_widget()
        self.setup_connections()

    def setup_widget(self) -> None:
        super().setup_widget()
        self.widget.setupUi(self.content)

        # Repurpose the "steps" spinbox as the aspect-ratio bucket count, and
        # relabel it (the generated label still says "Bucket Resolution Steps").
        ar_tooltip = (
            "num_ar_buckets: number of aspect-ratio buckets, evenly spaced in log "
            "space between min_ar (0.5) and max_ar (2.0). More buckets = finer "
            "aspect-ratio granularity and less cropping."
        )
        self.widget.steps_input.setMinimum(1)
        self.widget.steps_input.setMaximum(64)
        self.widget.steps_input.setSingleStep(1)
        self.widget.steps_input.setValue(7)
        self.widget.steps_input.setToolTip(ar_tooltip)
        if getattr(self.widget, "steps_label", None) is not None:
            self.widget.steps_label.setText("Number of AR Buckets")
            self.widget.steps_label.setToolTip(ar_tooltip)

        # Hide the resolution-step controls that don't map to AR bucketing.
        for name in ("min_input", "max_input", "bucket_no_upscale", "multires_training"):
            elem = getattr(self.widget, name, None)
            if elem is not None:
                elem.hide()
        for name in ("min_label", "max_label", "min_bucket_label", "max_bucket_label"):
            elem = getattr(self.widget, name, None)
            if elem is not None:
                elem.hide()

        self.enable_disable(self.widget.bucket_group.isChecked())

    def setup_connections(self) -> None:
        self.widget.steps_input.valueChanged.connect(
            lambda x: self.edit_dataset_args("num_ar_buckets", x)
        )
        self.widget.bucket_group.clicked.connect(self.enable_disable)

    def enable_disable(self, checked: bool) -> None:
        self.dataset_args = {}
        if not checked:
            self.edit_dataset_args("enable_ar_bucket", False)
            return
        self.edit_dataset_args("enable_ar_bucket", True)
        self.edit_dataset_args("num_ar_buckets", self.widget.steps_input.value())

    def load_dataset_args(self, dataset_args: dict) -> bool:
        dataset_args: dict = dataset_args.get(self.name, {})
        # accept both the new (enable_ar_bucket/num_ar_buckets) and old keys
        enabled = dataset_args.get(
            "enable_ar_bucket", dataset_args.get("enable_bucket", self.widget.bucket_group.isChecked())
        )
        self.widget.bucket_group.setChecked(enabled)
        self.widget.steps_input.setValue(
            dataset_args.get("num_ar_buckets", self.widget.steps_input.value())
        )
        self.enable_disable(self.widget.bucket_group.isChecked())
        return True
