from PySide6.QtWidgets import (
    QFormLayout,
    QGroupBox,
    QLabel,
    QLineEdit,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from modules.BaseWidget import BaseWidget


class CaptionsWidget(BaseWidget):
    """Global caption handling for diffusion-pipe.

    diffusion-pipe reads caption knobs dataset-wide (top level of dataset.toml),
    not per subset, so they live here instead of in each subset panel:

    - Multi-caption: train on both ``.txt`` (booru tags) and ``.caption`` (NL).
      The backend writes a ``captions.json`` per dataset folder during
      validation and turns on ``online_captions``; diffusion-pipe then trains
      one example per caption, so an image with both files is seen twice per
      epoch.
    - Tag shuffling: diffusion-pipe pre-shuffles captions at caching time
      (``cache_shuffle_num`` variants). Keep-tokens rules only apply while
      shuffling: a separator splits the caption into a fixed head and a
      shuffled tail (takes precedence), otherwise the first N tags stay fixed.

    Emits ``caption_args`` into the dataset payload.
    """

    def __init__(self, parent: QWidget = None) -> None:
        super().__init__(parent)
        self.colap.set_title("Captions")
        self.name = "caption_args"
        self.dataset_args = {}

        self.setup_widget()
        self.setup_connections()
        self._sync()

    def setup_widget(self) -> None:
        super().setup_widget()
        outer = QVBoxLayout(self.content)
        outer.setContentsMargins(4, 4, 4, 4)
        outer.setSpacing(6)

        self.combine_group = QGroupBox("Train on both .txt tags and .caption NLP")
        self.combine_group.setCheckable(True)
        self.combine_group.setChecked(False)
        self.combine_group.setToolTip(
            "Builds captions.json in each dataset folder (next to the images) during "
            "validation. Images with both <image>.txt and <image>.caption are trained "
            "twice per epoch (once on the tags, once on the NL caption); images with "
            "only .txt are trained once, as usual."
        )
        group_layout = QVBoxLayout(self.combine_group)

        note = QLabel(
            "Each image is trained once per caption it has:\n"
            "  • .txt only  →  1× per epoch (tags)\n"
            "  • .txt + .caption  →  2× per epoch (tags, then NL)\n"
            "Without this, only .txt is used and .caption is ignored."
        )
        note.setWordWrap(True)
        group_layout.addWidget(note)

        outer.addWidget(self.combine_group)

        self.shuffle_group = QGroupBox("Shuffle caption tags")
        self.shuffle_group.setCheckable(True)
        self.shuffle_group.setChecked(False)
        self.shuffle_group.setToolTip(
            "diffusion-pipe caches several pre-shuffled variants of each caption "
            "and picks one per step. Applies to every dataset folder."
        )
        shuffle_layout = QFormLayout(self.shuffle_group)

        self.keep_tokens_input = QSpinBox()
        self.keep_tokens_input.setRange(0, 999)
        self.keep_tokens_input.setToolTip(
            "The first N comma-separated tags are never shuffled. 0 disables."
        )
        shuffle_layout.addRow("Keep first N tags", self.keep_tokens_input)

        self.keep_tokens_sep_input = QLineEdit()
        self.keep_tokens_sep_input.setPlaceholderText("e.g. |||  (empty = off)")
        self.keep_tokens_sep_input.setToolTip(
            "Everything before this separator stays fixed; only the rest is "
            "shuffled. Takes precedence over 'Keep first N tags' when the "
            "separator appears in a caption."
        )
        shuffle_layout.addRow("Keep tokens separator", self.keep_tokens_sep_input)

        outer.addWidget(self.shuffle_group)

    def setup_connections(self) -> None:
        self.combine_group.toggled.connect(lambda _: self._sync())
        self.shuffle_group.toggled.connect(lambda _: self._sync())
        self.keep_tokens_input.valueChanged.connect(lambda _: self._sync())
        self.keep_tokens_sep_input.textChanged.connect(lambda _: self._sync())

    def _sync(self) -> None:
        args: dict = {}
        if self.combine_group.isChecked():
            args["combine_txt_caption"] = True
        if self.shuffle_group.isChecked():
            args["shuffle_caption"] = True
            if self.keep_tokens_input.value() > 0:
                args["keep_tokens"] = self.keep_tokens_input.value()
            separator = self.keep_tokens_sep_input.text()
            if separator:
                args["keep_tokens_separator"] = separator
        self.dataset_args = args

    def load_dataset_args(self, dataset_args: dict) -> bool:
        caption = dataset_args.get(self.name, {}) or {}
        self.combine_group.setChecked(bool(caption.get("combine_txt_caption", False)))
        self.shuffle_group.setChecked(bool(caption.get("shuffle_caption", False)))
        self.keep_tokens_input.setValue(int(caption.get("keep_tokens", 0) or 0))
        self.keep_tokens_sep_input.setText(str(caption.get("keep_tokens_separator", "") or ""))
        self._sync()
        return True
