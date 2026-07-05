from PySide6.QtWidgets import QGroupBox, QLabel, QVBoxLayout, QWidget

from modules.BaseWidget import BaseWidget


class CaptionsWidget(BaseWidget):
    """Train on both ``.txt`` (booru tags) and ``.caption`` (natural language).

    diffusion-pipe's dataset reads ``.txt`` only by default. When this is
    enabled, the backend writes a ``captions.json`` into each dataset folder
    during validation giving each image its available captions (tags and/or NL),
    and turns on ``online_captions`` (with random-caption OFF). diffusion-pipe
    then trains one example per caption, so an image with both files is seen
    twice per epoch. Emits ``caption_args`` into the dataset payload.
    """

    def __init__(self, parent: QWidget = None) -> None:
        super().__init__(parent)
        self.colap.set_title("Captions (.txt + .caption)")
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

    def setup_connections(self) -> None:
        self.combine_group.toggled.connect(lambda _: self._sync())

    def _sync(self) -> None:
        if self.combine_group.isChecked():
            self.dataset_args = {"combine_txt_caption": True}
        else:
            self.dataset_args = {}

    def load_dataset_args(self, dataset_args: dict) -> bool:
        caption = dataset_args.get(self.name, {}) or {}
        self.combine_group.setChecked(bool(caption.get("combine_txt_caption", False)))
        self._sync()
        return True
