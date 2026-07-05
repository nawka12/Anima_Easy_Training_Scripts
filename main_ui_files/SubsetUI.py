import contextlib
from pathlib import Path

from PySide6.QtCore import Signal
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QFileDialog, QWidget

from modules.BaseWidget import BaseWidget
from modules.DragDropLineEdit import DragDropLineEdit
from ui_files.sub_dataset_input import Ui_sub_dataset_input


# diffusion-pipe consumes exactly three per-directory keys: path (image_dir),
# num_repeats and mask_path. The other sd_scripts subset controls (augments,
# regularization, caption extension, per-subset shuffle/keep-tokens, ...) have
# no equivalent and are hidden; global caption handling lives in the Captions
# panel instead.
class SubsetWidget(BaseWidget):
    edited = Signal(dict, str)

    def __init__(
        self, parent: QWidget = None, display_name: str = "", name: str = ""
    ) -> None:
        super().__init__(parent)
        self.colap.set_title(display_name)
        self.colap.set_extra("remove")
        self.widget = Ui_sub_dataset_input()
        self.name = name

        self.dataset_args = {
            "num_repeats": 1,
            "name": self.name,
        }

        self.setup_widget()
        self.setup_connections()

    def setup_widget(self) -> None:
        super().setup_widget()
        self.widget.setupUi(self.content)
        self.widget.image_folder_input.setMode("folder")
        self.widget.image_folder_input.highlight = True
        self.widget.image_folder_selector.setIcon(
            QIcon(str(Path("icons/more-horizontal.svg")))
        )
        self.widget.masked_image_input.setMode("folder")
        self.widget.masked_image_input.highlight = True
        self.widget.masked_image_selector.setIcon(
            QIcon(str(Path("icons/more-horizontal.svg")))
        )

        # Controls with no diffusion-pipe mapping.
        for name in (
            "target_image_dir_label", "target_image_folder_input", "target_image_folder_selector",
            "flip_augment_enable", "shuffle_captions_enable", "random_crop_enable",
            "color_augment_enable", "regularization_images_enable", "validation_images_enable",
            "keep_tokens_label", "keep_tokens_input", "caption_label", "caption_extension_selector",
            "random_crop_padding_percent_label", "random_crop_padding_percent_input", "extra_args",
        ):
            elem = getattr(self.widget, name, None)
            if elem is not None:
                elem.hide()

    def setup_connections(self) -> None:
        self.widget.image_folder_input.textChanged.connect(
            lambda x: self.edit_dataset_args("image_dir", x, True)
        )
        self.widget.image_folder_selector.clicked.connect(
            lambda: self.set_folder_from_dialog(
                "Subset Image Folder", self.widget.image_folder_input
            )
        )
        self.widget.masked_image_input.textChanged.connect(
            lambda x: self.edit_dataset_args("mask_path", x, True)
        )
        self.widget.masked_image_selector.clicked.connect(
            lambda: self.set_folder_from_dialog(
                "Masked Image Folder", self.widget.masked_image_input, False
            )
        )
        self.widget.repeats_input.valueChanged.connect(
            lambda x: self.edit_dataset_args("num_repeats", x)
        )

    def edit_dataset_args(
        self, name: str, value: object, optional: bool = False
    ) -> None:
        super().edit_dataset_args(name, value, optional)
        self.edited.emit(self.dataset_args, self.name)

    def set_folder_from_dialog(
        self,
        title_str: str,
        element: DragDropLineEdit,
        calc_repeats: bool = True,
        path: Path = None,
    ) -> None:
        if path and path.exists():
            file_name = path
        else:
            default_dir = Path(element.text())
            file_name = QFileDialog.getExistingDirectory(
                self,
                title_str,
                dir=str(default_dir) if default_dir.exists() else "",
            )
            if not file_name:
                return
            file_name = Path(file_name)
        element.setText(file_name.as_posix())
        element.update_stylesheet()
        if not calc_repeats:
            return
        with contextlib.suppress(ValueError):
            repeats = int(file_name.name.split("_")[0])
            self.widget.repeats_input.setValue(repeats)

    def enable_disable_masked_loss(self, checked: bool) -> None:
        # diffusion-pipe does masked training per directory via [[directory]].mask_path.
        if "mask_path" in self.dataset_args:
            del self.dataset_args["mask_path"]
        self.widget.masked_image_input.setEnabled(checked)
        self.widget.masked_image_selector.setEnabled(checked)
        self.edit_dataset_args(
            "mask_path",
            self.widget.masked_image_input.text() if checked else False,
            True,
        )

    def load_dataset_args(self, dataset_args: dict) -> bool:
        self.widget.image_folder_input.setText(dataset_args.get("image_dir", ""))
        self.widget.masked_image_input.setText(
            dataset_args.get("mask_path", dataset_args.get("conditioning_data_dir", ""))
        )
        self.widget.repeats_input.setValue(dataset_args.get("num_repeats", 1))

        self.edit_dataset_args("image_dir", self.widget.image_folder_input.text(), True)
        self.edit_dataset_args("mask_path", self.widget.masked_image_input.text(), True)
        self.edit_dataset_args("num_repeats", self.widget.repeats_input.value())

        self.edited.emit(self.dataset_args, self.name)
        return True
