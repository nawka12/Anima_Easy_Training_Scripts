from PySide6.QtCore import Signal
from PySide6 import QtCore, QtWidgets

from main_ui_files.AccelerateUI import AccelerateWidget
from main_ui_files.BucketUI import BucketWidget
from main_ui_files.ExtraArgsUI import ExtraArgsWidget
from main_ui_files.GeneralUI import GeneralWidget
from main_ui_files.LoggingUI import LoggingWidget
from main_ui_files.NetworkUI import NetworkWidget
from main_ui_files.OptimizerUI import OptimizerWidget
from main_ui_files.SampleUI import SampleWidget
from main_ui_files.SavingUI import SavingWidget
from modules.BaseWidget import BaseWidget


class ArgsWidget(QtWidgets.QWidget):
    cacheLatentsChecked = Signal(bool)
    keepTokensSepChecked = Signal(bool)
    maskedLossChecked = Signal(bool)

    def __init__(self, parent: QtWidgets.QWidget = None) -> None:
        super().__init__(parent)
        self.scroll_area = QtWidgets.QScrollArea()
        self.scroll_widget = QtWidgets.QWidget()
        self.args_widget_array: list[BaseWidget] = []
        self.general_widget = GeneralWidget()
        self.network_widget = NetworkWidget()
        self.optimizer_widget = OptimizerWidget()

        self.setup_widget()
        self.setup_args_widgets()

    def setup_widget(self) -> None:
        self.setMinimumSize(600, 300)
        self.setLayout(QtWidgets.QVBoxLayout())
        self.scroll_area.setWidgetResizable(True)
        self.scroll_widget.setLayout(QtWidgets.QVBoxLayout())
        self.scroll_widget.layout().setSpacing(0)
        self.scroll_widget.layout().setAlignment(QtCore.Qt.AlignmentFlag.AlignTop)
        self.scroll_widget.layout().setContentsMargins(0, 0, 0, 0)
        self.scroll_area.setWidget(self.scroll_widget)
        self.layout().addWidget(self.scroll_area)

    def setup_args_widgets(self) -> None:
        self.general_widget.colap.toggle_collapsed()
        self.general_widget.colap.title_frame.setChecked(True)
        self.general_widget.cacheLatentsChecked.connect(self.cacheLatentsChecked.emit)
        self.general_widget.keepTokensSepChecked.connect(self.keepTokensSepChecked.emit)

        self.optimizer_widget.maskedLossChecked.connect(self.maskedLossChecked.emit)

        self.args_widget_array.append(self.general_widget)
        self.args_widget_array.append(self.network_widget)
        self.args_widget_array.append(self.optimizer_widget)
        self.args_widget_array.append(SavingWidget())
        self.args_widget_array.append(BucketWidget())
        self.args_widget_array.append(SampleWidget())
        self.args_widget_array.append(LoggingWidget())
        self.accelerate_widget = AccelerateWidget()
        self.args_widget_array.append(self.accelerate_widget)
        self.args_widget_array.append(ExtraArgsWidget())

        # Anima trains text encoders, so cache_text_encoder_outputs is available.
        self.network_widget.toggle_sdxl(True)

        for widget in self.args_widget_array:
            widget.setVisible(True)
            self.scroll_widget.layout().addWidget(widget)

    def get_args(self) -> dict:
        args = {}
        dataset_args = {}
        for widget in self.args_widget_array:
            if widget.args:
                args[widget.name] = widget.args
            if widget.dataset_args:
                dataset_args[widget.name] = widget.dataset_args

        anima_args = self.general_widget.get_anima_args()
        if anima_args:
            args["anima_args"] = anima_args

        return {"args": args, "dataset": dataset_args}

    def get_validation_errors(self) -> list[str]:
        errors: list[str] = []
        for widget in self.args_widget_array:
            validator = getattr(widget, "get_validation_errors", None)
            if callable(validator):
                errors.extend(validator())
        return errors

    def load_args(self, args: dict, dataset_args: dict) -> None:
        for widget in self.args_widget_array:
            widget.load_args(args)
            widget.load_dataset_args(dataset_args)
