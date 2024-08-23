from koil.qt import QtRunner
from mikro_next.api.schema import (
    ROI,
    aget_roi,
    aget_image,
    Image,
)
from qtpy import QtWidgets
from qtpy import QtCore
from arkitekt import App
from mikro_napari.utils import NapariROI

import webbrowser
from mikro_napari.widgets.table.table_widget import TableWidget
from mikro_napari.widgets.base import BaseMikroNapariWidget


class RoiWidget(QtWidgets.QWidget):
    """A widget for displaying ROIs."""

    def __init__(self, app: App, roi: NapariROI, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self._layout = QtWidgets.QVBoxLayout()
        self.setLayout(self._layout)

        self.detailquery = QtRunner(aget_roi)
        self.detailquery.returned.connect(self.update_layout)
        self.detailquery.run(roi.id)

    def update_layout(self, roi: ROI):
        self._layout.addWidget(QtWidgets.QLabel(roi.label))
        if roi.creator.email:
            self._layout.addWidget(QtWidgets.QLabel(roi.creator.email))
        self._layout.addWidget(QtWidgets.QLabel(roi.id))


class RepresentationWidget(QtWidgets.QWidget):
    """A widget for displaying ROIs."""

    def __init__(self, image: Image, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.start_image = image
        print(self.start_image)
        self._layout = QtWidgets.QVBoxLayout()
        self.setLayout(self._layout)

        self.detailquery = QtRunner(aget_image)
        self.detailquery.returned.connect(self.update_layout)
        self.detailquery.run(image.id)

    def clearLayout(self):
        while self._layout.count():
            child = self._layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

    def update_layout(self, image: Image):
        self.clearLayout()

        if image.name:
            self._layout.addWidget(QtWidgets.QLabel(image.name))


class SidebarWidget(BaseMikroNapariWidget):
    emit_image: QtCore.Signal = QtCore.Signal(object)

    def __init__(self, *args, **kwargs) -> None:
        super(SidebarWidget, self).__init__(*args, **kwargs)

        self.mylayout = QtWidgets.QVBoxLayout()

        self._active_widget = QtWidgets.QLabel("Nothing selected")
        self.mylayout.addWidget(self._active_widget)

        self.viewer.layers.selection.events.changed.connect(self.on_layer_changed)

        self.setLayout(self.mylayout)

    def replace_widget(self, widget):
        self.mylayout.removeWidget(self._active_widget)
        del self._active_widget
        self._active_widget = widget
        self.mylayout.addWidget(self._active_widget)

    def select_roi(self, roi: NapariROI):
        self.replace_widget(RoiWidget(self.app, roi))
        pass

    def on_layer_changed(self, event):
        self.viewer.layers.selection.active
        layer = self.viewer.layers.selection.active
        if layer is not None:
            if "representation" in layer.metadata:
                self.replace_widget(
                    RepresentationWidget(layer.metadata["representation"])
                )
                print(layer)
