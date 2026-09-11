from dataclasses import dataclass
from functools import partial
from koil.qt import qt_to_async, async_to_qt
from mikro_napari.models.representation import SELECT_MODE_MAP
from mikro.api.schema import (
    ROI,
    UpdateRoiInput,
    aget_roi,
    aget_image,
    aupdate_roi,
    Image,
)

from qtpy import QtWidgets
from qtpy import QtCore
from arkitekt import App
from mikro_napari.utils import NapariROI
from napari.layers import Layer, Shapes
import webbrowser
from mikro_napari.widgets.table.table_widget import TableWidget
from mikro_napari.widgets.base import BaseMikroNapariWidget
from rekuest.api.schema import alist_shortcuts
from rekuest_widgets.structure import Structure, StructureWidget


class RoiWidget(QtWidgets.QWidget):
    """A widget for displaying ROIs."""

    def __init__(self, app: App, roi: NapariROI, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self._layout = QtWidgets.QVBoxLayout()
        self.setLayout(self._layout)

        self.detailquery = async_to_qt(aget_roi)
        self.detailquery.returned.connect(self.update_layout)
        self.detailquery.run(roi.id)

    def update_layout(self, roi: ROI):
        self._layout.addWidget(QtWidgets.QLabel(roi.label))
        if roi.creator.email:
            self._layout.addWidget(QtWidgets.QLabel(roi.creator.email))
        self._layout.addWidget(QtWidgets.QLabel(roi.id))


class RoiLayerWidget(QtWidgets.QWidget):
    """A widget for displaying ROIs."""

    def __init__(self, app: App, layer: Shapes, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self._layout = QtWidgets.QVBoxLayout()
        self.setLayout(self._layout)

        self.layer = layer
        self.detailquery = async_to_qt(aget_roi)

        self.attach_entity = async_to_qt(self.aattach_entity)
        self.attach_relation = async_to_qt(self.aattach_relation)

        self.detailquery.returned.connect(self.update_layout)
        self.current_marked_rois = []

        self.layer.bind_key("m", self.show_relat_dialog, overwrite=True)
        self.layer.bind_key("n", self.show_new_entity_dialog, overwrite=True)
        self.layer.bind_key("o", self.open_in_browser, overwrite=True)

        self.layer.mouse_drag_callbacks.append(self.on_drag_roi_layer)

    async def aattach_entity(
        self,
        linked_expression_id: str,
    ):
        for roi in self.current_marked_rois:
            entity_id = await acreate_measurement(f"@mikro/roi:{roi.id}")
            print("Attaaaching")

        print("ATTACHED")

    async def aattach_relation(
        self,
        linked_expression_id: str,
    ):
        assert len(self.current_marked_rois) == 2, "Please select only two ROIs"
        x, y = self.current_marked_rois

        print("Creating relation based on rois", x, y)
        try:
            relation = await acreate_entity_relation(
                left=f"@mikro/roi:{x.id}",
                right=f"@mikro/roi:{y.id}",
                relation=linked_expression_id,
            )

            print("Relation created", relation)
        except Exception as e:
            print("Error", e)

    def open_in_browser(self, event):
        for i in self.current_marked_rois:
            webbrowser.open(i.id)

    def show_new_entity_dialog(self, event):
        if len(self.current_marked_rois) < 1:
            _ = QtWidgets.QMessageBox.warning(
                self,
                "Select at least one ROI",
                "To create a new entity, please select at least one ROI.",
                QtWidgets.QMessageBox.Ok,
            )
            return

        dialog = NewRoisEntityDialog(parent=self)
        if dialog.exec_():
            self.attach_entity.run(dialog.selected_item)
        else:
            print("Cancelled")

    def show_relat_dialog(self, event):
        if len(self.current_marked_rois) != 2:
            _ = QtWidgets.QMessageBox.warning(
                self,
                "Select only two ROIs",
                "Please select only two ROIs to show relations.",
                QtWidgets.QMessageBox.Ok,
            )
            return

        dialog = NewRelationDialog(parent=self)
        if dialog.exec_():
            self.attach_relation.run(dialog.selected_item)
        else:
            print("Cancelled")

    def destroy(self, destroyWindow: bool = ..., destroySubWindows: bool = ...) -> None:
        self.layer.mouse_drag_callbacks.remove(self.on_drag_roi_layer)
        self.layer.bind_key("r", None)
        return super().destroy(destroyWindow, destroySubWindows)

    def on_drag_roi_layer(self, layer, event):
        while event.type != "mouse_release":
            yield

        print("dragged")
        if layer.mode in SELECT_MODE_MAP:
            print(self.layer.selected_data)
            selected_rois = []
            for i in self.layer.selected_data:
                print("Selected", i)

                print(self.layer.features["roi"][i])

            self.current_marked_rois = [
                self.layer.features["roi"][i] for i in self.layer.selected_data
            ]

    def update_layout(self, roi: ROI):
        self._layout.addWidget(QtWidgets.QLabel(roi.label))
        if roi.creator.email:
            self._layout.addWidget(QtWidgets.QLabel(roi.creator.email))
        self._layout.addWidget(QtWidgets.QLabel(roi.id))


class RepresentationWidget(QtWidgets.QWidget):
    """A widget for displaying ROIs."""

    def __init__(self, image: Image, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        # set maximum width
        self.setMaximumWidth(100)

        self.start_image = image
        print(self.start_image)
        self._layout = QtWidgets.QVBoxLayout()
        self.setLayout(self._layout)

        self.detailquery = async_to_qt(aget_image)
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

        self.structure_widget = StructureWidget()

        self.viewer.bind_key("Control-1", self.run_first_shortcut)
        self.viewer.bind_key("Control-2", self.run_second_shortcut)
        self.viewer.bind_key("Control-3", self.run_third_shortcut)

        self.mylayout.addWidget(self.structure_widget)

        self.bus.add_structure_hook("sidebar_widget", self.structure_widget.load)

        self.mylayout.addWidget(self._active_widget)

        self.viewer.layers.selection.events.changed.connect(self.on_layer_changed)

        self.setLayout(self.mylayout)

    def run_first_shortcut(self, event):
        self.structure_widget.run_shortcut_index(0)

    def run_second_shortcut(self, event):
        self.structure_widget.run_shortcut_index(1)

    def run_third_shortcut(self, event):
        self.structure_widget.run_shortcut_index(2)

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
            if "identifier" in layer.metadata:
                print("Loading structure")
                self.structure_widget.load(
                    [
                        Structure(
                            identifier=layer.metadata["identifier"],
                            object=layer.metadata["object"],
                        )
                    ]
                )
            else:
                self.replace_widget(QtWidgets.QLabel("Unknown type"))
        else:
            self.replace_widget(QtWidgets.QLabel("Nothing selected"))

    def on_roi_selection_changed(self, event):
        self.viewer.layers.selection.active
        layer = self.viewer.layers.selection.active
