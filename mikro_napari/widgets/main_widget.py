from koil.qt import async_to_qt, qt_to_async, SignalProtocol
from mikro.api.schema import (
    Image,
    from_array_like,
    afrom_array_like,
)
from qtpy import QtWidgets
from qtpy import QtCore
from arkitekt.qt.magic_bar import MagicBar
from mikro_napari.models.representation import RepresentationQtModel
from mikro_napari.widgets.dialogs.open_image import OpenImageDialog
from rekuest_widgets.structure import Structure
from .base import BaseMikroNapariWidget
import xarray as xr
from rekuest.qt.builders import qtinloopactifier


class MikroNapariWidget(BaseMikroNapariWidget):
    emit_image: QtCore.Signal = QtCore.Signal(object)

    def __init__(self, *args, **kwargs) -> None:
        super(MikroNapariWidget, self).__init__(*args, **kwargs)
        self.mylayout = QtWidgets.QVBoxLayout()

        self.representation_controller = RepresentationQtModel(self)

        self.magic_bar = MagicBar(
            self.app,
            dark_mode=True,
            on_error=self.on_arkitekt_error,
        )

        self.magic_bar.app_up.connect(self.on_app_up)
        self.magic_bar.app_down.connect(self.on_app_down)

        self.upload_task = async_to_qt(afrom_array_like)
        self.upload_task.errored.connect(self.on_error)
        self.upload_task.returned.connect(
            self.representation_controller.on_image_loaded
        )

        self.task = None
        self.stask = None

        self.active_non_mikro_layers = []
        self.active_mikro_layers = []
        self.mylayout.addWidget(self.magic_bar)

        self.setWindowTitle("My Own Title")
        self.setLayout(self.mylayout)

        self.aopen = qt_to_async(self.image_loaded)

        self.viewer.layers.selection.events.active.connect(self.on_selection_changed)

        rekuest = self.app.services.get("rekuest")
        if rekuest is None:
            raise Exception("Rekuest service not found")

        rekuest.register(
            self.representation_controller.on_image_loaded,
            actifier=qtinloopactifier,
        )
        rekuest.register(
            self.representation_controller.open_image,
        )
        rekuest.register(
            self.representation_controller.tile_images,
        )
        rekuest.register(
            self.upload_layer,
        )
        rekuest.register(
            self.representation_controller.stream_rois,
        )

    def image_loaded(self, qtfuture, image: Image):
        self.viewer.add_image(
            image.data,
            name=image.name,
            metadata={
                "mikro": True,
                "identifier": "@mikro/image",
                "object": image.id,
                "type": "IMAGE",
            },
        )
        qtfuture.resolve(image)

    def on_arkitekt_error(self, e):
        print(e)
        self.viewer.status = str(e)

    def on_app_up(self):
        self.on_selection_changed()  # TRIGGER ALSO HERE

    def on_app_down(self):
        pass

    def on_selection_changed(self):
        self.active_non_mikro_layers = [
            layer
            for layer in self.viewer.layers.selection
            if not layer.metadata.get("mikro")
        ]
        self.active_mikro_layers = [
            layer
            for layer in self.viewer.layers.selection
            if layer.metadata.get("mikro")
        ]

        pass

    def upload_layer(self, name: str = "") -> Image:
        """Upload Napari Layer

        Upload the current image to the server.

        Args:
            name (str, optional): Overwrite the layer name. Defaults to None.

        Returns:
            RepresentationFragment: The uploaded image
        """
        if not self.active_non_mikro_layers:
            raise Exception("No active layer")

        image_layer = self.active_non_mikro_layers[0]

        if image_layer.ndim == 2:
            if image_layer.rgb:
                xarray = xr.DataArray(image_layer.data, dims=list("xyc"))
            else:
                xarray = xr.DataArray(image_layer.data, dims=list("xy"))

        if image_layer.ndim == 3:
            xarray = xr.DataArray(image_layer.data, dims=list("zxy"))

        if image_layer.ndim == 4:
            xarray = xr.DataArray(image_layer.data, dims=list("tzxy"))

        if image_layer.ndim == 5:
            xarray = xr.DataArray(image_layer.data, dims=list("ctzyx"))

        return from_array_like(xarray, name=name or image_layer.name)

    def cause_upload(self):
        for image_layer in self.active_non_mikro_layers:
            if image_layer.ndim == 2:
                if image_layer.rgb:
                    xarray = xr.DataArray(image_layer.data, dims=list("xyc"))
                else:
                    xarray = xr.DataArray(image_layer.data, dims=list("xy"))

            if image_layer.ndim == 3:
                xarray = xr.DataArray(image_layer.data, dims=list("zxy"))

            if image_layer.ndim == 4:
                xarray = xr.DataArray(image_layer.data, dims=list("tzxy"))

            if image_layer.ndim == 5:
                xarray = xr.DataArray(image_layer.data, dims=list("tzxyc"))

            self.upload_task.run(xarray, name=image_layer.name)
            self.upload_image_button.setText(f"Uploading {image_layer.name}...")
            self.upload_image_button.setEnabled(False)

    def on_upload_finished(self, image):
        self.on_selection_changed()

    def cause_image_load(self):
        rep_dialog = OpenImageDialog(self)
        x = rep_dialog.exec()
        if x:
            self.representation_controller.active_representation = (
                rep_dialog.selected_representation
            )

    def on_error(self, error):
        print(error)
