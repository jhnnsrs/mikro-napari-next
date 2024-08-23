from arkitekt_next.qt import QtApp
from napari import Viewer
from qtpy import QtWidgets
from mikro_napari.global_app import get_app_or_build_for_widget as get


class BaseMikroNapariWidget(QtWidgets.QWidget):
    app: QtApp
    viewer: Viewer

    def __init__(
        self,
        *args,
        viewer: Viewer = None,
        app: QtApp = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        assert viewer is not None, "viewer must be provided"
        self.app = app or get(self)
        self.viewer = viewer
