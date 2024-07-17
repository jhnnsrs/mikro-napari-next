from arkitekt_next.qt import QtApp
from napari import Viewer
from qtpy import QtCore


class BaseMikroNapariController(QtCore.QObject):
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
        assert app is not None, "app must be provided"
        self.app = app
        self.viewer = viewer
