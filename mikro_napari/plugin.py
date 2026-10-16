from napari.viewer import Viewer

from mikro_napari.widgets.main_widget import MikroNapariWidget


class ArkitektPluginWidget(MikroNapariWidget):
    def __init__(self, viewer: Viewer) -> None:
        super(ArkitektPluginWidget, self).__init__(viewer=viewer)
