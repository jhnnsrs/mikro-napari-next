from mikro_napari.widgets.main_widget import MikroNapariWidget
from mikro_napari.widgets.sidebar.sidebar import SidebarWidget
from napari import Viewer

global_app = None

class ArkitektPluginWidget(MikroNapariWidget):
    """Arkitekt Plugin Widget"""

    def __init__(self, viewer: Viewer) -> None:
        super(ArkitektPluginWidget, self).__init__(viewer=viewer)



class ArkitektPluginSidebar(SidebarWidget):


    def __init__(self, viewer: Viewer) -> None:
        super(ArkitektPluginSidebar, self).__init__(viewer=viewer)
