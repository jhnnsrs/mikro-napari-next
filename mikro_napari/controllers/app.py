from mikro_napari.controllers.base import BaseMikroNapariController
from .sidebar import SidebarController


class AppController(BaseMikroNapariController):
    def __init__(self, sidebar, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.sidebar = SidebarController(
            sidebar,
            app=self.app,
            viewer=self.viewer,
        )
