from mikro_napari.controllers.base import BaseMikroNapariController
from mikro_napari.widgets.sidebar.sidebar import SidebarWidget
from rekuest_next.qt.builders import qtinloopactifier
from mikro_next.api.schema import Table


class SidebarController(BaseMikroNapariController):
    def __init__(self, sidebar: SidebarWidget, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.sidebar = sidebar

        rekuest = self.app.services.get("rekuest")

        if rekuest:
            rekuest.register(
                self.open_table,
                actifier=qtinloopactifier,
                parent=self,
                collections=["display", "interactive"],
            )

    def open_table(self, table: Table):
        """Open Table in Sidebar

        Opens the table in an accessible sidebar widget.

        Args:
            table (TableFragment): Table to open


        """
        self.sidebar.show_table_widget(table)
