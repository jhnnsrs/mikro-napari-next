from mikro_next.api.schema import Table
from qtpy.QtWidgets import QTableView, QWidget
from qtpy import QtWidgets, QtCore
from mikro_napari.widgets.base import BaseMikroNapariWidget
from .dataframe import PandasModel


class TableWidget(BaseMikroNapariWidget):
    emit_image: QtCore.Signal = QtCore.Signal(object)

    def __init__(self, table: Table, *args, **kwargs) -> None:
        super(TableWidget, self).__init__(*args, **kwargs)
        self.view = QTableView()
        self.view.horizontalHeader().setStretchLastSection(False)
        self.view.setAlternatingRowColors(False)
        self.view.setSelectionBehavior(QTableView.SelectRows)
        self.view.setWindowTitle(table.name)

        self.model = PandasModel(table.data, parent=self)
        self.view.setModel(self.model)
        self.view.show()

        self.mylayout = QtWidgets.QVBoxLayout()
        self.mylayout.addWidget(self.view)

        self.setLayout(self.mylayout)

    def replace_widget(self, widget: QWidget):
        self.mylayout.removeWidget(self._active_widget)
        del self._active_widget
        self._active_widget = widget
        self.mylayout.addWidget(self._active_widget)
