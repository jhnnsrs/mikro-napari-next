from typing import Any, Callable, Optional

from mikro import Mikro
from mikro.api.schema import ArrayDataset, ArrayDatasetFilter, OffsetPaginationInput
from qtpy import QtCore, QtWidgets

PAGE = 50


def search(mikro: Mikro, text: str) -> tuple[ArrayDataset, ...]:
    """The datasets matching a search text, or the first page of all. Blocking."""
    return mikro.get_array_datasets(
        filters=ArrayDatasetFilter(search=text) if text else None,
        pagination=OffsetPaginationInput(limit=PAGE, offset=0),
    )


def describe(dataset: ArrayDataset) -> str:
    """A dataset as a line of the list: its name, its axes and their extents."""
    extents = " ".join(f"{name}:{extent}" for name, extent in zip(dataset.axis_names, dataset.shape))
    return f"{dataset.name}   [{extents}]" + ("   pyramid" if dataset.multiscale else "")


class OpenImageDialog(QtWidgets.QDialog):
    """Pick one of the datasets on the server."""

    def __init__(
        self,
        jobs: Any,
        mikro: Callable[[], Mikro],
        parent: Optional[QtWidgets.QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.jobs = jobs
        self.mikro = mikro
        self.selected: Optional[ArrayDataset] = None
        self.setWindowTitle("Open an image")

        self.label = QtWidgets.QLabel("Select the image you want to open")
        self.search = QtWidgets.QLineEdit()
        self.search.setPlaceholderText("Search by name")
        self.search.returnPressed.connect(self.load)
        self.list = QtWidgets.QListWidget()
        self.list.currentItemChanged.connect(self.on_selected)
        self.list.itemDoubleClicked.connect(lambda _: self.accept())

        self.buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        self.buttons.button(QtWidgets.QDialogButtonBox.Ok).setEnabled(False)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.label)
        layout.addWidget(self.search)
        layout.addWidget(self.list)
        layout.addWidget(self.buttons)
        self.setLayout(layout)

        self.load()

    def load(self) -> None:
        self.label.setText("Loading...")
        self.jobs.submit(
            search, self.mikro(), self.search.text().strip(), done=self.on_loaded, failed=self.on_failed
        )

    def on_loaded(self, datasets: tuple[ArrayDataset, ...]) -> None:
        self.list.clear()
        self.label.setText("Select the image you want to open" if datasets else "No image found")
        for dataset in datasets:
            item = QtWidgets.QListWidgetItem(describe(dataset))
            item.setData(QtCore.Qt.UserRole, dataset)
            self.list.addItem(item)

    def on_failed(self, error: BaseException) -> None:
        self.label.setText(f"Could not load the images: {error}")

    def on_selected(self, item: Optional[QtWidgets.QListWidgetItem], _: Any = None) -> None:
        self.selected = item.data(QtCore.Qt.UserRole) if item is not None else None
        self.buttons.button(QtWidgets.QDialogButtonBox.Ok).setEnabled(self.selected is not None)
