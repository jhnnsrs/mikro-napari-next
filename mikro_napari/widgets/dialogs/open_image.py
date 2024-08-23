from typing import List
from qtpy import QtWidgets
from koil.qt import QtRunner
from mikro_next.api.schema import Image, aget_image


class OpenImageDialog(QtWidgets.QDialog):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.setWindowTitle("Select Image!")
        self.label = QtWidgets.QLabel("Select An image that you want to open")

        QBtn = QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel

        self.repList = QtWidgets.QListWidget()

        aget_images = None

        self.repquery = QtRunner(aget_images)
        self.repquery.started.connect(lambda: self.label.setText("Loading..."))
        self.repquery.returned.connect(self.update_list)
        self.repquery.errored.connect(print)

        self.detailquery = QtRunner(aget_image)
        self.detailquery.started.connect(
            lambda: self.buttonBox.buttons()[0].setEnabled(False)
        )
        self.detailquery.errored.connect(lambda e: self.label.setText("Error loading"))
        self.detailquery.returned.connect(self.on_image_loaded)

        self.buttonBox = QtWidgets.QDialogButtonBox(QBtn)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)
        self.buttonBox.buttons()[0].setEnabled(False)

        self.layout = QtWidgets.QVBoxLayout()

        self.layout.addWidget(self.label)
        self.layout.addWidget(self.repList)
        self.layout.addWidget(self.buttonBox)
        self.setLayout(self.layout)

        self.fetch_images_task = self.repquery.run()
        self.fetch_image_task = None

    def on_image_loaded(self, rep: Image):
        self.buttonBox.buttons()[0].setEnabled(True)
        self.label.setText(f"Selected {rep.name} ")
        self.selected_representation = rep

    def update_list(self, reps: List[Image]):
        self.repList.clear()
        self.label.setText("Select An image that you want to open")

        for rep in reps:
            item = QtWidgets.QListWidgetItem(
                f"{rep.name}  { 'on '  + rep.sample.name if rep.sample else ''}"
            )
            item.__repid = rep.id
            self.repList.addItem(item)

        self.repList.itemClicked.connect(self.select_rep)

    def select_rep(self, test):
        if self.fetch_images_task and not self.fetch_images_task.done():
            self.fetch_images_task.cancel(wait=True)

        current_item = self.repList.currentItem().__repid
        self.fetch_image_task = self.detailquery.run(current_item)
