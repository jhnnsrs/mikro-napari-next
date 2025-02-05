from typing import List
from qtpy import QtWidgets
from koil.qt import async_to_qt
from mikro_next.api.schema import (
    Image,
    aget_image,
)
from kraph.api.schema import (
    alist_linked_expressions,
    ExpressionKind,
    LinkedExpressionFilter,
    ListLinkedExpression,
)


class NewRoisEntityDialog(QtWidgets.QDialog):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.setWindowTitle("Select Entity!")
        self.label = QtWidgets.QLabel(
            "Select an entity class you want to associate with the ROIs"
        )

        QBtn = QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel

        self.repList = QtWidgets.QListWidget()

        self.repquery = async_to_qt(alist_linked_expressions)
        self.repquery.started.connect(lambda: self.label.setText("Loading..."))
        self.repquery.returned.connect(self.update_list)
        self.repquery.errored.connect(print)

        self.buttonBox = QtWidgets.QDialogButtonBox(QBtn)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)
        self.buttonBox.buttons()[0].setEnabled(False)

        self.layout = QtWidgets.QVBoxLayout()

        self.reload_button = QtWidgets.QPushButton("Reload")
        self.reload_button.clicked.connect(self.reload)
        self.layout.addWidget(self.reload_button)

        self.layout.addWidget(self.label)
        self.layout.addWidget(self.repList)
        self.layout.addWidget(self.buttonBox)
        self.setLayout(self.layout)

        self.fetch_images_task = self.repquery.run(
            filters=LinkedExpressionFilter(pinned=True, kind=ExpressionKind.ENTITY)
        )
        self.selected_item = None

    def reload(self):
        self.fetch_images_task = self.repquery.run(
            filters=LinkedExpressionFilter(pinned=True, kind=ExpressionKind.ENTITY)
        )

    def on_image_loaded(self, rep: Image):
        self.buttonBox.buttons()[0].setEnabled(True)
        self.label.setText(f"Selected {rep.name} ")
        self.selected_representation = rep

    def update_list(self, exprs: List[ListLinkedExpression]):
        print("HERE")
        print(exprs)
        self.repList.clear()
        self.label.setText("Select an expression")

        print(exprs)

        for expr in exprs:
            item = QtWidgets.QListWidgetItem(
                f"{expr.expression.label}  ({expr.graph.id})"
            )
            item.__linked_id = expr.id
            self.repList.addItem(item)

        self.repList.itemClicked.connect(self.select_rep)

    def select_rep(self, test):
        if self.fetch_images_task and not self.fetch_images_task.done():
            self.fetch_images_task.cancel(wait=True)

        self.selected_item = self.repList.currentItem().__linked_id
        self.accept()
