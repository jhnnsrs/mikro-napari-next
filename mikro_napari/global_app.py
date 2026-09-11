from arkitekt import App
from .manifest import identifier, version, logo
from arkitekt.qt import qt
from qtpy import QtCore, QtWidgets

global_app = None
dev = True


def get_app_or_build_for_widget(widget: QtWidgets.QWidget) -> App:
    """Get the app for the widget or build a new one if it does not exist
    This is a necessary step because we need to attach the app to an existing
    widget. (As opposed to building a new app for each widget) Preferabley
    this would attach directly to the qtviewer, but that is currently deprecated"""

    global global_app
    if global_app is None:
        settings = QtCore.QSettings("napari", f"{identifier}:{version}")

        global_app = qt(
            identifier,
            version,
            parent=widget,
            logo=logo,
        )
    return global_app
