import os

# Before Qt is imported anywhere: the tests build widgets, and show none.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest


@pytest.fixture(scope="session")
def qapp():
    """A Qt application for the tests that build widgets."""
    QtWidgets = pytest.importorskip("qtpy.QtWidgets")
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
