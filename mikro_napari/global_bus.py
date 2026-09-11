from typing import Callable, Dict, List
from rekuest_widgets.structure import Structure
from qtpy import QtWidgets
from dataclasses import dataclass


@dataclass
class Bus:
    structure_hooks: Dict[str, Callable]

    def add_structure_hook(self, sub: str, hook: Callable[[List[Structure]], bool]):
        self.structure_hooks[sub] = hook

    def get_structure_hook(self, sub: str) -> Callable[[List[Structure]], bool]:
        return self.structure_hooks[sub]

    def run_structure_hook(self, structures: List[Structure]) -> bool:
        """Runs the structure hook for the given structures"""

        for key, hook in self.structure_hooks.items():
            print("Running hook", key)
            hook(structures)

        return True


global_bus = None
dev = True


def get_bus_or_build_bus(widget: QtWidgets.QWidget) -> Bus:
    """Get the app for the widget or build a new one if it does not exist
    This is a necessary step because we need to attach the app to an existing
    widget. (As opposed to building a new app for each widget) Preferabley
    this would attach directly to the qtviewer, but that is currently deprecated"""

    global global_bus
    if global_bus is None:
        global_bus = Bus({})
    return global_bus
