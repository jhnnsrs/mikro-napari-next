import napari

from mikro_napari.widgets.main_widget import MikroNapariWidget


def main() -> None:
    """Start napari with the plugin's widget docked."""
    viewer = napari.Viewer()
    widget = MikroNapariWidget(viewer=viewer)
    viewer.window.add_dock_widget(widget, area="left", name="Mikro Napari")
    napari.run()


if __name__ == "__main__":
    main()
