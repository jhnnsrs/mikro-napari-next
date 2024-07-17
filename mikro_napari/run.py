from mikro_napari.widgets.main_widget import MikroNapariWidget
import napari
import argparse
from mikro_napari.controllers.app import AppController
from mikro_napari.widgets.sidebar.sidebar import SidebarWidget
import os
from mikro_napari.global_app import get_app_or_build_for_widget

def main(**kwargs):
    os.environ["NAPARI_ASYNC"] = "1"

    viewer = napari.Viewer()

    app = get_app_or_build_for_widget(viewer.window.qt_viewer)

    sidebar = SidebarWidget(viewer=viewer, app=app, **kwargs)
    widget = MikroNapariWidget(viewer=viewer, app=app, **kwargs)

    AppController(sidebar, app=app, viewer=viewer)
    viewer.window.add_dock_widget(widget, area="left", name="Mikro Napari")
    viewer.window.add_dock_widget(sidebar, area="right", name="Mikro Sidebar")
    # viewer.add_image(astronaut(), name="astronaut")



    napari.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    main(config_path=args.config)
