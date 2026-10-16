"""What the plugin declares, read without connecting anywhere."""

import pytest

pytest.importorskip("qtpy.QtWidgets")


def test_importing_the_app_only_declares(qapp) -> None:
    """A run is made when a widget is built, never by importing the plugin."""
    from mikro_napari import global_app

    assert global_app._runtime is None
    assert global_app._widget is None


def test_the_app_offers_its_five_actions_over_mikro(qapp) -> None:
    from mikro_napari import global_app

    snapshot = global_app.app.snapshot()

    assert list(snapshot.services) == ["mikro"]
    offered = {
        interface: implementation.definition
        for interface, implementation in snapshot.registry.implementations.items()
    }
    assert set(offered) == {"show_on_napari", "open_image", "tile_images", "upload_layer", "stream_rois"}
    assert {definition.name for definition in offered.values()} == {
        "Show on Napari",
        "Show Image",
        "Tile Images on Napari",
        "Upload Napari Layer",
        "Stream ROIs",
    }

    def ports(interface: str) -> tuple[list[str], list[str]]:
        definition = offered[interface]
        return [port.key for port in definition.args], [port.identifier for port in definition.returns]

    # The Qt future and the Qt generator are how an action answers, not arguments of it.
    assert ports("show_on_napari") == (["image", "show_roi_layer", "scale_to_physical_size"], ["@mikro/arraylens"])
    assert ports("upload_layer") == (["name"], ["@mikro/arraylens"])
    assert ports("stream_rois") == (["image", "show_old_rois"], ["@mikro/annotation"])


def test_an_action_without_an_open_widget_says_so(qapp) -> None:
    from mikro_napari import global_app

    global_app.set_widget(None)
    with pytest.raises(RuntimeError, match="widget is not open"):
        global_app.live_widget()
