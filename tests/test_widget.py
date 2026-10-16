"""The widget, driving a viewer, against a stand-in server. Nothing connects."""

from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("qtpy.QtWidgets")
napari = pytest.importorskip("napari")

from arkitekt import connect  # noqa: E402
from arkitekt.qt import QtApp  # noqa: E402
from mikro import mikro_service  # noqa: E402
from mikro.api.schema import AnnotationKind  # noqa: E402
from napari.components import ViewerModel  # noqa: E402

from mikro_napari import global_app  # noqa: E402
from mikro_napari.widgets.main_widget import MikroNapariWidget  # noqa: E402

from .stand_ins import InlineJobs, Server, edge, graph, lens, registration, system  # noqa: E402


class Answer:
    """Stands in for the Qt future and the Qt generator an action answers through."""

    def __init__(self) -> None:
        self.resolved: list = []
        self.rejected: list = []
        self.stopped = False

    def resolve(self, value: object) -> None:
        self.resolved.append(value)

    def reject(self, error: Exception) -> None:
        self.rejected.append(error)

    def next(self, value: object) -> None:
        self.resolved.append(value)

    def throw(self, error: BaseException) -> None:
        self.rejected.append(error)

    def stop(self) -> None:
        self.stopped = True


@pytest.fixture
def server() -> Server:
    return Server()


@pytest.fixture
def widget(qapp, server):
    """The widget on a viewer without a window, over a run that is never entered."""
    # The plugin's own app has a logo, which the bar fetches as it is built: that
    # needs an entered run. The bar is not what is tested here.
    runtime = connect(QtApp("mikro-napari-test", services=[mikro_service]), provide=True)
    built = MikroNapariWidget(ViewerModel(), runtime=runtime, jobs=InlineJobs(), mikro=server)
    yield built
    global_app.set_widget(None)


def test_building_the_widget_names_it_to_the_actions_and_enters_nothing(widget) -> None:
    assert global_app.live_widget() is widget
    assert global_app._runtime is None
    assert widget.runtime.snapshot is None


def test_showing_an_image_adds_a_layer_per_channel_where_it_sits(widget, server) -> None:
    grid, stage = system("10", "czyx"), system("20", "zyx", unit="micrometer")
    server.graph = graph(registration("1", grid, stage, "zyx", [2.0, 0.5, 0.5], [0.0, 10.0, 20.0]))
    image, answer = lens("czyx", (2, 3, 8, 8), grid=grid), Answer()

    global_app.show_on_napari(answer, image, show_roi_layer=False)

    shown = list(widget.viewer.layers)
    assert [layer.name for layer in shown] == ["cells · 0", "cells · 1"]
    assert [layer.data.shape for layer in shown] == [(3, 8, 8)] * 2
    assert [tuple(layer.scale) for layer in shown] == [(2.0, 0.5, 0.5)] * 2
    assert [tuple(layer.translate) for layer in shown] == [(0.0, 10.0, 20.0)] * 2
    assert [layer.metadata["channel_index"] for layer in shown] == [0, 1]
    assert all(layer.metadata["dataset"] == "3" and layer.blending == "additive" for layer in shown)
    assert [str(unit) for unit in shown[0].units] == ["micrometer"] * 3
    assert answer.resolved == [image] and not answer.rejected


def test_an_image_can_be_shown_in_pixels(widget, server) -> None:
    grid, stage = system("10", "yx"), system("20", "yx", unit="micrometer")
    server.graph = graph(registration("1", grid, stage, "yx", [0.5, 0.5], [0.0, 0.0]))

    global_app.show_on_napari(Answer(), lens("yx", (8, 8), grid=grid), show_roi_layer=False, scale_to_physical_size=False)

    assert tuple(widget.viewer.layers[0].scale) == (1.0, 1.0)


def test_a_failure_to_read_the_image_is_the_answer_of_the_action(widget, server) -> None:
    def refuse(*_: object) -> None:
        raise RuntimeError("no such dataset")

    server.rath.query = refuse
    answer = Answer()

    global_app.open_image(answer, lens("yx", (8, 8)))

    assert len(widget.viewer.layers) == 0
    assert [str(error) for error in answer.rejected] == ["no such dataset"]
    assert widget.viewer.status == "no such dataset"


def test_tiled_images_lie_next_to_each_other(widget) -> None:
    images, answer = [lens("yx", (10, 20), id=str(index)) for index in range(2)], Answer()

    global_app.tile_images(answer, images)

    assert [tuple(layer.translate) for layer in widget.viewer.layers] == [(0.0, 0.0), (0.0, 22.0)]
    assert answer.resolved == [images]


def test_the_rois_of_an_image_are_drawn_on_its_grid(widget, server) -> None:
    grid, own = system("10", "tyx"), system("30", "tyx")
    server.graph = graph(edge("1", "IdentityTransformation", own, grid))
    server.collections = [SimpleNamespace(id="1", coordinate_system=own)]
    server.annotations = {
        "1": [
            SimpleNamespace(id="a", kind=AnnotationKind.RECTANGLE, vectors=((1.0, 2.0), (6.0, 9.0)), coordinates=(SimpleNamespace(name="t", value=2),)),
            SimpleNamespace(id="b", kind=AnnotationKind.POINT, vectors=((1.0, 2.0),), coordinates=()),
        ]
    }

    global_app.show_on_napari(Answer(), lens("tyx", (4, 16, 16), grid=grid))

    rois = widget.rois["10"]
    assert rois.layer.name == "ROIs for cells" and rois.layer.ndim == 3
    assert list(rois.layer.shape_type) == ["rectangle"]
    np.testing.assert_allclose(rois.layer.data[0], [[2, 1, 2], [2, 1, 9], [2, 6, 9], [2, 6, 2]])
    assert [row.id for row in rois.rows] == ["a"]
    assert rois.skipped == 1 and "1 annotation(s)" in widget.viewer.status
    # Drawing what the server said is not the user drawing: nothing is stored back.
    rois.flush()
    assert server.calls == []


def test_a_drawn_shape_is_stored_moved_and_deleted_with_the_user(widget, server) -> None:
    global_app.show_on_napari(Answer(), lens("cyx", (2, 16, 16), grid=system("10", "cyx")))
    rois = widget.rois["10"]

    rois.layer.add(np.array([[1.0, 2.0], [1.0, 9.0], [6.0, 9.0]]), shape_type="polygon")
    rois.flush()

    (created,) = server.called("create_annotation")
    assert created["kind"] == AnnotationKind.POLYGON
    # A plane: two components per vertex, and the channel is not a place to pin.
    assert created["vectors"] == [[1.0, 2.0], [1.0, 9.0], [6.0, 9.0]] and created["coordinates"] == []
    assert len(server.called("create_annotation_collection")) == 1
    stored = rois.rows[0].id

    moved = list(rois.layer.data)
    moved[0] = moved[0] + 1.0
    rois.layer.data = moved
    rois.flush()

    (updated,) = server.called("update_annotation")
    assert updated["id"] == stored
    assert updated["vectors"] == [[2.0, 3.0], [2.0, 10.0], [7.0, 10.0]]

    rois.layer.selected_data = {0}
    rois.layer.remove_selected()

    assert rois.rows == []
    assert server.called("delete_annotation") == [{"id": stored}]
    assert len(server.called("create_annotation")) == 1


def test_refreshing_keeps_what_was_drawn_since_the_image_was_opened(widget, server) -> None:
    """The collection of the first shape did not exist when the image was opened."""
    global_app.show_on_napari(Answer(), lens("yx", (16, 16)))
    rois = widget.rois["10"]
    rois.layer.add(np.array([[2.0, 2.0], [8.0, 8.0]]), shape_type="line")
    rois.flush()
    stored = rois.rows[0].id

    widget.refresh_button.click()

    assert [row.id for row in rois.rows] == [stored]
    np.testing.assert_allclose(rois.layer.data[0], [[2.0, 2.0], [8.0, 8.0]])
    assert len(server.called("create_annotation")) == 1


def test_a_polygon_in_the_making_is_not_stored(widget, server) -> None:
    global_app.show_on_napari(Answer(), lens("yx", (16, 16)))
    rois = widget.rois["10"]

    rois.layer.add(np.array([[1.0, 2.0], [1.0, 9.0], [6.0, 9.0]]), shape_type="polygon")
    rois.layer._is_creating = True
    rois.flush()
    assert server.called("create_annotation") == []

    rois.layer._is_creating = False
    rois.flush()
    assert len(server.called("create_annotation")) == 1


def test_streaming_hands_over_each_roi_until_all_are_marked(widget, server) -> None:
    grid = system("10", "yx")
    server.collections = [SimpleNamespace(id="1", coordinate_system=grid)]
    old = SimpleNamespace(id="old", kind=AnnotationKind.LINE, vectors=((1.0, 1.0), (4.0, 4.0)), coordinates=())
    server.annotations = {"1": [old]}
    stream = Answer()

    global_app.stream_rois(stream, lens("yx", (16, 16), grid=grid), show_old_rois=True)
    rois = widget.rois["10"]
    rois.layer.add(np.array([[2.0, 2.0], [8.0, 8.0]]), shape_type="line")
    rois.flush()

    assert [annotation.id for annotation in stream.resolved] == ["old", rois.rows[1].id]
    assert server.called("create_annotation")[0]["collection"] == "1"

    done = widget.mylayout.itemAt(widget.mylayout.count() - 1).widget()
    assert done.text() == "All ROIs marked on cells"
    done.click()
    assert stream.stopped

    rois.layer.add(np.array([[3.0, 3.0], [9.0, 9.0]]), shape_type="line")
    rois.flush()
    assert len(stream.resolved) == 2


def test_the_selected_layer_is_uploaded_and_becomes_a_mikro_layer(widget, server) -> None:
    stored: dict = {}

    def create_array_dataset(**arguments: object) -> SimpleNamespace:
        stored.update(arguments)
        return SimpleNamespace(lens=lambda: SimpleNamespace(id="70", dataset=SimpleNamespace(id="30")))

    server.create_array_dataset = create_array_dataset
    layer = widget.viewer.add_image(np.zeros((4, 16, 12), dtype="uint8"), name="snap")
    widget.viewer.layers.selection = {layer}
    answer = Answer()

    global_app.upload_layer(answer, name="renamed")

    assert stored["name"] == "renamed" and stored["axes"] == ["z", "y", "x"]
    assert answer.resolved[0].id == "70"
    assert layer.metadata["mikro"] is True and layer.metadata["dataset"] == "30"
    assert widget.upload_button.isEnabled()

    # Now on the server: it is not uploaded a second time.
    again = Answer()
    global_app.upload_layer(again)
    assert [str(error) for error in again.rejected] == ["Select an image layer that is not on the server yet."]


def test_the_plugin_connects_in_the_background_and_says_when_it_cannot(qapp, monkeypatch) -> None:
    """Entering a run logs in, so it must not happen in the widget's constructor.

    The run is pointed at a port nothing listens on, with a token and without a
    browser: it can only fail, and it fails without asking anyone anything.
    """
    import time

    made = []

    def unreachable(app, **_: object):
        runtime = connect(
            app, provide=True, token="token", url="http://127.0.0.1:1", skip_cache=True, headless=True
        )
        made.append(runtime)
        return runtime

    monkeypatch.setattr(global_app, "connect", unreachable)
    try:
        widget = MikroNapariWidget(ViewerModel())

        # Built at once, not connected: no bar yet, nothing to press.
        assert widget.magic_bar is None and not widget.open_button.isEnabled()
        assert widget.connecting.text().startswith("Connecting to arkitekt")
        with pytest.raises(RuntimeError, match="Not connected"):
            widget.mikro()

        deadline = time.monotonic() + 60
        while widget.retry_button.isHidden() and widget.magic_bar is None and time.monotonic() < deadline:
            qapp.processEvents()
            time.sleep(0.02)

        assert widget.magic_bar is None, "a run to a dead port cannot have connected"
        assert not widget.retry_button.isHidden()
        assert widget.connecting.text().startswith("Could not connect to arkitekt")
        assert not widget.open_button.isEnabled()

        # Connecting again is a new run: the failed one was used.
        widget.retry_button.click()
        assert len(made) == 2 and widget.runtime is made[1]
        assert widget.retry_button.isHidden()
    finally:
        global_app.set_widget(None)
        global_app.release_runtime()
    assert global_app._runtime is None and global_app._koil is None
