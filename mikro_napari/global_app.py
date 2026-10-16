"""The app: what this plugin offers, declared once, and the run that serves it.

A run takes a snapshot of the app when it is entered, so everything is declared
here, at import, before any widget exists. The actions are plain functions that
hand over to the widget that is open at the time they are called.

The actions that show something answer with what they showed: an assignment is
done when the layers are in the viewer, and a Qt future has to resolve to a value.
"""

import logging
from typing import TYPE_CHECKING, Any, Optional

from arkitekt import Runtime, connect
from arkitekt.qt import QtApp
from koil import Koil, unkoil
from koil.qt import QtFuture, QtGenerator
from mikro import mikro_service
from mikro.api.schema import Annotation, ArrayLens

from .manifest import identifier, logo, version

if TYPE_CHECKING:
    from .widgets.main_widget import MikroNapariWidget

logger = logging.getLogger(__name__)

app = QtApp(identifier, version, logo=logo, services=[mikro_service])

_widget: Optional["MikroNapariWidget"] = None
_runtime: Optional[Runtime[Any]] = None
_koil: Optional[Koil] = None


def new_runtime() -> Runtime[Any]:
    """A run of the app for the plugin's widget, and the loop it is entered on.

    Nothing connects here. Entering the run resolves mikro, which logs in, so the
    widget enters it on the loop and not in the Qt thread.
    """
    global _koil, _runtime
    if _koil is None:
        _koil = Koil()
        _koil.enter()
    _runtime = connect(app, provide=True)
    return _runtime


def release_runtime() -> None:
    """Leave the run and stop its loop. A run is used once: the next widget connects anew."""
    global _koil, _runtime
    runtime, _runtime = _runtime, None
    koil, _koil = _koil, None
    if runtime is not None and runtime.snapshot is not None:
        try:
            unkoil(runtime.__aexit__, None, None, None)
        except Exception:
            logger.exception("Failed to leave the run of %s", identifier)
    if koil is not None:
        koil.exit()


def set_widget(widget: Optional["MikroNapariWidget"]) -> None:
    """Name the widget the actions act on."""
    global _widget
    _widget = widget


def live_widget() -> "MikroNapariWidget":
    if _widget is None:
        raise RuntimeError("The Mikro Napari widget is not open, so there is no viewer to act on.")
    return _widget


def show_on_napari(
    future: QtFuture[ArrayLens],
    image: ArrayLens,
    show_roi_layer: bool = True,
    scale_to_physical_size: bool = True,
) -> None:
    """Show on Napari

    Loads the image into the viewer, a layer per channel.

    Args:
        image: The image to show.
        show_roi_layer: Whether to also show the ROIs drawn on the image.
        scale_to_physical_size: Whether to place the image at its pixel size.

    Returns:
        The image, once it is shown.
    """
    live_widget().show_image(
        image, future=future, with_rois=show_roi_layer, physical=scale_to_physical_size
    )


def open_image(future: QtFuture[ArrayLens], image: ArrayLens) -> None:
    """Show Image

    Show an image on a bioimage app

    Args:
        image: The image to show.

    Returns:
        The image, once it is shown.
    """
    live_widget().show_image(image, future=future, with_rois=False)


def tile_images(future: QtFuture[list[ArrayLens]], images: list[ArrayLens]) -> None:
    """Tile Images on Napari

    Loads the images and lays them out next to each other in the viewer.

    Args:
        images: The images to tile.

    Returns:
        The images, once they are shown.
    """
    live_widget().tile(images, future=future)


def upload_layer(future: QtFuture[ArrayLens], name: str = "") -> None:
    """Upload Napari Layer

    Upload the selected image layer to the server.

    Args:
        name: Overwrite the layer name.

    Returns:
        The uploaded image.
    """
    live_widget().upload(name=name, future=future)


def stream_rois(generator: QtGenerator[Annotation], image: ArrayLens, show_old_rois: bool = False) -> None:
    """Stream ROIs

    Shows the image and streams the ROIs the user marks on it, until the user
    says all of them are marked.

    Args:
        image: The image to mark ROIs on.
        show_old_rois: Whether to also stream the ROIs that are already there.

    Returns:
        A ROI the user marked.
    """
    live_widget().stream(image, generator=generator, existing=show_old_rois)


app.register_with_qt_future(show_on_napari, name="Show on Napari")
app.register_with_qt_future(open_image, name="Show Image")
app.register_with_qt_future(tile_images, name="Tile Images on Napari")
app.register_with_qt_future(upload_layer, name="Upload Napari Layer")
app.register_with_qt_generator(stream_rois, name="Stream ROIs")
