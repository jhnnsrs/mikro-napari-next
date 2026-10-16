import inspect
from typing import Any, Optional, Sequence

import napari
from arkitekt import Runtime
from arkitekt.qt import MagicBar
from koil.qt import QtFuture, QtGenerator, async_to_qt
from mikro import Mikro
from mikro.api.schema import Annotation, ArrayLens
from napari.utils.colormaps import Colormap
from qtpy import QtWidgets

from mikro_napari import global_app
from mikro_napari.models import layers, upload
from mikro_napari.models.layers import LayerPlan, LayerSpec
from mikro_napari.widgets.dialogs.open_image import OpenImageDialog
from mikro_napari.widgets.jobs import Jobs
from mikro_napari.widgets.roi_layer import RoiLayer

#: Whether this napari names a layer's dims and gives them units (0.5 and later).
HAS_UNITS = "units" in inspect.signature(napari.layers.Image.__init__).parameters


class MikroNapariWidget(QtWidgets.QWidget):
    """Connects napari to arkitekt, and does in the viewer what the app's actions ask.

    Args:
        viewer: The viewer to show images in.
        runtime: An entered run to serve the app through. By default the widget
            makes the plugin's run and enters it in the background.
        jobs: What runs blocking calls off the Qt thread. Worker threads of the
            run by default.
        mikro: The client to use. The run's by default.
        parent: The Qt parent.
    """

    def __init__(
        self,
        viewer: napari.Viewer,
        runtime: Optional[Runtime[Any]] = None,
        jobs: Any = None,
        mikro: Optional[Mikro] = None,
        parent: Optional[QtWidgets.QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.viewer = viewer
        self.owns_runtime = runtime is None
        self.jobs = jobs or Jobs(self)
        self._mikro = mikro
        self.rois: dict[str, RoiLayer] = {}
        self.magic_bar: Optional[MagicBar] = None

        self.connecting = QtWidgets.QLabel()
        self.connecting.setWordWrap(True)
        self.retry_button = QtWidgets.QPushButton("Connect again")
        self.retry_button.clicked.connect(self.connect_runtime)
        self.retry_button.hide()

        self.open_button = QtWidgets.QPushButton("Open image")
        self.open_button.clicked.connect(self.cause_open)
        self.upload_button = QtWidgets.QPushButton("Upload layer")
        self.upload_button.clicked.connect(lambda: self.upload())
        self.refresh_button = QtWidgets.QPushButton("Refresh ROIs")
        self.refresh_button.clicked.connect(self.refresh_rois)

        self.mylayout = QtWidgets.QVBoxLayout()
        self.mylayout.addWidget(self.connecting)
        self.mylayout.addWidget(self.retry_button)
        self.mylayout.addWidget(self.open_button)
        self.mylayout.addWidget(self.upload_button)
        self.mylayout.addWidget(self.refresh_button)
        self.setWindowTitle("Mikro Napari")
        self.setLayout(self.mylayout)

        self.entering = async_to_qt(self._aenter, parent=self)
        self.entering.returned.connect(self._on_entered)
        self.entering.errored.connect(self._on_enter_failed)

        global_app.set_widget(self)
        # Not a method of the widget: by the time Qt says it is destroyed, it is.
        self.destroyed.connect(_closed_shared if self.owns_runtime else _closed)

        if runtime is None:
            self.connect_runtime()
        else:
            # A run someone else made and enters.
            self.runtime = runtime
            self._on_entered()

    # ------------------------------------------------------------------ #
    # Connecting
    # ------------------------------------------------------------------ #

    def connect_runtime(self) -> None:
        """Make a run and enter it, off the Qt thread: entering logs in."""
        self._set_connected(False)
        self.retry_button.hide()
        self.connecting.setText("Connecting to arkitekt... If a login opens in your browser, approve it there.")
        self.connecting.show()
        self.runtime = global_app.new_runtime()
        self.entering.run()

    async def _aenter(self) -> None:
        await self.runtime.__aenter__()

    def _on_entered(self, _: Any = None) -> None:  # noqa: ANN401
        # The bar is built on the entered run: it fetches the app's logo as it is
        # made, and configures, logs in and provides through the run's fakts.
        self.magic_bar = MagicBar(self.runtime, dark_mode=True, on_error=self.on_error)
        self.mylayout.insertWidget(0, self.magic_bar)
        self.connecting.hide()
        self._set_connected(True)

    def _on_enter_failed(self, error: BaseException) -> None:
        self.connecting.setText(f"Could not connect to arkitekt: {error}")
        self.retry_button.show()
        self.on_error(error)

    def _set_connected(self, connected: bool) -> None:
        if not connected and self.magic_bar is not None:
            self.mylayout.removeWidget(self.magic_bar)
            self.magic_bar.deleteLater()
            self.magic_bar = None
        for button in (self.open_button, self.upload_button, self.refresh_button):
            button.setEnabled(connected)

    def mikro(self) -> Mikro:
        """The client of the run."""
        if self._mikro is not None:
            return self._mikro
        if self.magic_bar is None:
            raise RuntimeError("Not connected to arkitekt yet.")
        return self.runtime.require(Mikro)

    def on_error(self, error: BaseException) -> None:
        self.viewer.status = str(error)

    def _fail(self, error: BaseException, future: Any = None) -> None:  # noqa: ANN401
        self.on_error(error)
        if future is not None:
            future.reject(error if isinstance(error, Exception) else RuntimeError(str(error)))

    # ------------------------------------------------------------------ #
    # Showing
    # ------------------------------------------------------------------ #

    def show_image(
        self,
        source: Any,  # noqa: ANN401
        future: Optional[QtFuture[ArrayLens]] = None,
        with_rois: bool = True,
        physical: bool = True,
    ) -> None:
        """Show a lens or a dataset, a layer per channel.

        Args:
            source: The lens, or the dataset to look at whole.
            future: Resolved with the source once its layers are in the viewer.
            with_rois: Whether to also open the layer of its ROIs.
            physical: Whether to place the layers at their pixel size.
        """

        def shown(plan: LayerPlan) -> None:
            try:
                self.add_plan(plan, physical)
                if with_rois:
                    self.open_rois(plan, physical)
            except Exception as error:
                self._fail(error, future)
                return
            if future is not None:
                future.resolve(source)

        try:
            self.jobs.submit(
                layers.prepare,
                self.mikro(),
                source,
                done=shown,
                failed=lambda error: self._fail(error, future),
            )
        except Exception as error:
            self._fail(error, future)

    def add_plan(
        self, plan: LayerPlan, physical: bool = True, offset: Optional[dict[str, float]] = None
    ) -> list[Any]:
        """Add the layers of a plan to the viewer."""
        return [self._add_layer(plan, spec, physical, offset) for spec in plan.specs]

    def _add_layer(
        self, plan: LayerPlan, spec: LayerSpec, physical: bool, offset: Optional[dict[str, float]]
    ) -> Any:  # noqa: ANN401
        arguments = layers.layer_arguments(plan, spec, physical, offset)
        if spec.rgb is None:
            if spec.colormap is not None:
                arguments["colormap"] = spec.colormap
            elif spec.color is not None:
                # Named by its colour and not by the layer: the layer controls are
                # as wide as the longest colormap name napari knows.
                red, green, blue = (round(255 * float(part)) for part in spec.color[:3])
                arguments["colormap"] = Colormap(
                    [[0, 0, 0, 1], [*spec.color, 1]], name=f"#{red:02x}{green:02x}{blue:02x}"
                )
        data = layers.layer_data(plan, spec)
        data = [level.data for level in data] if isinstance(data, list) else data.data

        if HAS_UNITS:
            named = dict(arguments, axis_labels=spec.dims)
            if physical and plan.placement.units:
                named["units"] = tuple(plan.placement.units.get(dim, "pixel") for dim in spec.dims)
            try:
                return self.viewer.add_image(data, **named)
            except (ValueError, TypeError):
                # A unit napari does not know: the layer is shown without it.
                pass
        return self.viewer.add_image(data, **arguments)

    def tile(self, images: Sequence[ArrayLens], future: Optional[QtFuture[list[ArrayLens]]] = None) -> None:
        """Show several images next to each other."""

        def prepare_all(mikro: Mikro, sources: Sequence[ArrayLens]) -> list[LayerPlan]:
            return [layers.prepare(mikro, source) for source in sources]

        def shown(plans: list[LayerPlan]) -> None:
            try:
                for plan, offset in zip(plans, layers.grid_offsets(plans)):
                    self.add_plan(plan, offset=offset)
            except Exception as error:
                self._fail(error, future)
                return
            if future is not None:
                future.resolve(list(images))

        try:
            self.jobs.submit(
                prepare_all,
                self.mikro(),
                list(images),
                done=shown,
                failed=lambda error: self._fail(error, future),
            )
        except Exception as error:
            self._fail(error, future)

    def cause_open(self) -> None:
        dialog = OpenImageDialog(self.jobs, self.mikro, parent=self)
        if dialog.exec() and dialog.selected is not None:
            self.show_image(dialog.selected)

    # ------------------------------------------------------------------ #
    # ROIs
    # ------------------------------------------------------------------ #

    def rois_of(self, plan: LayerPlan, physical: bool = True) -> Optional[RoiLayer]:
        """The ROI layer of a shown image, opened if it is not; None if it has no grid to draw in."""
        if plan.coordinate_system is None:
            return None
        rois = self.rois.get(plan.coordinate_system)
        if rois is None or rois.layer not in self.viewer.layers:
            rois = RoiLayer(self.viewer, plan, self.jobs, self.mikro, physical=physical, parent=self)
            rois.failed.connect(self.on_error)
            rois.loaded.connect(lambda _, rois=rois: self._on_rois_loaded(rois))
            self.rois[plan.coordinate_system] = rois
        return rois

    def open_rois(self, plan: LayerPlan, physical: bool = True) -> Optional[RoiLayer]:
        """Open the ROI layer of a shown image and read its annotations."""
        rois = self.rois_of(plan, physical)
        if rois is not None:
            rois.refresh()
        return rois

    def _on_rois_loaded(self, rois: RoiLayer) -> None:
        if rois.skipped:
            self.viewer.status = (
                f"{rois.skipped} annotation(s) of {rois.plan.name} are of a kind a shapes layer cannot draw."
            )

    def refresh_rois(self) -> None:
        for rois in self.rois.values():
            if rois.layer in self.viewer.layers:
                rois.refresh()

    def stream(self, image: ArrayLens, generator: QtGenerator[Annotation], existing: bool = False) -> None:
        """Show an image and hand over the ROIs the user marks on it, until told they are all marked."""

        def opened(rois: Optional[RoiLayer]) -> None:
            if rois is None:
                generator.throw(RuntimeError("The image has no coordinate system to draw ROIs in."))
                return
            done = QtWidgets.QPushButton(f"All ROIs marked on {rois.plan.name}")

            def all_marked() -> None:
                rois.created.disconnect(generator.next)
                if existing:
                    rois.loaded.disconnect(replay)
                self.mylayout.removeWidget(done)
                done.deleteLater()
                rois.flush()
                generator.stop()

            def replay(found: list[Any]) -> None:
                for annotation in found:
                    generator.next(annotation)

            rois.created.connect(generator.next)
            if existing:
                rois.loaded.connect(replay)
            done.clicked.connect(all_marked)
            self.mylayout.addWidget(done)

        def failed(error: BaseException) -> None:
            self.on_error(error)
            generator.throw(error)

        try:
            self.jobs.submit(
                layers.prepare,
                self.mikro(),
                image,
                done=lambda plan: self._show_for_stream(plan, opened, failed),
                failed=failed,
            )
        except Exception as error:
            failed(error)

    def _show_for_stream(self, plan: LayerPlan, opened: Any, failed: Any) -> None:  # noqa: ANN401
        try:
            self.add_plan(plan)
            rois = self.rois_of(plan)
            # Told before the layer reads its annotations, so the ones that are
            # already there reach whoever asked for them.
            opened(rois)
            if rois is not None:
                rois.refresh()
        except Exception as error:
            failed(error)

    # ------------------------------------------------------------------ #
    # Uploading
    # ------------------------------------------------------------------ #

    def uploadable_layer(self) -> Any:  # noqa: ANN401
        """The selected image layer that is not on the server yet."""
        for layer in self.viewer.layers.selection:
            if isinstance(layer, napari.layers.Image) and not layer.metadata.get("mikro"):
                return layer
        raise RuntimeError("Select an image layer that is not on the server yet.")

    def upload(self, name: str = "", future: Optional[QtFuture[ArrayLens]] = None) -> None:
        """Store the selected image layer as a dataset."""
        try:
            layer = self.uploadable_layer()
            picture = upload.Picture(
                data=layer.data[0] if layer.multiscale else layer.data,
                rgb=bool(layer.rgb),
                name=name or layer.name,
                scale=tuple(float(s) for s in layer.scale),
                translate=tuple(float(t) for t in layer.translate),
                units=tuple(str(unit) for unit in layer.units) if HAS_UNITS else None,
            )

            def uploaded(lens: ArrayLens) -> None:
                layer.metadata.update(
                    {
                        "mikro": True,
                        "identifier": layers.IDENTIFIER,
                        "dataset": str(lens.dataset.id),
                        "lens": str(lens.id),
                    }
                )
                self.upload_button.setText("Upload layer")
                self.upload_button.setEnabled(True)
                if future is not None:
                    future.resolve(lens)

            def failed(error: BaseException) -> None:
                self.upload_button.setText("Upload layer")
                self.upload_button.setEnabled(True)
                self._fail(error, future)

            self.upload_button.setText(f"Uploading {picture.name}...")
            self.upload_button.setEnabled(False)
            self.jobs.submit(upload.upload, self.mikro(), picture, done=uploaded, failed=failed)
        except Exception as error:
            self.upload_button.setText("Upload layer")
            self.upload_button.setEnabled(True)
            self._fail(error, future)


def _closed() -> None:
    global_app.set_widget(None)


def _closed_shared() -> None:
    global_app.set_widget(None)
    global_app.release_runtime()
