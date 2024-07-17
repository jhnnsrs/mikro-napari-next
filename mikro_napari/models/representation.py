import asyncio
import math
from typing import Dict, List
from arkitekt import App
import dask.array as da
import napari
import numpy as np
from napari.layers.shapes._shapes_constants import Mode
from qtpy import QtCore, QtWidgets
from koil.qt import QtCoro, QtFuture, QtGeneratorRunner, QtRunner, QtSignal
from mikro_next.api.schema import AffineTransformationView, Image, RoiKind, ROI, Stage, acreate_roi, awatch_rois, adelete_roi, aget_rois, WatchRoisSubscription, FiveDVector, delete_roi, ColorMap
from mikro_napari.utils import NapariROI, convert_roi_to_napari_roi
import vispy.color


DESIGN_MODE_MAP = {
    Mode.ADD_RECTANGLE: RoiKind.RECTANGLE,
    Mode.ADD_ELLIPSE: RoiKind.ELLIPSIS,
    Mode.ADD_LINE: RoiKind.LINE,
}

SELECT_MODE_MAP = {
    Mode.DIRECT: "direct",
}


DOUBLE_CLICK_MODE_MAP = {
    Mode.ADD_POLYGON: RoiKind.POLYGON,
    Mode.ADD_PATH: RoiKind.PATH,
}




class AskForRoi(QtWidgets.QWidget):
    def __init__(
        self,
        controller,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.button = QtWidgets.QPushButton("All Rois Marked")
        self.button.clicked.connect(self.on_done)
        self.mylayout = controller.widget.mylayout

    def ask(self, qt_generator):
        self.qt_generator = qt_generator
        self.mylayout.addWidget(self.button)
        self.mylayout.update()

    def on_done(self) -> None:
        self.qt_generator.stop()
        self.mylayout.removeWidget(self.button)
        self.button.setParent(None)
        self.mylayout.update()


class TaskDone(QtWidgets.QWidget):
    def __init__(
        self,
        controller,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.mylayout = QtWidgets.QHBoxLayout()
        controller.widget.mylayout.addLayout(self.mylayout)
        self.listeners = {}
        self.buttons = {}
        self.futures = {}
        self.ask_coro = QtCoro(self.ask)
        self.ask_coro.cancelled.connect(self.on_cancelled)

    def ask(self, future: QtFuture, text):
        button = QtWidgets.QPushButton(text)
        button.clicked.connect(lambda: self.on_done(future))
        self.futures[future.id] = future
        self.buttons[future.id] = button
        self.update_buttons()

    def on_done(self, future) -> None:
        future.resolve(True)
        del self.buttons[future.id]
        self.update_buttons()

    def clearLayout(self, layout):
        while layout.count():
            child = layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

    def update_buttons(self):
        self.clearLayout(self.mylayout)
        for button in self.buttons.values():
            self.mylayout.addWidget(button)

        self.mylayout.update()

    def on_cancelled(self, future):
        del self.buttons[future.id]
        self.update_buttons()


class ManagedLayer(QtCore.QObject):
    def __init__(self, *args, viewer: napari.Viewer = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        assert viewer is not None, "Managed Layer needs access to the viewer"
        self.viewer = viewer
        self.managed_layers = {}

    def add_layer(self, layerid: str, layer: "ManagedLayer"):
        self.managed_layers[layerid] = layer

    def remove_layer(self, layerid: str):
        self.managed_layers.pop(layerid)

    def on_destroy(self):
        pass

    def destroy(self):
        for layer in self.managed_layers.values():
            layer.destroy()

        self.on_destroy()


class RoiLayer(ManagedLayer):
    roi_user_created = QtCore.Signal(ROI)
    roi_user_deleted = QtCore.Signal(str)
    roi_user_updated = QtCore.Signal(ROI)
    rois_user_selected = QtCore.Signal(list)
    roi_event_created = QtCore.Signal(ROI)
    roi_event_deleted = QtCore.Signal(str)
    roi_event_updated = QtCore.Signal(ROI)

    def __init__(
        self,
        image: Image,
        scale_to_physical_size: bool = True,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.image = image
        self.get_rois_query = QtRunner(aget_rois)
        self.get_rois_query.returned.connect(self._on_rois_loaded)
        self.get_rois_query.errored.connect(print)

        self.create_rois_runner = QtRunner(acreate_roi)
        self.create_rois_runner.returned.connect(self.on_roi_created)
        self.create_rois_runner.errored.connect(print)

        self.delete_rois_runner = QtRunner(adelete_roi)
        self.delete_rois_runner.returned.connect(self.on_roi_deleted)
        self.delete_rois_runner.errored.connect(print)

        self.watch_rois_subscription = QtGeneratorRunner(awatch_rois)
        self.watch_rois_subscription.yielded.connect(self.on_rois_updated)
        self.watch_rois_subscription.errored.connect(print)

        self.scale_to_physical_size = scale_to_physical_size
        self.koiled_create_rois = QtSignal(self.create_rois_runner.returned)

        self.layer = None
        self._get_rois_future = None
        self.is_watching = False
        self._watch_rois_future = None
        self._napari_rois: List[NapariROI] = []
        self._roi_layer = None
        self.roi_state = {}

    def on_destroy(self):
        self.viewer.remove_layer(self.layer)
        if self._get_rois_future is not None and not self._get_rois_future.done():
            self._get_rois_future.cancel()
        if self._watch_rois_future is not None and not self._watch_rois_future.done():
            self._watch_rois_future.cancel()

    def show(self, fetch_rois=True, watch_rois=True):
        self._roi_layer = self.viewer.add_shapes()
        self._roi_layer.mouse_drag_callbacks.append(self.on_drag_roi_layer)
        self._roi_layer.mouse_double_click_callbacks.append(
            self.on_double_click_roi_layer
        )
        if fetch_rois:
            self.show_rois()

        if watch_rois:
            self.watch_rois()

    def on_roi_deleted(self, result: str):
        if not self.is_watching:
            del self.roi_state[str(result)]

        self.roi_user_deleted.emit(str(result))

    def on_roi_created(self, result: ROI):
        if not self.is_watching:
            self.roi_state[result.id] = result

        self.roi_user_created.emit(result)

    def show_rois(self):
        self._get_rois_future = self.get_rois_query.run(image=self.image.id)

    def watch_rois(self):
        self.is_watching = True
        self._watch_rois_future = self.watch_rois_subscription.run(
            image=self.image.id
        )

    def update_roi_layer(self):
        self._napari_rois: List[NapariROI] = list(
            filter(
                lambda x: x is not None,
                [convert_roi_to_napari_roi(roi) for roi in self.roi_state.values()],
            )
        )
        self._roi_layer.name = f"ROIs for {self.image.name}"

        
        # Clear the selected data and remove the selected
        # This is hacky as fuck but self._roi_layer.data = [] does not longer work
        self._roi_layer.selected_data = set(range(self._roi_layer.nshapes))
        self._roi_layer.remove_selected()
        

        for i in self._napari_rois:
            print(i)
            self._roi_layer.add(
                i.data,
                shape_type=i.type,
                edge_width=1,
                edge_color="white",
                face_color=i.color,
            )

        self._roi_layer.features = {"roi": [r.id for r in self._napari_rois]}

    def _on_rois_loaded(self, rois: List[ROI]):
        self.roi_state = {roi.id: roi for roi in rois}
        self.update_roi_layer()

    def on_rois_updated(self, ev: WatchRoisSubscription):
        if ev.create:
            self.roi_state[ev.create.id] = ev.create
            self.roi_event_created.emit(ev.create)

        if ev.delete:
            del self.roi_state[ev.delete]
            self.roi_event_deleted.emit(str(ev.delete))

        self.update_roi_layer()

    def on_drag_roi_layer(self, layer, event):
        while event.type != "mouse_release":
            yield

        print("dragged")
        if layer.mode in SELECT_MODE_MAP:
            print(self._roi_layer.selected_data)
            selected_rois = []
            for i in self._roi_layer.selected_data:
                napari_roi = self._napari_rois[i]
                selected_rois.append(self.roi_state[napari_roi.id])

            self.rois_user_selected.emit(selected_rois)

        if layer.mode in DESIGN_MODE_MAP:
            if len(self._roi_layer.data) > len(self._napari_rois):
                c, t, z = event.position[:3]
                print("Creating")

                vectors = FiveDVector.list_from_numpyarray(
                        self._roi_layer.data[-1], t=t, z=z, c=c
                    )
                
                print("The vectors", vectors)

                self.create_rois_runner.run(
                    image=self.image.id,
                    vectors=vectors,
                    kind=DESIGN_MODE_MAP[layer.mode],
                )

        if len(self._roi_layer.data) < len(self._napari_rois):
            if "roi" in self._roi_layer.features:
                there_rois = set([f for f in self._roi_layer.features["roi"]])
                state_rois = set([f.id for f in self._napari_rois])
                difference_rois = state_rois - there_rois
                for roi_id in difference_rois:
                    self.delete_rois_runner.run(roi_id)

    def on_double_click_roi_layer(self, layer, event):
        print("double clicked")
        if layer.mode in DOUBLE_CLICK_MODE_MAP:
            if len(self._roi_layer.data) > len(self._napari_rois):
                t, z, c = event.position[:3]

                self.create_rois_runner.run(
                    image=self.image.id,
                    vectors=FiveDVector.list_from_numpyarray(
                        self._roi_layer.data[-1], t=t, z=z, c=c
                    ),
                    kind=DOUBLE_CLICK_MODE_MAP[layer.mode],
                )


class ImageLayer(ManagedLayer):
    on_rep_layer_clicked = QtCore.Signal(Image)

    def __init__(
        self,
        image: Image,
        scale_to_physical_size: bool = False,
        with_rois=True,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.managed_image = image
        self._image_layers = []
        self.with_rois = with_rois
        self.scale_to_physical_size = scale_to_physical_size
        self.roi_layer = None

    def on_destroy(self):
        for layer in self._image_layers:
            self.viewer.remove_layer(layer)

    def show(self, respect_physical_size=False):
        if respect_physical_size:
            raise NotImplementedError

        scale = None

        contexts = self.managed_image.rgb_contexts

        affinetransformation = [i for i in self.managed_image.views if isinstance(i, AffineTransformationView)]

        if affinetransformation:
            affinetransformation = affinetransformation[0]
            scaleX = affinetransformation.affine_matrix[0][0]
            scaleY = affinetransformation.affine_matrix[1][1]
            scaleZ = affinetransformation.affine_matrix[2][2]
            scale = (scaleZ, scaleY, scaleX)
        else:
            scale = (1, 1, 1)


        if contexts:
            context = contexts[0]

            for view in context.views:

                print(self.managed_image.data)

                if view.color_map == ColorMap.INTENSITY:
                    colormap = f"Intensity rgba({(',').join([str(i) for i in view.base_color])})", vispy.color.Colormap([[0, 0, 0, 0], [i / 255 for i in view.base_color]])
                else:
                    colormap = view.color_map


                new_layer = self.viewer.add_image(
                    self.managed_image.data.isel(c=view.c_min).transpose(*list("tzyx")),
                    metadata={
                        "mikro": True,
                        "representation": self.managed_image,
                        "type": "IMAGE",
                    },
                    colormap=colormap,
                    scale=scale,
                )

                self._image_layers.append(
                    new_layer
                )

        else:
            new_layer = self.viewer.add_image(
                self.managed_image.data.transpose(*list("ctzyx")),
                metadata={
                    "mikro": True,
                    "representation": self.managed_image,
                    "type": "IMAGE",
                },
                scale=scale,
            )

            self._image_layers.append(new_layer)

        print(scale)


class RepresentationQtModel(QtCore.QObject):
    rep_changed = QtCore.Signal(Image)

    def __init__(self, widget, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.widget = widget
        self.app: App = self.widget.app
        self.viewer: napari.Viewer = self.widget.viewer

        self.managed_layers: Dict[str, ImageLayer] = {}
        self.managed_roi_layers: Dict[str, RoiLayer] = {}

        self.ask_roi_dialog = AskForRoi(self)
        self.task_done_dialog = TaskDone(self)

        self._image_layer = None
        self._roi_layer = None
        self.roi_state: Dict[str, ROI] = {}

        self.create_image_layer_coro = QtCoro(self.create_image_layer, autoresolve=True)
        self.create_roi_layer_coro = QtCoro(self.create_roi_layer, autoresolve=True)

    def create_image_layer(
        self, image: Image, scale_to_physical_size: bool = True
    ) -> ImageLayer:
        if image.id not in self.managed_layers:
            layer = ImageLayer(
                image, viewer=self.viewer, scale_to_physical_size=scale_to_physical_size
            )
            self.managed_layers[image.id] = layer
        else:  # pragma: no cover
            layer = self.managed_layers[image.id]

        layer.show()
        return layer

    def create_roi_layer(
        self,
        image: Image,
        scale_to_physical_size: bool = True,
        fetch_rois=True,
        watch_rois=True,
    ) -> RoiLayer:
        if image.id not in self.managed_roi_layers:
            layer = RoiLayer(
                image, viewer=self.viewer, scale_to_physical_size=scale_to_physical_size
            )
            self.managed_roi_layers[image.id] = layer
        else:  # pragma: no cover
            layer = self.managed_roi_layers[image.id]

        layer.show(fetch_rois=fetch_rois, watch_rois=watch_rois)
        return layer

    def on_image_loaded(
        self,
        rep: Image,
        show_roi_layer: bool = True,
        scale_to_physical_size: bool = True,
    ):
        """Show on Napari

        Loads the image into the viewer

        Args:
            rep (RepresentationFragment): The Image
        """
        self.create_image_layer(rep, scale_to_physical_size=scale_to_physical_size)
        if show_roi_layer:
            self.create_roi_layer(rep)

    def open_image(
        self,
        image: Image,
    ) -> None:
        """Show Image

        Show an image on a bioimage app

        Parameters
        ----------
        a : RepresentationFragment
            The image

        """
        self.create_image_layer(image)

    
    def tile_images(self, reps: List[Image]):
        """Tile Images on Napari

        Loads the images and tiles them into the viewer

        Args:
            reps (List[RepresentationFragment]): The Image
        """

        shape_array = np.array([np.array(rep.data.shape[:4]) for rep in reps])
        max_shape = np.max(shape_array, axis=0)

        cdata = []
        for rep in reps:
            data = da.zeros(list(max_shape) + [rep.data.shape[4]])
            data[
                : rep.data.shape[0],
                : rep.data.shape[1],
                : rep.data.shape[2],
                : rep.data.shape[3],
                :,
            ] = rep.data
            cdata.append(data)

        x = da.concatenate(cdata, axis=-1).squeeze()
        name = " ".join([rep.name for rep in reps])

        self.viewer.add_image(
            x,
            name=name,
            metadata={"mikro": True, "type": "IMAGE"},
            scale=reps[0].omero.scale if reps[0].omero else None,
        )

    async def stream_rois(
        self, rep: Image, show_old_rois=False
    ) -> ROI:
        """Stream ROIs

        Asks the user to mark rois on the image, once user deams done, the rois are returned

        Args:
            rep (RepresentationFragment): The Image
            show_old_rois (bool, optional): Show already marked rois. Defaults to False.

        Returns:
            rois (List[RoiFragment]): The Image
        """
        await self.create_image_layer_coro.acall(rep)
        roilayer = await self.create_roi_layer_coro.acall(rep, fetch_rois=show_old_rois)

        create_listener = roilayer.koiled_create_rois
        stop_task = asyncio.create_task(
            self.task_done_dialog.ask_coro.acall("All rois marked?")
        )
        try:
            while True:
                x = asyncio.create_task(create_listener.aonce())

                done, pending = await asyncio.wait(
                    [x, stop_task],
                    return_when=asyncio.FIRST_COMPLETED,
                )

                for i in done:
                    if i == stop_task:
                        for p in pending:
                            p.cancel()
                        try:
                            await asyncio.gather(*pending)
                        except asyncio.CancelledError:
                            print("Cancelled")
                            pass
                        return
                    else:
                        roi = i.result()
                        yield roi

        except asyncio.CancelledError:
            print("Cancelled")
            if not stop_task.done():
                stop_task.cancel()

            if not x.done():
                x.cancel()

            try:
                await asyncio.gather(stop_task, x)
            except asyncio.CancelledError:
                print("Cancelled")
                pass

   

    def on_rois_loaded(self, rois: List[ROI]):
        self.roi_state = {roi.id: roi for roi in rois}
        self.update_roi_layer()

    def on_rois_updated(self, ev: WatchRoisSubscription):
        if ev.create:
            self.roi_state[ev.create.id] = ev.create
            if self.stream_roi_generator:
                self.stream_roi_generator.next(ev.create)

        if ev.delete:
            del self.roi_state[ev.delete]

        self.update_roi_layer()

    def on_drag_image_layer(self, layer, event):
        while event.type != "mouse_release":
            yield

    def on_drag_roi_layer(self, layer, event):
        while event.type != "mouse_release":
            yield

        if layer.mode in SELECT_MODE_MAP:
            print(self._roi_layer.selected_data)
            for i in self._roi_layer.selected_data:
                napari_roi = self._napari_rois[i]
                self.viewer.window.sidebar.select_roi(napari_roi)

        if layer.mode in DESIGN_MODE_MAP:
            if len(self._roi_layer.data) > len(self._napari_rois):
                c, t, z = event.position[:3]

                self.create_rois_runner.run(
                    representation=self._active_representation.id,
                    vectors=FiveDVector.list_from_numpyarray(
                        self._roi_layer.data[-1], t=t, z=z, c=c
                    ),
                    type=DESIGN_MODE_MAP[layer.mode],
                )

        if len(self._roi_layer.data) < len(self._napari_rois):
            there_rois = set([f for f in self._roi_layer.features["roi"]])
            state_rois = set([f.id for f in self._napari_rois])
            difference_rois = state_rois - there_rois
            for roi_id in difference_rois:
                delete_roi(roi_id)

    def on_double_click_roi_layer(self, layer, event):
        print("Fired")
        print(self._roi_layer.features)
        if layer.mode in DOUBLE_CLICK_MODE_MAP:
            if len(self._roi_layer.data) > len(self._napari_rois):
                t, z, c = event.position[:3]

                self.create_rois_runner.run(
                    representation=self._active_representation.id,
                    vectors=FiveDVector.list_from_numpyarray(
                        self._roi_layer.data[-1], t=t, z=z, c=c
                    ),
                    type=DOUBLE_CLICK_MODE_MAP[layer.mode],
                )

    def print(self, *args, **kwargs):
        print(*args, **kwargs)

    def update_roi_layer(self):
        if not self._roi_layer:
            self._roi_layer = self.viewer.add_shapes(
                metadata={
                    "mikro": True,
                    "type": "ROIS",
                    "representation": self._active_representation,
                }
            )
            self._roi_layer.mouse_drag_callbacks.append(self.on_drag_roi_layer)
            self._roi_layer.mouse_double_click_callbacks.append(
                self.on_double_click_roi_layer
            )

        self._napari_rois: List[NapariROI] = list(
            filter(
                lambda x: x is not None,
                [convert_roi_to_napari_roi(roi) for roi in self.roi_state.values()],
            )
        )

        self._roi_layer.data = []
        self._roi_layer.name = f"ROIs for {self._active_representation.name}"

        for i in self._napari_rois:
            self._roi_layer.add(
                i.data,
                shape_type=i.type,
                edge_width=1,
                edge_color="white",
                face_color=i.color,
            )

        self._roi_layer.features = {"roi": [r.id for r in self._napari_rois]}
