"""The shapes layer that keeps the ROIs of one opened image."""

from dataclasses import dataclass
from typing import Any, Callable, Optional

import napari
import numpy as np
from mikro import Mikro
from qtpy import QtCore

from mikro_napari.models import annotations
from mikro_napari.models.annotations import Frame, Surface
from mikro_napari.models.layers import LayerPlan, Placement

#: How long a shape is left alone before it is stored: a shape being drawn changes
#: with every mouse move.
SETTLE_MS = 400

#: The fewest vertices a shape is stored with; a polygon in the making has fewer.
MINIMUM_VERTICES = {"polygon": 3}


@dataclass(eq=False)
class _Row:
    """What is known of one shape of the layer."""

    id: Optional[str] = None
    storing: bool = False
    dirty: bool = False
    gone: bool = False


class RoiLayer(QtCore.QObject):
    """Shows the annotations drawn on an image, and stores what the user draws.

    There is no subscription to annotations, so the layer shows what was there
    when it was opened or last refreshed.
    """

    created = QtCore.Signal(object)
    loaded = QtCore.Signal(list)
    failed = QtCore.Signal(object)

    def __init__(
        self,
        viewer: napari.Viewer,
        plan: LayerPlan,
        jobs: Any,
        mikro: Callable[[], Mikro],
        physical: bool = True,
        parent: Optional[QtCore.QObject] = None,
    ) -> None:
        super().__init__(parent)
        self.viewer = viewer
        self.plan = plan
        self.jobs = jobs
        self.mikro = mikro
        self.frame = Frame(plan.dims, plan.space_axes)
        self.surface = Surface(plan.dataset, plan.system, plan.name)
        self.rows: list[_Row] = []
        self.skipped = 0
        self._syncing = False
        self._leaving: list[_Row] = []

        placement = plan.placement if physical else Placement()
        self.layer = viewer.add_shapes(
            ndim=len(plan.dims),
            name=f"ROIs for {plan.name}",
            scale=tuple(float(placement.scale.get(dim, 1.0)) for dim in plan.dims),
            translate=tuple(float(placement.translate.get(dim, 0.0)) for dim in plan.dims),
            edge_color="white",
            face_color="transparent",
            metadata={
                "mikro": True,
                "identifier": annotations.IDENTIFIER,
                "dataset": plan.dataset,
                "coordinate_system": plan.coordinate_system,
            },
        )
        self.layer.events.data.connect(self._on_data)

        self._settle = QtCore.QTimer(self)
        self._settle.setSingleShot(True)
        self._settle.setInterval(SETTLE_MS)
        self._settle.timeout.connect(self.flush)

    # ------------------------------------------------------------------ #
    # From the server
    # ------------------------------------------------------------------ #

    def refresh(self) -> None:
        """Read the annotations again and redraw them."""
        self.jobs.submit(
            annotations.fetch,
            self.mikro(),
            self.surface,
            done=self._on_fetched,
            failed=self.failed.emit,
        )

    def _position(self) -> dict[str, int]:
        """Where the viewer's sliders stand, per dim of this layer."""
        point = self.viewer.dims.point[-len(self.frame.dims) :]
        try:
            indices = self.layer.world_to_data(point)
        except Exception:
            return {}
        return {dim: int(round(float(index))) for dim, index in zip(self.frame.dims, indices)}

    def _on_fetched(self, found: list[Any]) -> None:
        at = self._position()
        self._syncing = True
        try:
            self.layer.data = []
            self.rows = []
            self.skipped = 0
            for annotation in found:
                shape = annotations.annotation_to_shape(annotation, self.frame, at)
                if shape is None:
                    self.skipped += 1
                    continue
                self.layer.add(shape[0], shape_type=shape[1])
                self.rows.append(_Row(id=str(annotation.id)))
        finally:
            self._syncing = False
        self.loaded.emit(list(found))

    # ------------------------------------------------------------------ #
    # From the user
    # ------------------------------------------------------------------ #

    def _on_data(self, event: Any) -> None:  # noqa: ANN401
        if self._syncing:
            return
        action = getattr(event.action, "value", event.action)
        count = len(self.layer.data)
        if action == "added":
            for index in sorted(index % count for index in event.data_indices):
                self.rows.insert(index, _Row())
        elif action == "removing":
            self._leaving = [self.rows[index] for index in event.data_indices if index < len(self.rows)]
        elif action == "removed":
            leaving, self._leaving = self._leaving, []
            for row in leaving:
                self.rows.remove(row)
                row.gone = True
                if row.id is not None:
                    self.jobs.submit(annotations.delete, self.mikro(), row.id, failed=self.failed.emit)
        elif action == "changed":
            for index in event.data_indices:
                if index % count < len(self.rows):
                    self.rows[index % count].dirty = True
        else:
            return
        self._settle.start()

    def _drawing(self, row: _Row) -> Optional[annotations.Drawing]:
        index = self.rows.index(row)
        vertices = np.asarray(self.layer.data[index])
        shape_type = self.layer.shape_type[index]
        if len(vertices) < MINIMUM_VERTICES.get(shape_type, 2):
            return None
        return annotations.shape_to_drawing(vertices, shape_type, self.frame)

    def flush(self) -> None:
        """Store what the user drew and changed since the last time."""
        if getattr(self.layer, "_is_creating", False):
            # Still being drawn: its vertices are not the shape yet.
            self._settle.start()
            return
        for row in list(self.rows):
            if row.storing:
                continue
            if row.id is not None and not row.dirty:
                continue
            drawing = self._drawing(row)
            if drawing is None:
                continue
            row.storing = True
            row.dirty = False
            if row.id is None:
                self.jobs.submit(
                    annotations.create,
                    self.mikro(),
                    self.surface,
                    drawing,
                    done=lambda annotation, row=row: self._on_stored(row, annotation, new=True),
                    failed=lambda error, row=row: self._on_failed(row, error),
                )
            else:
                self.jobs.submit(
                    annotations.update,
                    self.mikro(),
                    row.id,
                    drawing,
                    done=lambda annotation, row=row: self._on_stored(row, annotation, new=False),
                    failed=lambda error, row=row: self._on_failed(row, error),
                )

    def _on_stored(self, row: _Row, annotation: Any, new: bool) -> None:  # noqa: ANN401
        row.storing = False
        row.id = str(annotation.id)
        if row.gone:
            # Deleted while it was being stored.
            self.jobs.submit(annotations.delete, self.mikro(), row.id, failed=self.failed.emit)
            return
        if new:
            self.created.emit(annotation)
        if row.dirty:
            self._settle.start()

    def _on_failed(self, row: _Row, error: BaseException) -> None:
        row.storing = False
        self.failed.emit(error)

    def close(self) -> None:
        """Take the layer out of the viewer."""
        self._settle.stop()
        self.layer.events.data.disconnect(self._on_data)
        if self.layer in self.viewer.layers:
            self.viewer.layers.remove(self.layer)
