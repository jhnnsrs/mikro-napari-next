"""Between napari shapes and mikro annotations.

A napari vertex has a component per layer dim; a mikro vertex has one per SPACE
axis of the collection it is drawn in, in that system's axis order. Every other
dim a shape sits on (a time point) is a pinned coordinate of the annotation.

The conversions are plain data in, plain data out. The functions at the end ask
the server and block.
"""

import threading
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
from mikro import Mikro
from mikro.api.schema import (
    AnnotationCollectionFilter,
    AnnotationFilter,
    AnnotationKind,
    AxisInput,
    CoordinateInput,
    CoordinateSystemDerivedFromInput,
    IdentityTransformInput,
    OffsetPaginationInput,
)

from .layers import identity_linked

IDENTIFIER = "@mikro/annotation"
PAGE = 200

SHAPE_TO_KIND = {
    "rectangle": AnnotationKind.RECTANGLE,
    "ellipse": AnnotationKind.ELLIPSE,
    "line": AnnotationKind.LINE,
    "path": AnnotationKind.PATH,
    "polygon": AnnotationKind.POLYGON,
}

#: A circle is stored like an ellipse, as its bounding box, so it is drawn as one.
KIND_TO_SHAPE = {
    AnnotationKind.RECTANGLE: "rectangle",
    AnnotationKind.ELLIPSE: "ellipse",
    AnnotationKind.CIRCLE: "ellipse",
    AnnotationKind.LINE: "line",
    AnnotationKind.PATH: "path",
    AnnotationKind.POLYGON: "polygon",
}

#: How many vertices a rotated ellipse is traced with when it is stored as a polygon.
ELLIPSE_VERTICES = 64


@dataclass(frozen=True)
class Frame:
    """The dims of a shapes layer, and which of them are space.

    Attributes:
        dims: The layer's dims, in order.
        space: The SPACE axes, in the order of the coordinate system drawn in.
    """

    dims: tuple[str, ...]
    space: tuple[str, ...]

    @property
    def pinned(self) -> tuple[str, ...]:
        return tuple(dim for dim in self.dims if dim not in self.space)


@dataclass(frozen=True)
class Drawing:
    """A shape as mikro stores it."""

    kind: AnnotationKind
    vectors: list[list[float]]
    coordinates: dict[str, int] = field(default_factory=dict)


def _plane(frame: Frame) -> tuple[int, int]:
    """The two space columns a flat shape is drawn across: the last two."""
    return len(frame.space) - 2, len(frame.space) - 1


def _is_aligned(corners: np.ndarray) -> bool:
    """Whether the four corners of a box follow the axes: each side moves along one."""
    sides = np.roll(corners, -1, axis=0) - corners
    return bool(np.all((np.abs(sides) > 1e-9).sum(axis=1) <= 1))


def shape_to_drawing(vertices: Any, shape_type: str, frame: Frame) -> Drawing:
    """A napari shape as an annotation.

    A rectangle or an ellipse is two opposite corners of its bounding box. One the
    user rotated has no such box, so it is stored as the polygon it traces.

    Args:
        vertices: The shape's vertices, one row per vertex, one column per layer dim.
        shape_type: napari's name of the shape.
        frame: The dims of the layer the shape is on.

    Returns:
        The kind, the vertices over the space axes, and the pinned coordinates.

    Raises:
        ValueError: For a shape type mikro has no kind for.
    """
    if shape_type not in SHAPE_TO_KIND:
        raise ValueError(f"mikro has no annotation kind for a napari '{shape_type}'.")
    points = np.asarray(vertices, dtype=float)
    space = points[:, [frame.dims.index(name) for name in frame.space]]
    coordinates = {name: int(round(points[0, frame.dims.index(name)])) for name in frame.pinned}
    kind = SHAPE_TO_KIND[shape_type]

    if shape_type in ("rectangle", "ellipse"):
        if _is_aligned(space):
            space = np.stack([space.min(axis=0), space.max(axis=0)])
        else:
            kind = AnnotationKind.POLYGON
            if shape_type == "ellipse":
                centre = space.mean(axis=0)
                first = (space[1] - space[0]) / 2
                second = (space[3] - space[0]) / 2
                angles = np.linspace(0, 2 * np.pi, ELLIPSE_VERTICES, endpoint=False)
                space = centre + np.outer(np.cos(angles), first) + np.outer(np.sin(angles), second)
    return Drawing(kind, space.tolist(), coordinates)


def annotation_to_shape(
    annotation: Any, frame: Frame, at: Mapping[str, int] | None = None
) -> tuple[np.ndarray, str] | None:
    """An annotation as a napari shape, or None if this layer cannot draw it.

    Args:
        annotation: The annotation: its ``kind``, ``vectors`` and ``coordinates``.
        frame: The dims of the layer to draw on.
        at: Where to put the shape along a dim the annotation spans. napari draws a
            shape on one position; a dim missing here is drawn at 0.

    Returns:
        The vertices over the layer dims and napari's shape type.
    """
    shape_type = KIND_TO_SHAPE.get(annotation.kind)
    vectors = np.asarray(annotation.vectors, dtype=float)
    if shape_type is None or vectors.ndim != 2 or vectors.shape[1] != len(frame.space):
        return None

    if shape_type in ("rectangle", "ellipse"):
        if len(vectors) < 2 or len(frame.space) < 2:
            return None
        low, high = vectors[0], vectors[1]
        a, b = _plane(frame)
        corners = np.tile(low, (4, 1))
        corners[1, b] = high[b]
        corners[2, [a, b]] = high[[a, b]]
        corners[3, a] = high[a]
        vectors = corners

    pinned = {coordinate.name: coordinate.value for coordinate in annotation.coordinates}
    at = at or {}
    vertices = np.zeros((len(vectors), len(frame.dims)), dtype=float)
    for column, dim in enumerate(frame.dims):
        if dim in frame.space:
            vertices[:, column] = vectors[:, frame.space.index(dim)]
        else:
            vertices[:, column] = pinned.get(dim, at.get(dim, 0))
    return vertices, shape_type


# --------------------------------------------------------------------------- #
# Asking the server
# --------------------------------------------------------------------------- #


@dataclass
class Surface:
    """Where the shapes over one opened grid are drawn.

    Attributes:
        dataset: The dataset the grid belongs to.
        system: The coordinate system of the grid: a lens' own, or the dataset's.
        name: What the grid is called, for naming a collection made for it.
        collection: The collection new shapes go into. Made on the first shape.
    """

    dataset: str
    system: Any
    name: str
    collection: str | None = None
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)


def fetch(mikro: Mikro, surface: Surface) -> list[Any]:
    """The annotations drawn on this very grid. Blocking.

    A dataset's collections include ones drawn on a crop of it, whose vertices are
    in the crop's frame; only collections an identity away from the opened system
    share its pixel indices. The graph is read anew each time: a collection made
    since the image was opened is in no earlier one.
    """
    same_grid = identity_linked(mikro.get_coordinate_graph(surface.system.id), surface.system.id)
    collections = [
        collection
        for collection in mikro.get_annotation_collections(
            filters=AnnotationCollectionFilter(dataset=surface.dataset)
        )
        if str(collection.coordinate_system.id) in same_grid or str(collection.id) == surface.collection
    ]
    collections.sort(key=lambda collection: (len(str(collection.id)), str(collection.id)))
    if collections and surface.collection is None:
        surface.collection = str(collections[0].id)

    annotations: list[Any] = []
    for collection in collections:
        offset = 0
        while True:
            page = mikro.get_annotations(
                filters=AnnotationFilter(collection=collection.id),
                pagination=OffsetPaginationInput(limit=PAGE, offset=offset),
            )
            annotations.extend(page)
            if len(page) < PAGE:
                break
            offset += PAGE
    return annotations


def create(mikro: Mikro, surface: Surface, drawing: Drawing) -> Any:
    """Store a drawing, in the surface's collection. Blocking.

    The collection is made with the first shape: it owns a coordinate system of
    the grid's axes, tied to the grid by an identity, so its shapes are in the
    grid's pixel indices without the two systems being made one.
    """
    with surface.lock:
        if surface.collection is None:
            surface.collection = str(
                mikro.create_annotation_collection(
                    name=f"Drawings on {surface.name}",
                    axes=[
                        AxisInput(name=axis.name, type=axis.type, long_name=axis.long_name)
                        for axis in surface.system.axes
                    ],
                    derived_from=[
                        CoordinateSystemDerivedFromInput(
                            coordinate_system=surface.system.id,
                            transform=IdentityTransformInput(),
                        )
                    ],
                ).id
            )
    return mikro.create_annotation(
        kind=drawing.kind,
        vectors=drawing.vectors,
        collection=surface.collection,
        coordinates=_coordinates(drawing),
    )


def update(mikro: Mikro, annotation: str, drawing: Drawing) -> Any:
    """Store the moved or reshaped drawing of an annotation. Blocking."""
    return mikro.update_annotation(
        id=annotation,
        kind=drawing.kind,
        vectors=drawing.vectors,
        coordinates=_coordinates(drawing),
    )


def delete(mikro: Mikro, annotation: str) -> str:
    """Delete an annotation. Blocking."""
    return str(mikro.delete_annotation(id=annotation))


def _coordinates(drawing: Drawing) -> Sequence[CoordinateInput]:
    return [CoordinateInput(name=name, value=value) for name, value in drawing.coordinates.items()]
