"""Between napari shapes and mikro annotations."""

from types import SimpleNamespace

import numpy as np
import pytest
from mikro.api.schema import AnnotationKind

from mikro_napari.models import annotations
from mikro_napari.models.annotations import Drawing, Frame, Surface

from .stand_ins import Server, edge, graph, system

PLANE = Frame(dims=("y", "x"), space=("y", "x"))
MOVIE = Frame(dims=("t", "z", "y", "x"), space=("z", "y", "x"))


def stored(drawing: Drawing) -> SimpleNamespace:
    """A drawing as the server hands it back."""
    return SimpleNamespace(
        kind=drawing.kind,
        vectors=tuple(tuple(vector) for vector in drawing.vectors),
        coordinates=tuple(SimpleNamespace(name=name, value=value) for name, value in drawing.coordinates.items()),
    )


def test_a_shape_on_a_plane_has_two_components_per_vertex() -> None:
    """A plane has two space axes, so its vertices are (y, x): nothing is padded."""
    triangle = [[1.0, 2.0], [1.0, 9.0], [6.0, 9.0]]
    drawing = annotations.shape_to_drawing(triangle, "polygon", PLANE)
    assert drawing == Drawing(AnnotationKind.POLYGON, triangle, {})


def test_a_shape_in_a_movie_keeps_its_plane_and_pins_its_time_point() -> None:
    """The space components are (z, y, x); the slider it was drawn at is a coordinate."""
    line = [[4.0, 2.0, 10.0, 11.0], [4.0, 2.0, 30.0, 31.0]]
    drawing = annotations.shape_to_drawing(line, "line", MOVIE)
    assert drawing == Drawing(AnnotationKind.LINE, [[2.0, 10.0, 11.0], [2.0, 30.0, 31.0]], {"t": 4})


def test_a_rectangle_is_stored_as_two_opposite_corners() -> None:
    corners = [[1.0, 2.0], [1.0, 9.0], [6.0, 9.0], [6.0, 2.0]]
    drawing = annotations.shape_to_drawing(corners, "rectangle", PLANE)
    assert drawing == Drawing(AnnotationKind.RECTANGLE, [[1.0, 2.0], [6.0, 9.0]], {})


@pytest.mark.parametrize("shape_type", ["rectangle", "ellipse", "line", "path", "polygon"])
@pytest.mark.parametrize("frame", [PLANE, MOVIE], ids=["plane", "movie"])
def test_what_is_stored_is_drawn_back_as_it_was(shape_type: str, frame: Frame) -> None:
    flat = {
        "rectangle": [[1.0, 2.0], [1.0, 9.0], [6.0, 9.0], [6.0, 2.0]],
        "ellipse": [[1.0, 2.0], [1.0, 9.0], [6.0, 9.0], [6.0, 2.0]],
        "line": [[1.0, 2.0], [6.0, 9.0]],
        "path": [[1.0, 2.0], [6.0, 9.0], [7.0, 3.0]],
        "polygon": [[1.0, 2.0], [1.0, 9.0], [6.0, 9.0]],
    }[shape_type]
    lead = [5.0, 3.0][: len(frame.dims) - 2]
    vertices = np.array([lead + vertex for vertex in flat])

    drawing = annotations.shape_to_drawing(vertices, shape_type, frame)
    drawn, drawn_type = annotations.annotation_to_shape(stored(drawing), frame)

    assert drawn_type == shape_type
    np.testing.assert_allclose(drawn, vertices)


def test_a_rotated_rectangle_is_stored_as_the_polygon_it_traces() -> None:
    """Two corners cannot say a rotation, so the four are kept."""
    diamond = [[0.0, 5.0], [5.0, 10.0], [10.0, 5.0], [5.0, 0.0]]
    drawing = annotations.shape_to_drawing(diamond, "rectangle", PLANE)
    assert drawing.kind == AnnotationKind.POLYGON and drawing.vectors == diamond


def test_a_rotated_ellipse_is_traced_not_boxed() -> None:
    diamond = [[0.0, 5.0], [5.0, 10.0], [10.0, 5.0], [5.0, 0.0]]
    drawing = annotations.shape_to_drawing(diamond, "ellipse", PLANE)
    traced = np.array(drawing.vectors)
    assert drawing.kind == AnnotationKind.POLYGON and len(traced) == annotations.ELLIPSE_VERTICES
    # Every vertex lies on the ellipse inscribed in the rotated box.
    along = (traced - [5.0, 5.0]) @ np.array([[1.0, 1.0], [-1.0, 1.0]]).T / 5.0
    np.testing.assert_allclose((along**2).sum(axis=1), 1.0)


def test_a_circle_is_drawn_as_the_ellipse_of_its_box() -> None:
    circle = SimpleNamespace(kind=AnnotationKind.CIRCLE, vectors=((1.0, 1.0), (5.0, 5.0)), coordinates=())
    drawn, drawn_type = annotations.annotation_to_shape(circle, PLANE)
    assert drawn_type == "ellipse"
    np.testing.assert_allclose(drawn, [[1, 1], [1, 5], [5, 5], [5, 1]])


@pytest.mark.parametrize(
    "annotation",
    [
        SimpleNamespace(kind=AnnotationKind.SPHERE, vectors=((0.0, 0.0, 0.0), (4.0, 4.0, 4.0)), coordinates=()),
        SimpleNamespace(kind=AnnotationKind.POINT, vectors=((1.0, 1.0),), coordinates=()),
        # Drawn in a volume: its vertices are not positions on this plane.
        SimpleNamespace(kind=AnnotationKind.LINE, vectors=((0.0, 1.0, 1.0), (0.0, 4.0, 4.0)), coordinates=()),
    ],
    ids=["sphere", "point", "another-rank"],
)
def test_what_a_shapes_layer_cannot_draw_is_left_out(annotation: SimpleNamespace) -> None:
    assert annotations.annotation_to_shape(annotation, PLANE) is None


def test_an_annotation_that_spans_time_is_drawn_where_the_slider_stands() -> None:
    spanning = SimpleNamespace(kind=AnnotationKind.LINE, vectors=((2.0, 1.0, 1.0), (2.0, 4.0, 4.0)), coordinates=())
    drawn, _ = annotations.annotation_to_shape(spanning, MOVIE, at={"t": 6})
    np.testing.assert_allclose(drawn[:, 0], 6)


def test_a_napari_shape_mikro_has_no_kind_for_is_refused() -> None:
    with pytest.raises(ValueError, match="no annotation kind"):
        annotations.shape_to_drawing([[0.0, 0.0]], "star", PLANE)


def test_the_first_drawing_makes_the_collection_and_the_next_reuse_it() -> None:
    server = Server()
    surface = Surface(dataset="3", system=system("10", "cyx"), name="cells")
    drawing = Drawing(AnnotationKind.LINE, [[1.0, 2.0], [6.0, 9.0]], {})

    annotations.create(server, surface, drawing)
    annotations.create(server, surface, drawing)

    (collection,) = server.called("create_annotation_collection")
    assert collection["name"] == "Drawings on cells"
    assert [axis.name for axis in collection["axes"]] == ["c", "y", "x"]
    (derived,) = collection["derived_from"]
    assert derived.coordinate_system == "10" and derived.transform.kind == "IDENTITY"
    assert [call["collection"] for call in server.called("create_annotation")] == [surface.collection] * 2
    assert server.called("create_annotation")[0]["vectors"] == [[1.0, 2.0], [6.0, 9.0]]


def test_only_collections_on_this_very_grid_are_read() -> None:
    """One drawn on a crop is in the crop's frame: its numbers mean other pixels here."""
    grid, own, cropped = system("10", "yx"), system("30", "yx"), system("40", "yx")
    server = Server(
        graph=graph(
            edge("1", "IdentityTransformation", own, grid),
            edge("2", "TranslationTransformation", cropped, grid, translation=(5.0, 5.0)),
        )
    )
    server.collections = [
        SimpleNamespace(id="2", coordinate_system=cropped),
        SimpleNamespace(id="1", coordinate_system=own),
    ]
    server.annotations = {"1": [SimpleNamespace(id="a")], "2": [SimpleNamespace(id="b")]}
    surface = Surface(dataset="3", system=grid, name="cells")

    found = annotations.fetch(server, surface)

    assert [annotation.id for annotation in found] == ["a"]
    assert surface.collection == "1"


def test_every_page_of_a_collection_is_read() -> None:
    grid = system("10", "yx")
    server = Server()
    server.collections = [SimpleNamespace(id="1", coordinate_system=grid)]
    server.annotations = {"1": [SimpleNamespace(id=str(index)) for index in range(annotations.PAGE + 5)]}
    found = annotations.fetch(server, Surface(dataset="3", system=grid, name="cells"))
    assert len(found) == annotations.PAGE + 5
