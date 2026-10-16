"""From a napari image layer to a mikro dataset."""

from types import SimpleNamespace

import numpy as np
import pytest

from mikro_napari.models import upload
from mikro_napari.models.upload import Picture


def picture(shape: tuple[int, ...], rgb: bool = False, **said: object) -> Picture:
    rank = len(shape) - (1 if rgb else 0)
    return Picture(
        data=np.zeros(shape, dtype="uint8"),
        rgb=rgb,
        name="snap",
        scale=said.get("scale", (1.0,) * rank),
        translate=said.get("translate", (0.0,) * rank),
        units=said.get("units"),
    )


def test_the_last_two_dims_are_y_then_x() -> None:
    """napari is row-major; the old plugin named them x, y and stored every image transposed."""
    assert upload.dims_of(2) == ("y", "x")
    assert upload.dims_of(3) == ("z", "y", "x")
    assert upload.dims_of(4) == ("t", "z", "y", "x")
    assert upload.dims_of(5) == ("t", "c", "z", "y", "x")
    assert upload.dims_of(2, rgb=True) == ("y", "x", "c")


def test_a_rank_no_convention_names_is_refused() -> None:
    with pytest.raises(ValueError, match="6-dimensional"):
        upload.dims_of(6)


@pytest.mark.parametrize("edge, levels", [(100, 1), (512, 1), (1024, 2), (4096, 4), (100_000, 4)])
def test_a_pyramid_is_as_deep_as_the_image_is_large(edge: int, levels: int) -> None:
    assert upload.levels_for({"y": edge, "x": 64}) == levels


def test_a_scale_without_a_unit_is_not_a_pixel_size() -> None:
    assert upload.physical(picture((8, 8), scale=(0.5, 0.5)), ("y", "x")) == ({}, {}, {})
    assert upload.physical(picture((8, 8), scale=(0.5, 0.5), units=("pixel", "pixel")), ("y", "x")) == ({}, {}, {})


def test_a_layer_with_units_states_its_pixel_size_along_space_only() -> None:
    said = picture(
        (3, 8, 8), scale=(2.0, 0.5, 0.5), translate=(0.0, 10.0, 20.0), units=("second", "micrometer", "micrometer")
    )
    assert upload.physical(said, ("t", "y", "x")) == (
        {"y": 0.5, "x": 0.5},
        {"y": 10.0, "x": 20.0},
        {"y": "micrometer", "x": "micrometer"},
    )


class Store:
    """Records the dataset it is asked to create."""

    def __init__(self) -> None:
        self.created: dict = {}

    def create_array_dataset(self, **arguments: object) -> SimpleNamespace:
        self.created = arguments
        return SimpleNamespace(lens=lambda: "the lens")


def test_a_packed_colour_is_stored_channel_first() -> None:
    """The dataset is canonical: the colour axis napari keeps last comes before space."""
    store = Store()
    assert upload.upload(store, picture((16, 12, 3), rgb=True)) == "the lens"
    assert store.created["name"] == "snap"
    assert store.created["axes"] == ["c", "y", "x"]
    assert store.created["data"].shape == (3, 16, 12)
    assert len(store.created["anchors"]) == 3


def test_a_volume_is_stored_as_it_lies_in_the_layer() -> None:
    store = Store()
    upload.upload(store, picture((4, 16, 12)))
    assert store.created["axes"] == ["z", "y", "x"]
    assert store.created["data"].shape == (4, 16, 12)
