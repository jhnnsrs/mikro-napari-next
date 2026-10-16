"""From a napari image layer to a mikro dataset."""

import math
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import xarray as xr
from mikro import Mikro, Unit, canonical, create_space, dataset_arrays
from mikro.api.schema import ArrayLens, CoordinateAnchorInput

#: The axes of an image layer by its rank. napari is row-major: the last two dims
#: are y then x.
DIMS_BY_RANK = {
    2: ("y", "x"),
    3: ("z", "y", "x"),
    4: ("t", "z", "y", "x"),
    5: ("t", "c", "z", "y", "x"),
}

#: The edge a pyramid's coarsest level is brought under.
COARSEST_EDGE = 512
MAX_LEVELS = 4

#: What napari calls a dim nobody gave a unit.
UNITLESS = ("pixel", "dimensionless", "")


@dataclass(frozen=True)
class Picture:
    """What of an image layer is uploaded, taken while still in the Qt thread."""

    data: Any
    rgb: bool
    name: str
    scale: tuple[float, ...]
    translate: tuple[float, ...]
    units: tuple[str, ...] | None = None


def dims_of(rank: int, rgb: bool = False) -> tuple[str, ...]:
    """The axis names of an image layer's array.

    Args:
        rank: The number of dims of the layer (a packed colour is not one of them).
        rgb: Whether the array carries a packed colour as its last axis.

    Raises:
        ValueError: For a rank napari shows but no convention names.
    """
    if rank not in DIMS_BY_RANK:
        raise ValueError(f"Cannot name the axes of a {rank}-dimensional layer; give it 2 to 5 dims.")
    return DIMS_BY_RANK[rank] + (("c",) if rgb else ())


def levels_for(shape: Mapping[str, int]) -> int:
    """How many pyramid levels bring the larger of y and x under the coarsest edge."""
    edge = max(shape.get("y", 1), shape.get("x", 1))
    return max(1, min(MAX_LEVELS, int(math.log2(max(edge / COARSEST_EDGE, 1))) + 1))


def physical(picture: Picture, dims: tuple[str, ...]) -> tuple[dict[str, float], dict[str, float], dict[str, str]]:
    """The pixel size, origin and unit per space axis the layer states, if it states any.

    A scale without a unit is a number without a meaning, so it is not a pixel size.
    """
    if picture.units is None:
        return {}, {}, {}
    scale, translate, units = {}, {}, {}
    for dim, factor, shift, unit in zip(dims, picture.scale, picture.translate, picture.units):
        if dim in ("z", "y", "x") and unit not in UNITLESS:
            scale[dim], translate[dim], units[dim] = float(factor), float(shift), unit
    return scale, translate, units


def upload(mikro: Mikro, picture: Picture) -> ArrayLens:
    """Store an image layer as a dataset, with a pyramid and its contrast. Blocking.

    Args:
        mikro: The client to store with.
        picture: The layer's array and what the layer says about it.

    Returns:
        A lens on the whole of the new dataset.
    """
    rank = np.ndim(picture.data) - (1 if picture.rgb else 0)
    dims = dims_of(rank, picture.rgb)
    image = canonical(xr.DataArray(picture.data, dims=dims))
    level_zero, scales = dataset_arrays(image, levels=levels_for(dict(image.sizes)), method="mean")
    dataset = mikro.create_array_dataset(
        data=level_zero,
        scales=scales,
        name=picture.name,
        axes=[str(dim) for dim in image.dims],
        anchors=CoordinateAnchorInput.histogram_anchors(image),
    )

    scale, translate, units = physical(picture, dims[:rank])
    if units:
        space = create_space(mikro, picture.name, {dim: Unit(unit) for dim, unit in units.items()})
        space.register(dataset, scale=scale, **translate)
    return dataset.lens()
