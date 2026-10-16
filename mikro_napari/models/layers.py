"""From a mikro lens or dataset to the napari layers that show it.

Everything up to `prepare` is plain data in, plain data out: no Qt, no network.
`prepare` is the one blocking call; it asks the server and opens the arrays, so it
belongs on a worker thread.
"""

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import xarray as xr
from mikro import Mikro
from mikro.api.schema import AxisType, ColorMap, LayerFilter
from mikro.inputs.render import fallback_color

IDENTIFIER = "@mikro/arraydataset"

#: The colormaps of a stored intensity layer that napari knows by name. Every
#: other one is drawn as a ramp to the layer's colour.
NAPARI_COLORMAPS = {
    ColorMap.VIRIDIS: "viridis",
    ColorMap.PLASMA: "plasma",
    ColorMap.INFERNO: "inferno",
    ColorMap.MAGMA: "magma",
    ColorMap.RED: "red",
    ColorMap.GREEN: "green",
    ColorMap.BLUE: "blue",
    ColorMap.CYAN: "cyan",
    ColorMap.MAGENTA: "magenta",
    ColorMap.YELLOW: "yellow",
    ColorMap.GREY: "gray",
}

CONTEXT_DOCUMENT = """
query NapariContext($id: ID!) {
  arrayDataset(id: $id) {
    id
    name
    anchors {
      coordinates
      valueHistogram {
        p1
        p99
        min
        max
      }
      channelLabel {
        label
      }
    }
  }
}
"""


@dataclass(frozen=True)
class Placement:
    """Where a pixel grid sits in a physical space, per axis name."""

    scale: Mapping[str, float] = field(default_factory=dict)
    translate: Mapping[str, float] = field(default_factory=dict)
    units: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class ChannelDisplay:
    """What is stored about how one channel is shown."""

    label: str | None = None
    limits: tuple[float, float] | None = None
    color: tuple[int, ...] | None = None
    colormap: str | None = None


@dataclass(frozen=True)
class LayerSpec:
    """One napari image layer: a channel, or the packed colour of three."""

    name: str
    dims: tuple[str, ...]
    channel_axis: str | None
    channel_index: int | None
    rgb: tuple[int, int, int] | None = None
    colormap: str | None = None
    color: tuple[float, float, float] | None = None
    contrast_limits: tuple[float, float] | None = None


@dataclass(eq=False)
class LayerPlan:
    """Everything the viewer needs to show one lens or dataset."""

    name: str
    dataset: str
    lens: str | None
    coordinate_system: str | None
    system: Any
    axes: tuple[tuple[str, str, int], ...]
    specs: list[LayerSpec]
    placement: Placement
    levels: list[xr.DataArray]

    @property
    def dims(self) -> tuple[str, ...]:
        """The dims every layer of this plan shares: the axes without the channel."""
        return self.specs[0].dims

    @property
    def space_axes(self) -> tuple[str, ...]:
        return tuple(name for name, kind, _ in self.axes if kind == AxisType.SPACE.value)


def channel_axis_of(axes: Sequence[tuple[str, str, int]]) -> str | None:
    """The axis whose positions are channels, by its type and never by its name."""
    for name, kind, _ in axes:
        if kind == AxisType.CHANNEL.value:
            return name
    return None


def _limits(histogram: Mapping[str, Any] | None) -> tuple[float, float] | None:
    if not histogram:
        return None
    for low, high in (("p1", "p99"), ("min", "max")):
        a, b = histogram.get(low), histogram.get(high)
        if a is not None and b is not None and a < b:
            return float(a), float(b)
    return None


def displays_from_anchors(
    anchors: Sequence[Mapping[str, Any]], channel_axis: str | None, channels: int
) -> list[ChannelDisplay]:
    """The label and contrast limits each channel's anchor recorded.

    An anchor that does not pin the channel axis describes the whole array, so it
    stands in for every channel that has none of its own.
    """
    shared = ChannelDisplay()
    found: dict[int, ChannelDisplay] = {}
    for anchor in anchors:
        coordinates = anchor.get("coordinates") or {}
        display = ChannelDisplay(
            label=(anchor.get("channelLabel") or {}).get("label"),
            limits=_limits(anchor.get("valueHistogram")),
        )
        index = coordinates.get(channel_axis) if channel_axis else None
        if index is None:
            if display.limits or display.label:
                shared = display
        elif 0 <= int(index) < channels:
            found[int(index)] = display
    return [found.get(index, ChannelDisplay(limits=shared.limits)) for index in range(channels)]


def channel_of(component: Any, source: Any, channel_axis: str | None) -> int:
    """Which channel of ``source`` a single-channel crop of it pins.

    Slices are spelled in the dataset's own indices, and an axis taken whole is
    left out: a lens that says nothing of the channel axis starts at channel 0.
    """

    def start_and_step(lens: Any) -> tuple[int, int]:
        for cut in getattr(lens, "slices", None) or ():
            if cut.axis == channel_axis:
                return int(cut.start or 0), int(cut.step or 1)
        return 0, 1

    pinned, _ = start_and_step(component)
    origin, step = start_and_step(source)
    return (pinned - origin) // step


def displays_from_layers(
    layers: Sequence[Any],
    displays: Sequence[ChannelDisplay],
    source: Any = None,
    channel_axis: str | None = None,
) -> tuple[list[ChannelDisplay], tuple[int, int, int] | None, tuple[float, float] | None]:
    """Lay what the lens' own scene layers say over the anchors.

    A layer draws its lens and nothing narrower: an intensity layer of this lens
    styles every channel the lens holds (one, as scenes are authored), and an RGB
    layer names its three components by the lenses that pin them.

    Returns the displays, and the channels and limits of an RGB layer if the lens
    has one: the only thing that makes three channels one packed colour.
    """
    merged = list(displays)
    rgb = None
    rgb_limits = None
    for layer in layers:
        kind = getattr(layer, "typename", None)
        limits = None
        low, high = getattr(layer, "clim_min", None), getattr(layer, "clim_max", None)
        if low is not None and high is not None and low < high:
            limits = (float(low), float(high))
        if kind == "RgbLayer":
            rgb = (
                channel_of(layer.red_lens, source, channel_axis),
                channel_of(layer.green_lens, source, channel_axis),
                channel_of(layer.blue_lens, source, channel_axis),
            )
            rgb_limits = limits
        elif kind == "IntensityLayer":
            merged = [
                ChannelDisplay(
                    label=known.label,
                    limits=limits or known.limits,
                    color=tuple(layer.color) if layer.color else None,
                    colormap=NAPARI_COLORMAPS.get(layer.intensity_colormap),
                )
                for known in merged
            ]
    return merged, rgb, rgb_limits


def plan_specs(
    name: str,
    axes: Sequence[tuple[str, str, int]],
    displays: Sequence[ChannelDisplay] = (),
    rgb: tuple[int, int, int] | None = None,
    rgb_limits: tuple[float, float] | None = None,
) -> list[LayerSpec]:
    """One layer per channel; a single layer only for a stored packed colour."""
    channel_axis = channel_axis_of(axes)
    dims = tuple(axis for axis, _, _ in axes if axis != channel_axis)
    if channel_axis is None:
        display = displays[0] if displays else ChannelDisplay()
        return [LayerSpec(name, dims, None, None, contrast_limits=display.limits)]

    channels = next(extent for axis, _, extent in axes if axis == channel_axis)
    if rgb is not None and all(0 <= index < channels for index in rgb):
        return [LayerSpec(name, dims, channel_axis, None, rgb=rgb, contrast_limits=rgb_limits)]

    specs = []
    for index in range(channels):
        display = displays[index] if index < len(displays) else ChannelDisplay()
        color = display.color or fallback_color(index, channels)
        specs.append(
            LayerSpec(
                name=f"{name} · {display.label or index}",
                dims=dims,
                channel_axis=channel_axis,
                channel_index=index,
                colormap=display.colormap,
                color=(color[0] / 255, color[1] / 255, color[2] / 255),
                contrast_limits=display.limits,
            )
        )
    return specs


# --------------------------------------------------------------------------- #
# Physical placement
# --------------------------------------------------------------------------- #


def _names(system: Any) -> list[str]:
    return [axis.name for axis in sorted(system.axes, key=lambda axis: axis.order)]


def _diagonal(affine: Sequence[Sequence[float]], rank: int) -> tuple[list[float], list[float]] | None:
    """The scale and offset of an axis-aligned affine; None if it shears or rotates."""
    if len(affine) < rank or any(len(row) < rank for row in affine[:rank]):
        return None
    for i in range(rank):
        for j in range(rank):
            if i != j and abs(affine[i][j]) > 1e-12:
                return None
    scale = [float(affine[i][i]) for i in range(rank)]
    offset = [float(affine[i][rank]) if len(affine[i]) > rank else 0.0 for i in range(rank)]
    return scale, offset


def _step(edge: Any, names: Sequence[str]) -> dict[str, tuple[float, float]] | None:
    """The per-axis ``(factor, offset)`` one transformation applies, by input axis name.

    None for a transformation that is not a per-axis scale and shift.
    """
    kind = getattr(edge, "typename", None)
    if kind == "IdentityTransformation":
        return {}
    if kind == "ScaleTransformation":
        return {name: (float(s), 0.0) for name, s in zip(names, edge.scale)}
    if kind == "TranslationTransformation":
        return {name: (1.0, float(t)) for name, t in zip(names, edge.translation)}
    if kind == "AffineTransformation":
        parts = _diagonal(edge.affine, len(names))
        if parts is None:
            return None
        return {name: (s, t) for name, s, t in zip(names, *parts)}
    if kind == "ByDimensionTransformation":
        step: dict[str, tuple[float, float]] = {}
        for child in edge.by_dimension_children:
            acted = list(getattr(child, "input_axes", None) or edge.input_axes)
            part = _step(child, acted)
            if part is None:
                return None
            for name, (factor, offset) in part.items():
                known_factor, known_offset = step.get(name, (1.0, 0.0))
                step[name] = (known_factor * factor, known_offset * factor + offset)
        return step
    return None


def placement_from_graph(graph: Any, start: str) -> Placement:
    """The pixel size and origin of the grid `start`, in the nearest physical space.

    Walks the registrations out of `start`, composing the ones that scale and shift
    axis by axis, and stops at the first space whose space axes carry a unit. Of
    rival registrations the one stored first wins.
    """
    start = str(start)
    outgoing: dict[str, list[Any]] = {}
    for edge in graph.transformations:
        if edge.input is not None and edge.output is not None:
            outgoing.setdefault(str(edge.input.id), []).append(edge)
    for edges in outgoing.values():
        edges.sort(key=lambda edge: (len(str(edge.id)), str(edge.id)))

    frontier: list[tuple[str, dict[str, tuple[float, float]]]] = [(start, {})]
    seen = {start}
    while frontier:
        system, placed = frontier.pop(0)
        for edge in outgoing.get(system, []):
            target = str(edge.output.id)
            if target in seen:
                continue
            step = _step(edge, _names(edge.input))
            if step is None:
                continue
            reached = set(_names(edge.output))
            # An axis the step does not name passes through; one the target lacks
            # is left behind.
            moved = {}
            for name in set(placed) | set(step):
                if name not in reached:
                    continue
                factor, offset = placed.get(name, (1.0, 0.0))
                by, shift = step.get(name, (1.0, 0.0))
                moved[name] = (factor * by, offset * by + shift)
            units = {
                axis.name: str(axis.unit)
                for axis in edge.output.axes
                if axis.type == AxisType.SPACE and axis.unit is not None
            }
            if units:
                return Placement(
                    scale={name: factor for name, (factor, _) in moved.items() if name in units},
                    translate={name: offset for name, (_, offset) in moved.items() if name in units},
                    units=units,
                )
            seen.add(target)
            frontier.append((target, moved))
    return Placement()


def identity_linked(graph: Any, start: str) -> set[str]:
    """The systems that are the very grid of `start`: reachable over identities alone."""
    linked = {str(start)}
    grew = True
    while grew:
        grew = False
        for edge in graph.transformations:
            if getattr(edge, "typename", None) != "IdentityTransformation":
                continue
            if edge.input is None or edge.output is None:
                continue
            ends = {str(edge.input.id), str(edge.output.id)}
            if ends & linked and not ends <= linked:
                linked |= ends
                grew = True
    return linked


# --------------------------------------------------------------------------- #
# napari arguments
# --------------------------------------------------------------------------- #


def layer_data(plan: LayerPlan, spec: LayerSpec) -> xr.DataArray | list[xr.DataArray]:
    """The array of one layer: its channel of every level, or of the one there is."""

    def select(level: xr.DataArray) -> xr.DataArray:
        if spec.channel_axis is None:
            return level
        if spec.rgb is not None:
            # napari reads a packed colour from the last axis.
            return level.isel({spec.channel_axis: list(spec.rgb)}).transpose(..., spec.channel_axis)
        return level.isel({spec.channel_axis: spec.channel_index})

    selected = [select(level) for level in plan.levels]
    return selected if len(selected) > 1 else selected[0]


def layer_metadata(plan: LayerPlan, spec: LayerSpec) -> dict[str, Any]:
    """What traces a layer back to its dataset: ids, never the objects."""
    return {
        "mikro": True,
        "identifier": IDENTIFIER,
        "dataset": plan.dataset,
        "lens": plan.lens,
        "coordinate_system": plan.coordinate_system,
        "channel_axis": spec.channel_axis,
        "channel_index": spec.channel_index,
        "space_axes": list(plan.space_axes),
    }


def layer_arguments(
    plan: LayerPlan,
    spec: LayerSpec,
    physical: bool = True,
    offset: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """The keyword arguments of ``viewer.add_image`` for one layer, data aside."""
    offset = offset or {}
    placement = plan.placement if physical else Placement()
    arguments: dict[str, Any] = {
        "name": spec.name,
        "metadata": layer_metadata(plan, spec),
        "scale": tuple(float(placement.scale.get(dim, 1.0)) for dim in spec.dims),
        "translate": tuple(
            float(placement.translate.get(dim, 0.0)) + float(offset.get(dim, 0.0)) for dim in spec.dims
        ),
        "multiscale": len(plan.levels) > 1,
    }
    if spec.rgb is not None:
        arguments["rgb"] = True
    else:
        arguments["blending"] = "additive"
    if spec.contrast_limits is not None:
        arguments["contrast_limits"] = spec.contrast_limits
    return arguments


# --------------------------------------------------------------------------- #
# Asking the server
# --------------------------------------------------------------------------- #


def _is_whole(lens: Any) -> bool:
    return all(
        s.start is None and s.stop is None and s.step in (None, 1) for s in getattr(lens, "slices", ())
    )


#: The most voxels of a volume napari is handed as its coarsest level (512 cubed),
#: and the longest edge a 3D texture takes.
VOLUME_VOXELS = 512**3
VOLUME_EDGE = 2048


def renderable_levels(levels: Sequence[xr.DataArray], axes: Sequence[tuple[str, str, int]]) -> list[xr.DataArray]:
    """The levels of a pyramid down to the one a volume is still worth rendering at.

    napari draws a multiscale layer in 3D from its coarsest level alone, and a
    pyramid built for browsing ends in a thumbnail. So a volume's pyramid is cut
    after the first level a 3D texture holds: the finest that fits is the coarsest
    napari gets. A pyramid of planes is left whole: nothing renders it in 3D.
    """
    space = [index for index, (_, kind, _) in enumerate(axes) if kind == AxisType.SPACE.value]
    if len(space) < 3:
        return list(levels)
    for count, level in enumerate(levels, start=1):
        extents = [level.shape[index] for index in space]
        voxels = 1
        for extent in extents:
            voxels *= extent
        if voxels <= VOLUME_VOXELS and max(extents) <= VOLUME_EDGE:
            return list(levels[:count])
    return list(levels)


def pyramid_levels(source: Any, axes: Sequence[tuple[str, str, int]]) -> list[xr.DataArray]:
    """The levels napari can page through, finest first; one level if it cannot.

    A pyramid is only usable whole (its levels know nothing of a lens' crop) and
    only when the levels differ along space: napari indexes every other axis the
    same at each level. Of a volume's pyramid the levels too coarse to render
    are left out (see `renderable_levels`).
    """
    dataset = getattr(source, "dataset", source)
    if not _is_whole(source) or len(getattr(dataset, "data_arrays", ())) < 2:
        return [source.data]
    levels = dataset.multi_scale_data()
    fixed = [index for index, (_, kind, _) in enumerate(axes) if kind != AxisType.SPACE.value]
    if any(level.shape[index] != levels[0].shape[index] for level in levels for index in fixed):
        return [source.data]
    return renderable_levels(levels, axes)


def prepare(mikro: Mikro, source: Any) -> LayerPlan:
    """Read what is needed to show a lens or a dataset. Blocking: asks the server.

    Args:
        mikro: The client to ask.
        source: An ``ArrayLens``, or an ``ArrayDataset`` to look at whole.

    Returns:
        The plan: its layers, their arrays (lazy) and where they sit.
    """
    is_lens = getattr(source, "typename", None) == "ArrayLens"
    dataset_id = str(source.dataset.id if is_lens else source.id)
    system = source.coordinate_system if is_lens else source.intrinsic_system
    axes = tuple(source.axis_table)
    channel_axis = channel_axis_of(axes)
    channels = next((extent for axis, _, extent in axes if axis == channel_axis), 1)

    answer = mikro.rath.query(CONTEXT_DOCUMENT, {"id": dataset_id})
    context = (answer.data or {}).get("arrayDataset") or {}
    name = context.get("name") or getattr(source, "name", None) or f"Dataset {dataset_id}"

    displays = displays_from_anchors(context.get("anchors") or [], channel_axis, channels)
    rgb = rgb_limits = None
    if is_lens:
        layers = mikro.layers(filters=LayerFilter(lens=source.id))
        displays, rgb, rgb_limits = displays_from_layers(layers, displays, source, channel_axis)

    graph = mikro.get_coordinate_graph(system.id) if system is not None else None
    return LayerPlan(
        name=name,
        dataset=dataset_id,
        lens=str(source.id) if is_lens else None,
        coordinate_system=str(system.id) if system is not None else None,
        system=system,
        axes=axes,
        specs=plan_specs(name, axes, displays, rgb, rgb_limits),
        placement=placement_from_graph(graph, system.id) if graph is not None else Placement(),
        levels=pyramid_levels(source, axes),
    )


def grid_offsets(plans: Sequence[LayerPlan], gutter: float = 0.1) -> list[dict[str, float]]:
    """Where to put each of several images so they lie next to each other.

    An image registered in a physical space already has its place and keeps it.
    The others fill a square grid in reading order across their last two space
    axes, a gutter apart.
    """
    loose = [plan for plan in plans if not plan.placement.units and len(plan.space_axes) >= 2]
    if not loose:
        return [{} for _ in plans]
    columns = max(1, int(len(loose) ** 0.5 + 0.999))
    extents = [{name: extent for name, _, extent in plan.axes} for plan in loose]
    pitch = {
        "row": max(extent[plan.space_axes[-2]] for plan, extent in zip(loose, extents)) * (1 + gutter),
        "column": max(extent[plan.space_axes[-1]] for plan, extent in zip(loose, extents)) * (1 + gutter),
    }
    offsets: list[dict[str, float]] = []
    placed = 0
    for plan in plans:
        if plan not in loose:
            offsets.append({})
            continue
        row, column = divmod(placed, columns)
        offsets.append(
            {plan.space_axes[-2]: row * pitch["row"], plan.space_axes[-1]: column * pitch["column"]}
        )
        placed += 1
    return offsets
