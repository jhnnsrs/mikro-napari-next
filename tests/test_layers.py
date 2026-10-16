"""From what the server says about an image to the napari layers that show it."""

from pathlib import Path
from types import SimpleNamespace

import dask.array as da
import pytest
import xarray as xr
from mikro.api.schema import ColorMap

from mikro_napari.models import layers
from mikro_napari.models.layers import ChannelDisplay

from .stand_ins import Server, edge, graph, lens, registration, system


def pin(channel: int) -> SimpleNamespace:
    """A single-channel crop: the lens an RGB layer names a component by."""
    return SimpleNamespace(slices=(SimpleNamespace(axis="c", start=channel, stop=channel + 1, step=1),))


def test_an_image_without_a_channel_axis_is_one_layer() -> None:
    specs = layers.plan_specs("cells", lens("zyx", (4, 8, 8)).axis_table)
    assert [(spec.name, spec.dims, spec.channel_index) for spec in specs] == [("cells", ("z", "y", "x"), None)]


def test_every_channel_is_a_layer_of_its_own() -> None:
    displays = [ChannelDisplay(label="DAPI", limits=(1.0, 9.0)), ChannelDisplay()]
    specs = layers.plan_specs("cells", lens("czyx", (2, 4, 8, 8)).axis_table, displays)
    assert [spec.name for spec in specs] == ["cells · DAPI", "cells · 1"]
    assert [spec.dims for spec in specs] == [("z", "y", "x")] * 2
    assert [spec.channel_index for spec in specs] == [0, 1]
    assert specs[0].contrast_limits == (1.0, 9.0) and specs[1].contrast_limits is None
    assert specs[0].color != specs[1].color


def test_three_channels_alone_are_not_a_packed_colour() -> None:
    """Only a stored RGB layer makes one: nothing is guessed from a channel count."""
    axes = lens("cyx", (3, 8, 8)).axis_table
    assert len(layers.plan_specs("slide", axes)) == 3
    (packed,) = layers.plan_specs("slide", axes, rgb=(0, 1, 2), rgb_limits=(0.0, 255.0))
    assert packed.rgb == (0, 1, 2) and packed.contrast_limits == (0.0, 255.0)


def test_the_channel_axis_is_found_by_type_not_by_name() -> None:
    axes = (("wavelength", "CHANNEL", 2), ("y", "SPACE", 8), ("x", "SPACE", 8))
    assert layers.channel_axis_of(axes) == "wavelength"
    assert [spec.dims for spec in layers.plan_specs("cells", axes)] == [("y", "x")] * 2


def test_anchors_give_each_channel_its_label_and_limits() -> None:
    anchors = [
        {"coordinates": {"c": 1}, "valueHistogram": {"p1": 2, "p99": 40}, "channelLabel": {"label": "GFP"}},
        {"coordinates": {"c": 0}, "valueHistogram": {"p1": None, "p99": None, "min": 0, "max": 7}},
    ]
    assert layers.displays_from_anchors(anchors, "c", 2) == [
        ChannelDisplay(limits=(0.0, 7.0)),
        ChannelDisplay(label="GFP", limits=(2.0, 40.0)),
    ]


def test_an_anchor_that_pins_no_channel_describes_them_all() -> None:
    anchors = [{"coordinates": {}, "valueHistogram": {"p1": 3, "p99": 30}}]
    assert layers.displays_from_anchors(anchors, None, 1) == [ChannelDisplay(limits=(3.0, 30.0))]


def test_a_flat_histogram_gives_no_limits() -> None:
    anchors = [{"coordinates": {"c": 0}, "valueHistogram": {"p1": 5, "p99": 5, "min": 5, "max": 5}}]
    assert layers.displays_from_anchors(anchors, "c", 1) == [ChannelDisplay()]


def test_the_scene_layers_of_the_lens_overrule_the_anchors() -> None:
    """A layer draws its lens and nothing narrower: it styles the channel the lens holds."""
    stained = [SimpleNamespace(typename="IntensityLayer", intensity_colormap=ColorMap.VIRIDIS, color=None, clim_min=10, clim_max=20)]
    displays, rgb, _ = layers.displays_from_layers(stained, [ChannelDisplay(label="GFP", limits=(2.0, 40.0))])
    assert rgb is None
    assert displays == [ChannelDisplay(label="GFP", limits=(10.0, 20.0), colormap="viridis")]

    tinted = [SimpleNamespace(typename="IntensityLayer", intensity_colormap=ColorMap.INTENSITY, color=(255, 0, 0, 255), clim_min=None, clim_max=None)]
    displays, _, _ = layers.displays_from_layers(tinted, [ChannelDisplay(label="DAPI", limits=(1.0, 9.0))])
    assert displays == [ChannelDisplay(label="DAPI", limits=(1.0, 9.0), color=(255, 0, 0, 255))]


def test_an_rgb_layer_of_the_lens_names_the_packed_channels() -> None:
    stored = [SimpleNamespace(typename="RgbLayer", red_lens=pin(2), green_lens=pin(1), blue_lens=pin(0), clim_min=0, clim_max=255)]
    _, rgb, limits = layers.displays_from_layers(stored, [ChannelDisplay()] * 3, lens("cyx", (3, 8, 6)), "c")
    assert rgb == (2, 1, 0) and limits == (0.0, 255.0)


def test_a_component_is_counted_from_where_the_shown_lens_starts() -> None:
    """Slices are in the dataset's indices; the shown lens may itself be a crop of the channels."""
    shown = SimpleNamespace(slices=(SimpleNamespace(axis="c", start=2, stop=5, step=1),))
    assert layers.channel_of(pin(3), shown, "c") == 1
    # A lens that leaves the channel axis whole says nothing of it: channel 0.
    assert layers.channel_of(SimpleNamespace(slices=()), SimpleNamespace(slices=()), "c") == 0


def test_a_registration_gives_the_pixel_size_and_origin() -> None:
    """The edge `space.register` authors names the shared axes: the channel is not one."""
    grid, stage = system("10", "czyx"), system("20", "zyx", unit="micrometer")
    found = layers.placement_from_graph(
        graph(registration("1", grid, stage, "zyx", [2.0, 0.5, 0.5], [0.0, 100.0, 40.0])), "10"
    )
    assert found.scale == {"z": 2.0, "y": 0.5, "x": 0.5}
    assert found.translate == {"z": 0.0, "y": 100.0, "x": 40.0}
    assert found.units == {"z": "micrometer", "y": "micrometer", "x": "micrometer"}


def test_steps_compose_until_a_space_has_units() -> None:
    grid, lensed, stage = system("10", "yx"), system("11", "yx"), system("20", "yx", unit="micrometer")
    found = layers.placement_from_graph(
        graph(
            edge("1", "TranslationTransformation", grid, lensed, translation=(10.0, 20.0)),
            edge("2", "ScaleTransformation", lensed, stage, scale=(0.5, 0.25)),
        ),
        "10",
    )
    assert found.scale == {"y": 0.5, "x": 0.25}
    assert found.translate == {"y": 5.0, "x": 5.0}


def test_an_axis_aligned_affine_is_read_and_a_rotation_is_not() -> None:
    grid, stage = system("10", "yx"), system("20", "yx", unit="micrometer")
    aligned = edge("1", "AffineTransformation", grid, stage, affine=((0.5, 0, 3.0), (0, 0.5, 4.0), (0, 0, 1)))
    assert layers.placement_from_graph(graph(aligned), "10").scale == {"y": 0.5, "x": 0.5}
    rotated = edge("1", "AffineTransformation", grid, stage, affine=((0, -1, 0), (1, 0, 0), (0, 0, 1)))
    assert layers.placement_from_graph(graph(rotated), "10") == layers.Placement()


def test_of_rival_registrations_the_first_stored_wins() -> None:
    grid = system("10", "yx")
    first, second = system("20", "yx", unit="micrometer"), system("21", "yx", unit="micrometer")
    rivals = graph(
        edge("12", "ScaleTransformation", grid, second, scale=(9.0, 9.0)),
        edge("5", "ScaleTransformation", grid, first, scale=(0.5, 0.5)),
    )
    assert layers.placement_from_graph(rivals, "10").scale == {"y": 0.5, "x": 0.5}


def test_a_grid_nobody_registered_has_no_placement() -> None:
    assert layers.placement_from_graph(graph(), "10") == layers.Placement()


def test_only_identities_make_two_systems_the_same_grid() -> None:
    grid, drawn, cropped = system("10", "yx"), system("30", "yx"), system("40", "yx")
    linked = layers.identity_linked(
        graph(
            edge("1", "IdentityTransformation", drawn, grid),
            edge("2", "TranslationTransformation", cropped, grid, translation=(5.0, 5.0)),
        ),
        "10",
    )
    assert linked == {"10", "30"}


def test_prepare_reads_name_contrast_and_placement(qapp) -> None:
    grid, stage = system("10", "cyx"), system("20", "yx", unit="micrometer")
    server = Server(
        name="cells",
        anchors=[{"coordinates": {"c": 0}, "valueHistogram": {"p1": 1, "p99": 9}, "channelLabel": {"label": "DAPI"}}],
        graph=graph(registration("1", grid, stage, "yx", [0.5, 0.5], [0.0, 0.0])),
    )
    plan = layers.prepare(server, lens("cyx", (2, 8, 8), grid=grid))
    assert (plan.name, plan.dataset, plan.lens, plan.coordinate_system) == ("cells", "3", "7", "10")
    assert [spec.name for spec in plan.specs] == ["cells · DAPI", "cells · 1"]

    arguments = layers.layer_arguments(plan, plan.specs[0])
    assert arguments["scale"] == (0.5, 0.5) and arguments["blending"] == "additive"
    assert arguments["contrast_limits"] == (1.0, 9.0)
    assert arguments["metadata"] == {
        "mikro": True,
        "identifier": "@mikro/arraydataset",
        "dataset": "3",
        "lens": "7",
        "coordinate_system": "10",
        "channel_axis": "c",
        "channel_index": 0,
        "space_axes": ["y", "x"],
    }
    assert layers.layer_arguments(plan, plan.specs[0], physical=False)["scale"] == (1.0, 1.0)
    assert layers.layer_data(plan, plan.specs[1]).shape == (8, 8)


def test_a_packed_colour_is_read_from_the_last_axis() -> None:
    server = Server()
    server.scene_layers = [SimpleNamespace(typename="RgbLayer", red_lens=pin(0), green_lens=pin(1), blue_lens=pin(2), clim_min=None, clim_max=None)]
    plan = layers.prepare(server, lens("cyx", (3, 8, 6)))
    (spec,) = plan.specs
    assert layers.layer_data(plan, spec).shape == (8, 6, 3)
    assert layers.layer_arguments(plan, spec)["rgb"] is True


def test_unregistered_images_are_tiled_and_registered_ones_stay() -> None:
    stage = system("20", "yx", unit="micrometer")
    placed_grid = system("11", "yx")
    placed = layers.prepare(
        Server(graph=graph(registration("1", placed_grid, stage, "yx", [1.0, 1.0], [0.0, 0.0]))),
        lens("yx", (10, 10), grid=placed_grid),
    )
    loose = [layers.prepare(Server(), lens("yx", (10, 20))) for _ in range(3)]
    offsets = layers.grid_offsets([loose[0], placed, loose[1], loose[2]])
    assert offsets[1] == {}
    assert [offsets[index] for index in (0, 2, 3)] == [
        {"y": 0.0, "x": 0.0},
        {"y": 0.0, "x": pytest.approx(22.0)},
        {"y": pytest.approx(11.0), "x": 0.0},
    ]


def test_the_context_document_is_valid_against_the_schema() -> None:
    """The one query no generated method covers, held to the schema the client ships."""
    graphql = pytest.importorskip("graphql")
    import mikro.api

    sdl = Path(mikro.api.__file__).parent / "schema.graphql"
    # The SDL uses federation directives it does not declare; they are not what is tested.
    schema = graphql.build_schema(sdl.read_text(), assume_valid=True)
    assert graphql.validate(schema, graphql.parse(layers.CONTEXT_DOCUMENT)) == []


def pyramid(dims: str, shape: tuple[int, ...], levels: int, coarsen: str = "zyx") -> list[xr.DataArray]:
    """A pyramid of lazy arrays, halving along `coarsen` at every level."""
    return [
        xr.DataArray(
            da.zeros(
                tuple(max(1, extent // 2**level) if dim in coarsen else extent for dim, extent in zip(dims, shape)),
                chunks=-1,
            ),
            dims=list(dims),
        )
        for level in range(levels)
    ]


def test_a_volume_is_rendered_from_the_finest_level_a_texture_holds() -> None:
    """napari draws 3D from the coarsest level it is given: a thumbnail is no volume."""
    axes = (("c", "CHANNEL", 2), ("z", "SPACE", 512), ("y", "SPACE", 4096), ("x", "SPACE", 4096))
    levels = pyramid("czyx", (2, 512, 4096, 4096), 6)

    kept = layers.renderable_levels(levels, axes)

    # 512x4096x4096 and 256x2048x2048 are too many voxels; 128x1024x1024 fits.
    assert [level.shape[1:] for level in kept] == [(512, 4096, 4096), (256, 2048, 2048), (128, 1024, 1024)]


def test_a_volume_that_fits_whole_is_shown_at_full_resolution() -> None:
    axes = (("z", "SPACE", 64), ("y", "SPACE", 512), ("x", "SPACE", 512))
    kept = layers.renderable_levels(pyramid("zyx", (64, 512, 512), 4), axes)
    assert [level.shape for level in kept] == [(64, 512, 512)]


def test_a_long_edge_is_too_much_for_a_texture_whatever_the_voxel_count() -> None:
    axes = (("z", "SPACE", 4), ("y", "SPACE", 8192), ("x", "SPACE", 512))
    kept = layers.renderable_levels(pyramid("zyx", (4, 8192, 512), 4, coarsen="yx"), axes)
    assert kept[-1].shape == (4, 2048, 128)


def test_a_pyramid_no_level_of_which_fits_is_kept_whole() -> None:
    axes = (("z", "SPACE", 4096), ("y", "SPACE", 4096), ("x", "SPACE", 4096))
    levels = pyramid("zyx", (4096, 4096, 4096), 2)
    assert len(layers.renderable_levels(levels, axes)) == 2


def test_a_pyramid_of_planes_keeps_its_thumbnail() -> None:
    """Nothing renders a slide in 3D, and browsing it wants every level."""
    axes = (("c", "CHANNEL", 3), ("y", "SPACE", 100_000), ("x", "SPACE", 100_000))
    levels = pyramid("cyx", (3, 100_000, 100_000), 9, coarsen="yx")
    assert len(layers.renderable_levels(levels, axes)) == 9
