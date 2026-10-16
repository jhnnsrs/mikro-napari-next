"""Stand-ins for what the server answers, so nothing here needs one."""

from types import SimpleNamespace
from typing import Any

import numpy as np
import xarray as xr
from mikro.api.schema import AxisType

TYPES = {"t": AxisType.TIME, "c": AxisType.CHANNEL, "z": AxisType.SPACE, "y": AxisType.SPACE, "x": AxisType.SPACE}


def system(id: str, names: str, unit: str | None = None) -> SimpleNamespace:
    """A coordinate system over single-letter axes; `unit` is carried by its space axes."""
    return SimpleNamespace(
        id=id,
        name=f"system {id}",
        axes=[
            SimpleNamespace(
                name=name,
                order=order,
                type=TYPES[name],
                long_name=None,
                unit=unit if TYPES[name] == AxisType.SPACE else None,
            )
            for order, name in enumerate(names)
        ],
    )


def edge(id: str, kind: str, source: Any, target: Any, **fields: Any) -> SimpleNamespace:
    return SimpleNamespace(id=id, typename=kind, input=source, output=target, **fields)


def registration(id: str, source: Any, target: Any, axes: str, scale: list[float], translation: list[float]) -> SimpleNamespace:
    """A BY_DIMENSION edge as `space.register` authors it: a scale, then a shift."""
    return edge(
        id,
        "ByDimensionTransformation",
        source,
        target,
        input_axes=tuple(axes),
        output_axes=tuple(axes),
        by_dimension_children=(
            SimpleNamespace(typename="ScaleTransformation", input_axes=tuple(axes), scale=tuple(scale)),
            SimpleNamespace(typename="TranslationTransformation", input_axes=tuple(axes), translation=tuple(translation)),
        ),
    )


def graph(*edges: Any) -> SimpleNamespace:
    return SimpleNamespace(transformations=tuple(edges))


def lens(names: str, shape: tuple[int, ...], id: str = "7", dataset: str = "3", grid: Any = None) -> SimpleNamespace:
    """A lens on the whole of a dataset, over single-letter axes."""
    data = xr.DataArray(np.arange(np.prod(shape), dtype="uint16").reshape(shape), dims=list(names))
    return SimpleNamespace(
        typename="ArrayLens",
        id=id,
        dataset=SimpleNamespace(id=dataset, data_arrays=()),
        coordinate_system=grid or system("10", names),
        axis_table=tuple((name, TYPES[name].value, extent) for name, extent in zip(names, shape)),
        slices=(),
        data=data,
    )


class Server:
    """What a `Mikro` answers, from a few lists, recording what it was asked to store."""

    def __init__(self, name: str = "cells", anchors: list[dict] | None = None, graph: Any = None) -> None:
        self.name = name
        self.anchors = anchors or []
        self.graph = graph or SimpleNamespace(transformations=())
        self.scene_layers: list[Any] = []
        self.collections: list[Any] = []
        self.annotations: dict[str, list[Any]] = {}
        self.calls: list[tuple[str, dict]] = []
        self.rath = SimpleNamespace(query=self._query)

    def _query(self, document: str, variables: dict) -> SimpleNamespace:
        return SimpleNamespace(
            data={"arrayDataset": {"id": variables["id"], "name": self.name, "anchors": self.anchors}}
        )

    def layers(self, filters: Any = None) -> tuple:
        return tuple(self.scene_layers)

    def get_coordinate_graph(self, coordinate_system: Any) -> Any:
        return self.graph

    def get_annotation_collections(self, filters: Any = None) -> tuple:
        return tuple(self.collections)

    def get_annotations(self, filters: Any = None, pagination: Any = None) -> tuple:
        found = self.annotations.get(str(filters.collection), [])
        return tuple(found[pagination.offset : pagination.offset + pagination.limit])

    def create_annotation_collection(self, **arguments: Any) -> SimpleNamespace:
        """Keeps the collection, with a system of its own tied to what it derives from."""
        self.calls.append(("create_annotation_collection", arguments))
        own = SimpleNamespace(id=f"s{len(self.calls)}", axes=[])
        collection = SimpleNamespace(id=f"c{len(self.calls)}", coordinate_system=own)
        self.collections.append(collection)
        for derived in arguments.get("derived_from") or []:
            self.graph = SimpleNamespace(
                transformations=tuple(self.graph.transformations)
                + (
                    edge(
                        f"d{len(self.calls)}",
                        "IdentityTransformation",
                        own,
                        SimpleNamespace(id=derived.coordinate_system),
                    ),
                )
            )
        return collection

    def create_annotation(self, **arguments: Any) -> SimpleNamespace:
        self.calls.append(("create_annotation", arguments))
        annotation = SimpleNamespace(
            id=f"a{len(self.calls)}",
            kind=arguments["kind"],
            vectors=tuple(tuple(vector) for vector in arguments["vectors"]),
            coordinates=tuple(arguments.get("coordinates") or ()),
            collection=arguments.get("collection"),
        )
        self.annotations.setdefault(str(arguments.get("collection")), []).append(annotation)
        return annotation

    def update_annotation(self, **arguments: Any) -> SimpleNamespace:
        self.calls.append(("update_annotation", arguments))
        return SimpleNamespace(**arguments)

    def delete_annotation(self, id: str) -> str:
        self.calls.append(("delete_annotation", {"id": id}))
        return id

    def called(self, name: str) -> list[dict]:
        return [arguments for called, arguments in self.calls if called == name]


class InlineJobs:
    """Runs a job where it is submitted: the tests have no worker threads."""

    def submit(self, function: Any, *args: Any, done: Any = None, failed: Any = None) -> None:
        try:
            result = function(*args)
        except Exception as error:
            if failed is None:
                raise
            failed(error)
            return
        if done is not None:
            done(result)
