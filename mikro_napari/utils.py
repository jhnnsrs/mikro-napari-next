from dataclasses import dataclass
from mikro_next.api.schema import ROI, RoiKind


@dataclass
class NapariROI:
    """A napari ROI."""

    type: str
    data: list
    color: str
    id: str


def convert_roi_to_napari_roi(roi: ROI) -> NapariROI:
    """Convert a ROI to a napari ROI."""

    if roi.kind in [
        RoiKind.RECTANGLE,
        RoiKind.POLYGON,
        RoiKind.LINE,
        RoiKind.PATH,
    ]:
        return NapariROI(
            **{
                "type": roi.kind.lower(),
                "data": roi.get_vector_data("xy"),
                "color": "white",
                "id": roi.id,
            }
        )

    return None
