from pydantic import BaseModel, Field
from mikro_next.api.schema import Image
import uuid


class BaseContainer(BaseModel):
    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    representation: Image
    with_rois: bool = False
    with_labels: bool = False
