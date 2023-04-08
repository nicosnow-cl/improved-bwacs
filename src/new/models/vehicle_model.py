from typing import Optional
from typing_extensions import TypedDict


class VehicleModel(TypedDict):
    load: float
    max_capacity: float
    tare: Optional[float]
