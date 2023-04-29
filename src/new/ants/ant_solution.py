from typing import List
from typing_extensions import TypedDict
import numpy as np


class AntSolution(TypedDict):
    cost: float
    routes_arcs: List[np.ndarray]
    routes_costs: List[float]
    routes_loads: List[float]
    routes: List[int]
