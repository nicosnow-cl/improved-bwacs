from typing import List, Tuple
from typing_extensions import TypedDict


class AntSolution(TypedDict):
    cost: float
    routes_arcs: List[List[Tuple[int, int]]]
    routes_arcs_flatten: List[Tuple[int, int]]
    routes_costs: List[float]
    routes_loads: List[float]
    routes: List[int]
