from random import randint
from typing import List, Tuple
import numpy as np

from ..helpers import check_if_route_load_is_valid


def two_routes_swap(
    solution: List[List[int]],
    demands: List[float],
    max_capacity: float = None,
    distances_matrix: np.ndarray = None,
) -> Tuple[List[List[int]], bool, Tuple[int, int]]:
    if len(solution) < 2:
        return solution, False, None

    route_idx_1 = randint(0, len(solution) - 1)
    route_idx_2 = randint(0, len(solution) - 1)

    while route_idx_1 == route_idx_2:
        route_idx_2 = randint(0, len(solution) - 1)

    route_1 = solution[route_idx_1][:]
    route_2 = solution[route_idx_2][:]

    if len(route_1) < 3 or len(route_2) < 3:
        return solution, False, None

    node_idx_1 = randint(1, len(route_1) - 2)
    node_idx_2 = randint(1, len(route_2) - 2)
    node_1 = route_1[node_idx_1]
    node_2 = route_2[node_idx_2]

    route_1[node_idx_1] = node_2
    route_2[node_idx_2] = node_1

    if max_capacity and (
        not check_if_route_load_is_valid(route_1, demands, max_capacity)
        or not check_if_route_load_is_valid(route_2, demands, max_capacity)
    ):
        return solution, False, None

    solution[route_idx_1] = route_1
    solution[route_idx_2] = route_2

    return solution, True, (route_idx_1, route_idx_2)
