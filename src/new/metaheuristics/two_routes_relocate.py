from random import randint
from typing import List
import numpy as np

from ..helpers import check_if_route_load_is_valid


def two_routes_relocate(
    solution: List[List[int]],
    demands: List[float],
    max_capacity: float = None,
    distances_matrix: np.ndarray = None,
) -> List[List[int]]:
    if len(solution) < 2:
        return solution

    route_idx_1 = randint(0, len(solution) - 1)
    route_idx_2 = randint(0, len(solution) - 1)

    while route_idx_1 == route_idx_2:
        route_idx_2 = randint(0, len(solution) - 1)

    route_1 = solution[route_idx_1][:]
    route_2 = solution[route_idx_2][:]

    if len(route_1) < 3 or len(route_2) < 3:
        return solution

    pop_index = randint(1, len(route_1) - 2)
    node = route_1.pop(pop_index)

    insert_index = randint(1, len(route_2) - 2)
    route_2.insert(insert_index, node)

    if max_capacity and (
        not check_if_route_load_is_valid(route_2, demands, max_capacity)
    ):
        return solution

    solution[route_idx_1] = route_1
    solution[route_idx_2] = route_2

    return solution
