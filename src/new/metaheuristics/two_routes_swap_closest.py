from random import randint
from typing import List
import numpy as np

from ..helpers import check_if_route_load_is_valid, get_closest_nodes


def two_routes_swap_closest(
    solution: List[List[int]],
    demands: List[float],
    distances_matrix: np.ndarray,
    max_capacity: float = None,
) -> List[List[int]]:
    if len(solution) < 2:
        return solution

    route_idx_1 = randint(0, len(solution) - 1)
    route_1 = solution[route_idx_1][:]

    if len(route_1) < 3:
        return solution

    node_idx_1 = randint(1, len(route_1) - 2)
    node_1 = route_1[node_idx_1]

    closest_nodes = get_closest_nodes(node_1, distances_matrix, max_nodes=15)
    closest_nodes = [n for n in closest_nodes if n not in route_1]

    if not closest_nodes:
        return solution

    node_2 = closest_nodes[randint(0, len(closest_nodes) - 1)]

    route_2 = None
    node_idx_2 = None
    route_idx_2 = None
    for route_idx, route in enumerate(solution):
        if node_2 in route:
            route_2 = route[:]
            route_idx_2 = route_idx
            node_idx_2 = route_2.index(node_2)
            break

    route_1[node_idx_1] = node_2
    route_2[node_idx_2] = node_1

    if max_capacity and (
        not check_if_route_load_is_valid(route_1, demands, max_capacity)
        or not check_if_route_load_is_valid(route_2, demands, max_capacity)
    ):
        return solution

    solution[route_idx_1] = route_1
    solution[route_idx_2] = route_2

    return solution
