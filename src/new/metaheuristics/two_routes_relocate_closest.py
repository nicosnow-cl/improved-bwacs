from random import randint
from typing import List, Tuple
import numpy as np

from ..helpers import get_closest_nodes_sorted, check_if_route_load_is_valid


def two_routes_relocate_closest(
    solution: List[List[int]],
    demands: List[float],
    distances_matrix: np.ndarray,
    max_capacity: float = None,
) -> Tuple[List[List[int]], bool, Tuple[int, int]]:
    if len(solution) < 2:
        return solution, False, None

    route_idx_1 = randint(0, len(solution) - 1)
    route_1 = solution[route_idx_1][:]

    if len(route_1) < 3:
        return solution, False, None

    node_idx_1 = randint(1, len(route_1) - 2)
    node_1 = route_1[node_idx_1]

    other_nodes = [
        node
        for idx, route in enumerate(solution)
        if idx != route_idx_1
        for node in route
        if node != node_1 and node != 0
    ]

    closest_nodes = get_closest_nodes_sorted(
        node_1, other_nodes, distances_matrix, max_nodes=20
    )

    if not closest_nodes:
        return solution, False, None

    route_1.pop(node_idx_1)
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

    route_2.insert(node_idx_2, node_1)

    if max_capacity and (
        not check_if_route_load_is_valid(route_1, demands, max_capacity)
        or not check_if_route_load_is_valid(route_2, demands, max_capacity)
    ):
        return solution, False, None

    solution[route_idx_1] = route_1
    solution[route_idx_2] = route_2

    return solution, True, (route_idx_1, route_idx_2)
