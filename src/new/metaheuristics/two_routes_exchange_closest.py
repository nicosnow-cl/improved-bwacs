from random import randint
from typing import List, Tuple
import numpy as np

from ..helpers import get_closest_nodes_sorted, check_if_route_load_is_valid


def two_routes_exchange_closest(
    solution: List[List[int]],
    demands: List[float],
    distances_matrix: np.ndarray,
    max_capacity: float = None,
) -> Tuple[List[List[int]], bool, Tuple[int, int]]:
    if len(solution) < 2:
        return solution, False, None

    route_1_idx = randint(0, len(solution) - 1)
    route_1 = solution[route_1_idx][:]

    if len(route_1) < 4:
        return solution, False, None

    route_1_node_1_idx = randint(1, len(route_1) - 3)
    route_1_node_2_idx = route_1_node_1_idx + 1
    route_1_node_1 = route_1[route_1_node_1_idx]
    route_1_node_2 = route_1[route_1_node_2_idx]

    other_nodes = [
        node
        for idx, route in enumerate(solution)
        if idx != route_1_idx
        for node in route
        if node != route_1_node_1 and node != 0
    ]

    closest_nodes = get_closest_nodes_sorted(
        route_1_node_1, other_nodes, distances_matrix, max_nodes=30
    )

    if not closest_nodes:
        return solution, False, None

    route_2_node_1 = closest_nodes[randint(0, len(closest_nodes) - 1)]

    route_2 = None
    route_2_idx = None
    route_2_node_1_idx = None
    for route_idx, route in enumerate(solution):
        if route_2_node_1 in route:
            route_2 = route[:]
            route_2_idx = route_idx
            route_2_node_1_idx = route_2.index(route_2_node_1)
            break

    if len(route_2) < 4:
        return solution, False, None

    route_2_node_1_idx = (
        route_2_node_1_idx
        if route_2_node_1_idx < len(route_2) - 2
        else route_2_node_1_idx - 1
    )
    route_2_node_2_idx = route_2_node_1_idx + 1
    route_2_node_1 = route_2[route_2_node_1_idx]
    route_2_node_2 = route_2[route_2_node_2_idx]

    route_1[route_1_node_1_idx], route_2[route_2_node_1_idx] = (
        route_2_node_1,
        route_1_node_1,
    )
    route_1[route_1_node_2_idx], route_2[route_2_node_2_idx] = (
        route_2_node_2,
        route_1_node_2,
    )

    if max_capacity and (
        not check_if_route_load_is_valid(route_1, demands, max_capacity)
        or not check_if_route_load_is_valid(route_2, demands, max_capacity)
    ):
        return solution, False, None

    solution[route_1_idx] = route_1
    solution[route_2_idx] = route_2

    return solution, True, (route_1_idx, route_2_idx)
