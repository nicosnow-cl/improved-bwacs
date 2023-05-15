from random import randint
from typing import List


def single_route_swap(route: List[int]) -> List[int]:
    if len(route) < 4:
        return route

    new_route = route[:]
    node_idx_1 = randint(1, len(new_route) - 2)
    node_idx_2 = randint(1, len(new_route) - 2)

    new_route[node_idx_1], new_route[node_idx_2] = (
        new_route[node_idx_2],
        new_route[node_idx_1],
    )

    return new_route
