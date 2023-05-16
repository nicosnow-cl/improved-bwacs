from random import randint, random
from typing import List


def single_route_swap(route: List[int]) -> List[int]:
    if len(route) < 4:
        return route

    new_route = route[:]

    if random() < 0.3:
        node_idx_1 = randint(1, len(new_route) - 2)
        node_idx_2 = randint(1, len(new_route) - 2)

        while node_idx_1 == node_idx_2:
            node_idx_2 = randint(1, len(new_route) - 2)
    else:
        node_idx_1 = randint(1, len(new_route) - 3)
        node_idx_2 = node_idx_1 + 1

    new_route[node_idx_1], new_route[node_idx_2] = (
        new_route[node_idx_2],
        new_route[node_idx_1],
    )

    return new_route
