from random import randint
from typing import List


def single_route_relocate(route: List[int]) -> List[int]:
    if len(route) < 4:
        return route

    new_route = route[:]
    node = randint(1, len(new_route) - 2)
    new_position = randint(1, len(new_route) - 2)

    new_route.insert(new_position, new_route.pop(node))

    return new_route
