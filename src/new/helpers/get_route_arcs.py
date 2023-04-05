import numpy as np


def get_route_arcs_legacy(route):
    route_arcs = ()
    prev_node = None

    for pos, i in enumerate(route):
        if pos == 0:
            prev_node = i
            continue
        else:
            route_arcs += ((prev_node, i),)

    return route_arcs


def get_route_arcs(route):
    if len(route) < 2:
        raise ValueError("Route must have at least two nodes")

    return [(route[i], route[i+1]) for i in range(len(route)-1)]
