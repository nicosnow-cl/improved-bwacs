def get_route_arcs(route):
    route_arcs = ()
    prev_node = None

    for pos, i in enumerate(route):
        if pos == 0:
            prev_node = i
            continue
        else:
            route_arcs += ((prev_node, i),)

    return route_arcs
