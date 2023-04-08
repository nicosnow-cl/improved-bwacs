from .get_route_load import get_route_load


def check_if_route_load_is_valid(route, demands, max_capacity):
    route_load = get_route_load(route, demands)

    return route_load <= max_capacity
