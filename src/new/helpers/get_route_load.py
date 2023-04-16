def get_route_load(route, demands):
    load = 0
    for node in route:
        load += demands[node]
    return load
