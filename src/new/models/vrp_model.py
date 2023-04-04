import numpy as np


class VRPModel:
    @staticmethod
    def ant_get_updated_values_after_new_move(
            actual_node,
            new_node,
            actual_route_cost,
            actual_vehicle_load,
            distances_matrix,
            demands,
            unvisited_nodes=[]):
        new_route_cost = actual_route_cost + \
            distances_matrix[actual_node][new_node]
        new_vehicle_load = actual_vehicle_load + demands[new_node]
        remaining_unvisited_nodes = []

        if len(unvisited_nodes) > 0:
            remaining_unvisited_nodes = unvisited_nodes[:]
            remaining_unvisited_nodes.remove(new_node)

        return (new_route_cost, new_vehicle_load, remaining_unvisited_nodes)

    @staticmethod
    def fitness_function_legacy(route_arcs, distances_matrix):
        route_cost = 0

        for arc in route_arcs:
            route_cost += distances_matrix[arc[0]][arc[1]]

        return route_cost

    @staticmethod
    def fitness_function(route_arcs, distances_matrix):
        route_arcs_np = np.array(route_arcs)
        route_arcs_2d = np.column_stack(
            (route_arcs_np[:-1], route_arcs_np[1:]))
        route_cost = distances_matrix[route_arcs_2d[:, 0],
                                      route_arcs_2d[:, 1]].sum()

        return route_cost

    @staticmethod
    def fitness(solution, distances_matrix):
        return [__class__.fitness_function(route, distances_matrix)
                for route in solution]
