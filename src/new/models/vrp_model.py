import numpy as np


@staticmethod
class VRPModel:
    @staticmethod
    def ant_get_updated_values_after_new_move(self,
                                              actual_node,
                                              new_node,
                                              actual_route_cost,
                                              actual_vehicle_load,
                                              distances_matrix,
                                              demands,
                                              unvisited_nodes=None):
        new_route_cost = actual_route_cost + \
            distances_matrix[actual_node][new_node]
        new_vehicle_load = actual_vehicle_load + demands[new_node]

        remaining_unvisited_nodes = []
        if unvisited_nodes is not None:
            remaining_unvisited_nodes = unvisited_nodes[
                unvisited_nodes != new_node]

        return (new_route_cost, new_vehicle_load, remaining_unvisited_nodes)

    @staticmethod
    def fitness_function_legacy(self, route_arcs, distances_matrix):
        route_cost = 0

        for arc in route_arcs:
            route_cost += distances_matrix[arc[0]][arc[1]]

        return route_cost

    @staticmethod
    def fitness_function(self, route_arcs, distances_matrix):
        route_arcs_np = np.array(route_arcs)
        route_cost = distances_matrix[route_arcs_np[:, 0],
                                      route_arcs_np[:, 1]].sum()

        return route_cost

    @staticmethod
    def fitness(self, solution):
        return [self.fitness_function(route) for route in solution]
