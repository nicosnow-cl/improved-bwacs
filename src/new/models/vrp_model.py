from typing import List, Tuple
import numpy as np

from .problem_model import ProblemModel


class VRPModel(ProblemModel):
    @staticmethod
    def get_cost_between_two_nodes(node1: int,
                                   node2: int,
                                   matrix_distances: np.ndarray) -> float:
        return matrix_distances[node1][node2]

    @staticmethod
    def fitness_function_legacy(route_arcs: List[Tuple[int, int]],
                                matrix_distances: np.ndarray) -> float:
        """
            Calculates the fitness value of a given route.

            Parameters:
                route_arcs (List[Tuple[int, int]]): A list of route arcs,
                where each arc represents a connection between two points.

                distances_matrix (np.ndarray): A matrix of distances
                between various points.

            Returns:
                float: The fitness value of the given route, calculated as the
                sum of the distances between the points in the route.
        """

        route_cost = 0

        for arc in route_arcs:
            route_cost += matrix_distances[arc[0]][arc[1]]

        return route_cost

    @staticmethod
    def fitness_function(route_arcs: np.ndarray,
                         matrix_distances: np.ndarray) -> float:
        route_arcs_2d = np.column_stack((route_arcs[:-1], route_arcs[1:]))

        return matrix_distances[route_arcs_2d[:, 0], route_arcs_2d[:, 1]].sum()

    @staticmethod
    def fitness(solution: List[List[int]],
                matrix_distances: np.ndarray) -> List[float]:
        return [__class__.fitness_function(route_arcs, matrix_distances)
                for route_arcs in solution]

    @staticmethod
    def validate_instance(nodes, demands, max_capacity) -> dict:
        errors = {}

        if 0 not in nodes:
            errors['depot_exists'] = 'Depot node not found.'

        if len(nodes) != len(set(nodes)):
            errors['duplicated_nodes'] = 'Duplicate nodes are not allowed.'

        if len(nodes) != len(demands):
            errors['all_demands'] = 'Demands must be provided for all nodes.'

        if not all(demand >= 0 for demand in demands):
            errors['only_positive_demands'] = \
                'Negative demands are not allowed.'

        if max_capacity <= 0:
            errors['negative_max_capacity'] = \
                'Max capacity must be greater than 0.'

        return errors
