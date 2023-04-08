from typing import List, Tuple
import numpy as np

from .problem_model import ProblemModel


class VRPModel(ProblemModel):
    @staticmethod
    def get_cost_between_two_nodes(node1: int,
                                   node2: int,
                                   distances_matrix: np.ndarray) -> float:
        return distances_matrix[node1][node2]

    @staticmethod
    def fitness_function_legacy(route_arcs: List[Tuple[int, int]],
                                distances_matrix: np.ndarray) -> float:
        """
            Calculates the fitness value of a given route.

            Parameters:
                route_arcs (List[Tuple[int, int]]): A list of route arcs,
                where each arc represents a connection between two points.

                distances_matrix (List[List[float]]): A matrix of distances
                between various points.

            Returns:
                float: The fitness value of the given route, calculated as the
                sum of the distances between the points in the route.
        """

        route_cost = 0

        for arc in route_arcs:
            route_cost += distances_matrix[arc[0]][arc[1]]

        return route_cost

    @staticmethod
    def fitness_function(route_arcs: np.ndarray,
                         distances_matrix: np.ndarray) -> float:
        route_arcs_2d = np.column_stack((route_arcs[:-1], route_arcs[1:]))

        return distances_matrix[route_arcs_2d[:, 0], route_arcs_2d[:, 1]].sum()

    @staticmethod
    def fitness(solution: List[List[int]],
                distances_matrix: np.ndarray) -> List[float]:
        return [__class__.fitness_function(route_arcs, distances_matrix)
                for route_arcs in solution]
