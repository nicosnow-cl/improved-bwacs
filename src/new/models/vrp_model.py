from typing import List, Tuple
import numpy as np


class VRPModel:
    @staticmethod
    def get_cost_between_two_nodes(node1: int,
                                   node2: int,
                                   distances_matrix: np.ndarray) -> float:
        """
            Returns the cost of the edge between one node and a seconde node.

            Parameters:
                node1 (int): The initial node.

                node2 (int): The ending node.

                distances_matrix (np.ndarray): A 2D NumPy array of distances
                between various points.

            Returns:
                float: cost of the edge between one node and a seconde node.
        """

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
        """
            Calculates the fitness value of a given route.

            Parameters:
                route_arcs (np.ndarray): A 1D NumPy array of route arcs,
                where each arc represents a connection between two points.

                distances_matrix (np.ndarray): A 2D NumPy array of distances
                between various points.

            Returns:
                float: The fitness value of the given route, calculated as the
                sum of the distances between the points in the route.
        """

        route_arcs_2d = np.column_stack((route_arcs[:-1], route_arcs[1:]))

        return distances_matrix[route_arcs_2d[:, 0], route_arcs_2d[:, 1]].sum()

    @staticmethod
    def fitness(solution: List[List[int]],
                distances_matrix: np.ndarray) -> List[float]:
        """
            Calculates the fitness value for each route in the given solution.

            Parameters:
                solution (List[List[int]]): A list of route arcs, where each
                arc represents a connection between two points.

                distances_matrix (np.ndarray): A 2D NumPy array of distances
                between various points.

            Returns:
                List[float]: A list of fitness values, where the fitness value
                of each route arc in the solution is calculated using the
                fitness_function() method of the class that contains this
                static method.
        """

        return [__class__.fitness_function(route_arcs, distances_matrix)
                for route_arcs in solution]
