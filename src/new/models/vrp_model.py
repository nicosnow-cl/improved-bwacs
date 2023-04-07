from typing import List, Tuple
import numpy as np


class VRPModel:
    @staticmethod
    def get_cost_between_two_nodes(actual_node: int, new_node: int,
                                   distances_matrix: np.ndarray) -> float:
        """
            Returns the cost of the edge between the actual node and the new
            node.

            Parameters:
                actual_node (int): The current node of the ant.

                new_node (int): The node that the ant is moving to.

                distances_matrix (np.ndarray): A 2D NumPy array of distances
                between various points.

            Returns:
                float: The cost of the edge between the actual node and the new
        """

        return distances_matrix[actual_node][new_node]

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
    def fitness_function(route_arcs_np: np.ndarray,
                         distances_matrix: np.ndarray) -> float:
        """
            Calculates the fitness value of a given route.

            Parameters:
                route_arcs_np (np.ndarray): A 1D NumPy array of route arcs,
                where each arc represents a connection between two points.

                distances_matrix (np.ndarray): A 2D NumPy array of distances
                between various points.

            Returns:
                float: The fitness value of the given route, calculated as the
                sum of the distances between the points in the route.
        """

        route_arcs_2d = np.column_stack(
            (route_arcs_np[:-1], route_arcs_np[1:]))
        route_cost = distances_matrix[route_arcs_2d[:, 0],
                                      route_arcs_2d[:, 1]].sum()

        return route_cost

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

    @staticmethod
    def ant_get_updated_values_after_new_move(
            actual_node: int,
            new_node: int,
            actual_route_cost: float,
            actual_vehicle_load: float,
            distances_matrix: np.ndarray,
            demands: np.ndarray,
            unvisited_nodes: List[int] = []) -> Tuple[float, float, List[int]]:
        """
            Updates the route cost, vehicle load, and unvisited nodes list for
            an ant after it makes a new move.

            Parameters:
                actual_node (int): The current node of the ant.

                new_node (int): The node that the ant is moving to.

                actual_route_cost (float): The current cost of the ant's route.

                actual_vehicle_load (float): The current load of the ant's
                vehicle.

                distances_matrix (np.ndarray): A 2D NumPy array of distances
                between various points.

                demands (np.ndarray): A 1D NumPy array of demands for each
                node.

                unvisited_nodes (List[int], optional): A list of unvisited
                nodes. Defaults to an empty list.

            Returns:
                Tuple[float, float, List[int]]: A tuple containing the updated
                route cost, vehicle load, and list of remaining unvisited
                nodes.
        """

        new_route_cost = actual_route_cost + \
            __class__.get_cost_between_two_nodes(
                actual_node, new_node, distances_matrix)
        new_vehicle_load = actual_vehicle_load + demands[new_node]
        remaining_unvisited_nodes = []

        if new_node in unvisited_nodes:
            remaining_unvisited_nodes = unvisited_nodes[:]
            remaining_unvisited_nodes.remove(new_node)

        return (new_route_cost, new_vehicle_load, remaining_unvisited_nodes)
