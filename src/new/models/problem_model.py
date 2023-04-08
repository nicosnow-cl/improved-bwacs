from abc import ABC, abstractmethod
from typing import List
import numpy as np

from .vehicle_model import VehicleModel


class ProblemModel(ABC):
    @abstractmethod
    def get_cost_between_two_nodes(
            node1: int,
            node2: int,
            distances_matrix: np.ndarray,
            vehicle: VehicleModel = None) -> float:
        """
            Returns the cost of the edge between one node and a seconde node.

            Parameters:
                node1 (int): The initial node.

                node2 (int): The ending node.

                distances_matrix (np.ndarray): A 2D NumPy array of distances
                between various points.

                vehicle (VehicleModel, optional): The vehicle that is used to
                calculate the cost. Defaults to None.

            Returns:
                float: cost of the edge between one node and a seconde node.
        """

        raise NotImplementedError(
            'get_cost_between_two_nodes() need to be implemented')

    @abstractmethod
    def fitness_function(route_arcs: np.ndarray,
                         distances_matrix: np.ndarray,
                         vehicle: VehicleModel = None) -> float:
        """
            Calculates the fitness value of a given route.

            Parameters:
                route_arcs (np.ndarray): A 1D NumPy array of route arcs,
                where each arc represents a connection between two points.

                distances_matrix (np.ndarray): A 2D NumPy array of distances
                between various points.

                vehicle (VehicleModel, optional): The vehicle that is used to
                calculate the cost. Defaults to None.

            Returns:
                float: The fitness value of the given route, calculated as the
                sum of the distances between the points in the route.
        """

        raise NotImplementedError(
            'fitness_function() need to be implemented')

    @abstractmethod
    def fitness(solution: List[List[int]],
                distances_matrix: np.ndarray,
                vehicle: VehicleModel = None) -> List[float]:
        """
            Calculates the fitness value for each route in the given solution.

            Parameters:
                solution (List[List[int]]): A list of route arcs, where each
                arc represents a connection between two points.

                distances_matrix (np.ndarray): A 2D NumPy array of distances
                between various points.

                vehicle (VehicleModel, optional): The vehicle that is used to
                calculate the cost. Defaults to None.

            Returns:
                List[float]: A list of fitness values, where the fitness value
                of each route arc in the solution is calculated using the
                fitness_function() method of the class that contains this
                static method.
        """

        raise NotImplementedError('fitness() need to be implemented')
