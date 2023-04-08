from abc import ABC, abstractmethod
from typing import List, Tuple
import numpy as np


class ACS(ABC):
    ants_num: int
    nodes: List[int]
    p: float
    evaporation_rate: float
    pheromones_matrix: np.ndarray

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    @abstractmethod
    def create_pheromones_matrix(self, t_delta: float = 0.1) -> np.ndarray:
        """
        Creates the initial matrix of pheromone trail levels.

        Parameters:
            t_delta (float, optional): The initial value of the pheromone
            trail levels.

        Returns:
            A matrix of pheromone trail levels with all values initialized to
            t_delta.
            The matrix has shape (num_nodes, num_nodes), where num_nodes
            is the number of nodes in the problem.
        """

        return np.full((len(self.nodes), len(self.nodes)), t_delta)

    @abstractmethod
    def calculate_t_min_t_max(self,
                              best_solution_quality: float,
                              pheromones_matrix: np.ndarray,
                              probabilities_matrix: np.ndarray) \
            -> Tuple[float, float]:
        """
        Calculates the minimum and maximum values of the pheromone trail
        levels.

        Parameters:
            best_quality (float): The quality of the best solution found so
            far.

            pheromones (ndarray): The matrix of pheromone trail levels.

            probabilities (ndarray): The matrix of transition probabilities.

        Returns:
            A tuple containing the minimum and maximum values of the pheromone
            trail levels.
        """

        t_max = (1 / self.evaporation_rate) * (1 / best_solution_quality)

        max_pheromone = pheromones_matrix.max()
        max_probability = probabilities_matrix.max()
        n_root_probabilitiy = max_probability ** (1 / self.ants_num)

        a = (2 * max_pheromone) / ((self.ants_num - 2) * n_root_probabilitiy)
        b = (2 * max_pheromone) * (1 - n_root_probabilitiy)
        t_min = a / b

        return t_min, t_max

    @abstractmethod
    def get_acs_fitness(solutin_quality: float) -> float:
        """
        Calculates the quality of a solution as the inverse of the sum of its
        costs.

        Parameters:
            solution_costs (List[float]): A list of the costs of each tour in
            the solution.

        Returns:
            The quality of the solution as a float. The higher the value, the
            better the solution.
        """

        return 1 / solutin_quality

    @abstractmethod
    def evaporate_pheromones_matrix(self) -> None:
        """
        Evaporates the pheromone trail levels in the pheromone matrix.

        Parameters:
            None.

        Returns:
            None.
        """

        self.pheromones_matrix *= self.evaporation_rate

    @abstractmethod
    def update_pheromones_matrix(self, solution_arcs, solution_quality):
        """
        Updates the pheromones matrix based on the given solution arcs and
        quality.

        Parameters:
            solution_arcs (List[np.ndarray]): List of 2D numpy arrays
            containing the arcs used in the solution.

            solution_quality (float): The quality of the solution.

        Returns:
            None.
        """

        pheromones_amout = self.p * self.get_acs_fitness(solution_quality)

        for arcs_idxs in solution_arcs:
            self.pheromones_matrix[arcs_idxs[:, 0],
                                   arcs_idxs[:, 1]] += pheromones_amout
