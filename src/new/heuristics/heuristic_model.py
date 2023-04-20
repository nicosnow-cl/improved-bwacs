from typing import List
import numpy as np

from ..helpers import get_distances_matrix, get_saving_matrix, \
    get_saving_matrix_2015, get_inversed_matrix


class HeuristicModel:
    demands: List[int] = None
    importance_distances: float = 2.0
    importance_savings: float = 1.0
    matrix_coords: np.ndarray = None
    matrix_heuristics: np.ndarray = None
    metric: str = 'euclidean'
    nodes: List[int] = []

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def get_heuristic_matrix(self, heuristics=['distance']) -> np.ndarray:
        for heuristic in set(heuristics):
            if heuristic == 'distance':
                matrix_distances = get_distances_matrix(
                    self.nodes, self.matrix_coords, self.metric)
                norm_matrix_distances = get_inversed_matrix(
                    matrix_distances)
                parametrized_matrix = np.power(norm_matrix_distances,
                                               self.importance_distances)

                if self.matrix_heuristics is None:
                    self.matrix_heuristics = parametrized_matrix
                else:
                    self.matrix_heuristics = np.multiply(
                        self.matrix_heuristics, parametrized_matrix)

            elif heuristic == 'saving':
                matrix_distances = get_distances_matrix(
                    self.nodes, self.matrix_coords, self.metric)
                matrix_savings = get_saving_matrix_2015(self.nodes[0],
                                                        self.nodes,
                                                        self.demands,
                                                        matrix_distances,
                                                        2, 1, 1)
                # matrix_savings = get_saving_matrix(self.nodes[0],
                #                                    self.nodes,
                #                                    matrix_distances)
                parametrized_matrix = np.power(matrix_savings,
                                               self.importance_savings)

                if self.matrix_heuristics is None:
                    self.matrix_heuristics = parametrized_matrix
                else:
                    self.matrix_heuristics = np.multiply(
                        self.matrix_heuristics, parametrized_matrix)

        return self.matrix_heuristics
