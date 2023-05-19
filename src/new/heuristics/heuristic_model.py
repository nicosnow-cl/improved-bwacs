from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from typing import List
import numpy as np

from ..helpers import (
    get_distances_matrix,
    get_saving_matrix,
    get_saving_matrix_2015,
    get_inversed_matrix,
    get_capacity_matrix,
)


class HeuristicModel:
    demands: List[int] = None
    importance_distances: float = 2.0
    importance_savings: float = 1.0
    matrix_coords: np.ndarray = None
    matrix_heuristics: np.ndarray = None
    max_capacity: int = 0
    metric: str = "euclidean"
    nodes: List[int] = []
    importance_capacity: float = 1.0
    normalization: str = None

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def normalize_matrix(self, matrix: np.ndarray, normalization: str):
        if normalization == "standard":
            matrix = StandardScaler().fit_transform(matrix)
        elif normalization == "minmax":
            matrix = MinMaxScaler().fit_transform(matrix)
        elif normalization == "robust":
            matrix = RobustScaler().fit_transform(matrix)

        np.fill_diagonal(matrix, 1.0)

        return matrix

    def get_heuristic_matrix(self, heuristics=["distance"]) -> np.ndarray:
        for heuristic in set(heuristics):
            if heuristic == "distance":
                matrix_distances = get_distances_matrix(
                    self.nodes, self.matrix_coords, self.metric
                )
                norm_matrix_distances = get_inversed_matrix(matrix_distances)

                parametrized_matrix = np.power(
                    norm_matrix_distances, self.importance_distances
                )

                if self.matrix_heuristics is None:
                    self.matrix_heuristics = parametrized_matrix
                else:
                    self.matrix_heuristics = np.multiply(
                        self.matrix_heuristics, parametrized_matrix
                    )

            elif heuristic == "saving":
                matrix_distances = get_distances_matrix(
                    self.nodes, self.matrix_coords, self.metric
                )

                matrix_savings = get_saving_matrix(
                    self.nodes[0], self.nodes, matrix_distances
                )

                parametrized_matrix = np.power(
                    matrix_savings, self.importance_savings
                )

                if self.matrix_heuristics is None:
                    self.matrix_heuristics = parametrized_matrix
                else:
                    self.matrix_heuristics = np.multiply(
                        self.matrix_heuristics, parametrized_matrix
                    )
            elif heuristic == "capacity":
                capacity_matrix = get_capacity_matrix(
                    self.nodes, self.demands, self.max_capacity
                )
                parametrized_matrix = np.power(
                    capacity_matrix, self.importance_capacity
                )

                if self.matrix_heuristics is None:
                    self.matrix_heuristics = parametrized_matrix
                else:
                    self.matrix_heuristics = np.multiply(
                        self.matrix_heuristics, parametrized_matrix
                    )

        if self.normalization is not None:
            self.matrix_heuristicss = self.normalize_matrix(
                self.matrix_heuristics, self.normalization
            )

        return self.matrix_heuristics
