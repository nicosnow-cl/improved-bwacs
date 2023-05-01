import numpy as np


def get_capacity_matrix(nodes, demands, max_capacity):
    capacity_matrix = np.zeros((len(nodes), len(nodes)))

    for i in nodes:
        for j in nodes:
            if i != j:
                capacity_matrix[i][j] = (
                    demands[i] + demands[j]) / max_capacity

    capacity_matrix[0, :] = 1
    capacity_matrix[:, [0]] = 1
    np.fill_diagonal(capacity_matrix, 1)

    return capacity_matrix
