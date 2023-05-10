from sklearn.preprocessing import MinMaxScaler
import numpy as np

from ..helpers import get_inversed_matrix

# Basic


def get_saving_matrix(depot, nodes, matrix_distances):
    depot_idx = nodes.index(depot)
    shape = matrix_distances.shape
    saving_matrix = np.zeros(shape)

    for i in nodes[(depot_idx + 1) :]:
        for j in nodes[(depot_idx + 1) :]:
            if i != j:
                s_i0 = matrix_distances[i][depot]
                s_0j = matrix_distances[depot][j]
                s_ij = matrix_distances[i][j]
                saving = (s_i0 + s_0j) - s_ij
                saving_matrix[i][j] = saving

    inv_matrix_distances = get_inversed_matrix(matrix_distances)
    min_not_zero_value = inv_matrix_distances[inv_matrix_distances != 0].min()
    max_value = inv_matrix_distances[inv_matrix_distances != np.inf].max()

    # Here we normalice the values between min distance and max distance.
    scaler = MinMaxScaler(feature_range=(min_not_zero_value, max_value))
    saving_matrix = scaler.fit_transform(saving_matrix)

    saving_matrix[0, :] = 1
    saving_matrix[:, [0]] = 1
    np.fill_diagonal(saving_matrix, 1)

    return saving_matrix


# (Benito Quintanilla, 2015)


def get_saving_matrix_2015(
    depot, nodes, demands, matrix_distances, lamb=2, mu=1, nu=1
):
    depot_idx = nodes.index(depot)
    shape = matrix_distances.shape
    saving_matrix = np.zeros(shape)

    for i in nodes[depot_idx + 1 :]:
        for j in nodes[depot_idx + 1 :]:
            if i != j:
                d_0i = matrix_distances[depot][i]
                d_0j = matrix_distances[depot][j]
                d_i0 = matrix_distances[i][depot]
                d_ij = matrix_distances[i][j]
                d_j0 = matrix_distances[j][depot]
                mean_demands = np.mean(demands)
                utilization = (demands[i] + demands[j]) / mean_demands
                saving = (
                    d_i0
                    + d_0j
                    - (lamb * d_ij)
                    + (mu * abs(d_0i - d_j0))
                    - (nu * utilization)
                )
                saving_matrix[i][j] = saving

    inv_matrix_distances = get_inversed_matrix(matrix_distances)
    min_not_zero_value = inv_matrix_distances[inv_matrix_distances != 0].min()
    max_value = inv_matrix_distances[inv_matrix_distances != np.inf].max()

    # Here we normalice the values between min distance and max distance.
    scaler = MinMaxScaler(feature_range=(min_not_zero_value, max_value))
    saving_matrix = scaler.fit_transform(saving_matrix)

    saving_matrix[0, :] = 1
    saving_matrix[:, [0]] = 1
    np.fill_diagonal(saving_matrix, 1)

    return saving_matrix
