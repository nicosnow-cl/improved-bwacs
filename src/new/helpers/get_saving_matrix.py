from sklearn.preprocessing import MinMaxScaler
import numpy as np

# Basic


def get_saving_matrix(depot, nodes, matrix_distances):
    depot_idx = nodes.index(depot)
    shape = matrix_distances.shape
    saving_matrix = np.zeros(shape)

    for i in nodes[depot_idx + 1:]:
        for j in nodes[depot_idx + 1:]:
            if i != j:
                s_i0 = matrix_distances[i][depot]
                s_0j = matrix_distances[depot][j]
                s_ij = matrix_distances[i][j]
                saving = (s_i0 + s_0j) - s_ij
                saving_matrix[i][j] = saving

    min_not_zero = saving_matrix[saving_matrix != 0].min()

    # Here we normalice the values between min distance and max distance.
    scaler = MinMaxScaler(feature_range=(min_not_zero, matrix_distances.max()))
    scaler.fit(saving_matrix)
    saving_matrix = scaler.transform(saving_matrix)

    return saving_matrix

# (Benito Quintanilla, 2015)


def get_saving_matrix_2015(depot,
                           nodes,
                           demands,
                           matrix_distances,
                           lamb=2,
                           mu=1,
                           nu=1):
    depot_idx = nodes.index(depot)
    shape = matrix_distances.shape
    saving_matrix = np.zeros(shape)

    for i in nodes[depot_idx + 1:]:
        for j in nodes[depot_idx + 1:]:
            if i != j:
                d_0i = matrix_distances[depot][i]
                d_0j = matrix_distances[depot][j]
                d_i0 = matrix_distances[i][depot]
                d_ij = matrix_distances[i][j]
                d_j0 = matrix_distances[j][depot]
                mean_demands = np.mean(demands)
                utilization = (demands[i] + demands[j]) / mean_demands
                saving = d_i0 + d_0j - (lamb * d_ij) + \
                    (mu * abs(d_0i - d_j0)) - (nu * utilization)
                saving_matrix[i][j] = saving

    min_not_zero = saving_matrix[saving_matrix != 0].min()

    # Here we normalice the values between min distance and max distance.
    scaler = MinMaxScaler(feature_range=(min_not_zero, matrix_distances.max()))
    scaler.fit(saving_matrix)
    saving_matrix = scaler.transform(saving_matrix)

    return saving_matrix
