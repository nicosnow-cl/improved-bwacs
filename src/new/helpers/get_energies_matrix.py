import numpy as np


def get_energies_matrix(nodes, depot, tare, distances_matrix, demands_array):
    """
        Create a matrix of energy requirements between the given nodes.

        Args:
            nodes (list of int): A list of node indices.

            depot (int): The index of the depot node.

            tare (float): The tare weight of the vehicle (i.e. the weight of
            the vehicle when empty).

            distances_matrix (numpy.ndarray): A matrix of distances, where
            each element represents the distance between the nodes
            corresponding to the row and column indices.

            demands_array (numpy.ndarray): An array of demand values for each
            node.

        Returns:
            numpy.ndarray: A matrix of energy requirements, where each element
            represents the energy required to travel between the nodes
            corresponding to the row and column indices.

        Raises:
            ValueError: If the length of nodes is not compatible with the
            shape of distances_matrix, or if the length of demands_array
            is not equal to the length of nodes.
    """

    if len(nodes) != distances_matrix.shape[0]:
        raise ValueError("Length of nodes ({}) is not compatible with shape"
                         .format(len(nodes)) +
                         " of distances_matrix ({})."
                         .format(distances_matrix.shape))

    if len(demands_array) != len(nodes):
        raise ValueError("Length of demands_array ({}) is not equal to length"
                         .format(len(demands_array)) +
                         " of nodes ({})."
                         .format(len(nodes)))

    demands = demands_array[nodes]
    distances = distances_matrix[nodes]
    energies = distances * (demands + tare)
    energies[depot, :] = distances[depot, :] * tare
    np.fill_diagonal(energies, 0.0)

    return energies
