import numpy as np


def get_distances_matrix(nodes, coords_matrix, metric='euclidean'):
    """
        Create a matrix of distances between the given nodes.

        Parameters:
            nodes (list of int): A list of node indices.

            coords_matrix (numpy.ndarray): A matrix of coordinates, where each
            row represents a node and the first column contains the
            x-coordinate and the second column contains the y-coordinate.

            metric (str, optional): The distance metric to use. Valid options
            are 'euclidean' (the default) and 'manhattan'.

        Returns:
            numpy.ndarray: A matrix of distances, where each element
            represents the distance between the nodes corresponding to the row
            and column indices.

        Raises:
            ValueError: If the metric parameter is not 'euclidean' or
            'manhattan', or if the length of nodes is not compatible with the
            shape of coords_matrix.
    """

    if metric not in ('euclidean', 'manhattan'):
        raise ValueError(
            "Invalid metric: {}. Valid options are 'euclidean' and"
            .format(metric) + " 'manhattan'.")

    if len(nodes) != coords_matrix.shape[0]:
        raise ValueError("Length of nodes ({}) is not compatible with"
                         .format(len(nodes)) + " shape of coords_matrix ({})."
                         .format(coords_matrix.shape))

    l_norm = 1 if metric == 'manhattan' else 2
    diffs = coords_matrix[nodes, np.newaxis] - coords_matrix[np.newaxis, nodes]
    distances_matrix = np.linalg.norm(diffs, ord=l_norm, axis=-1)
    np.fill_diagonal(distances_matrix, 0.0)

    return distances_matrix
