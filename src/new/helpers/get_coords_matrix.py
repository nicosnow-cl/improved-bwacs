from typing import List
import numpy as np


def get_coords_matrix(nodes: List[int], loc_x, loc_y):
    """
        Return a matrix of coordinates for the given nodes.

        Parameters:
        nodes (list): A list of node indices.

        loc_x (list): A list of x-coordinates corresponding to the nodes.
        loc_y (list): A list of y-coordinates corresponding to the nodes.

        Returns:
        coords_matrix (numpy.ndarray): A matrix of coordinates, where each
        row represents a node and the first column contains the x-coordinate
        and the second column contains the y-coordinate.
    """

    if not (len(nodes) == len(loc_x) == len(loc_y)):
        raise ValueError("nodes, loc_x, and loc_y must have the same length")

    return np.column_stack((loc_x, loc_y))[nodes]
