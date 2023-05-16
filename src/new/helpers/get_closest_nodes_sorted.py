from typing import List
import numpy as np


def get_closest_nodes_sorted(
    base_node: int,
    nodes: List[int],
    distances_matrix: np.ndarray,
    max_nodes: int = None,
) -> List[int]:
    distances = distances_matrix[base_node]
    selected_nodes = sorted(nodes, key=lambda x: distances[x])

    if max_nodes:
        selected_nodes = selected_nodes[:max_nodes]

    return selected_nodes
