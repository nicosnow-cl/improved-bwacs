from typing import List


def get_closest_nodes(
    node, distances_matrix, max_nodes: int = 10
) -> List[int]:
    nodes = distances_matrix[node]
    nodes = sorted(range(len(nodes)), key=lambda k: nodes[k])
    nodes = [node for node in nodes if node != 0]
    nodes = nodes[1 : max_nodes + 1]

    return nodes
