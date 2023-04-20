import numpy as np


def get_inversed_matrix(matrix: np.ndarray) -> np.ndarray:
    mask = (matrix != 0) & np.isfinite(matrix)

    with np.errstate(divide='ignore'):  # ignore division by zero warnings
        matrix_inversed = np.divide(1, matrix,  where=mask)
        np.fill_diagonal(matrix_inversed, 1)

    return matrix_inversed
