import numpy as np


def get_normalized_matrix(matrix: np.ndarray) -> np.ndarray:
    mask = (matrix != 0) & np.isfinite(matrix)

    with np.errstate(divide='ignore'):  # ignore division by zero warnings
        return np.divide(1, matrix, out=np.zeros_like(matrix), where=mask)
