from typing import List
from typing_extensions import TypedDict
import numpy as np

from ..ants.ant_solution import AntSolution


class ACOSolution(TypedDict):
    best_solutions: List[AntSolution]
    global_best_solution: AntSolution
    iterations_best_solutions: List[AntSolution]
    iterations_mean_costs: List[float]
    iterations_median_costs: List[float]
    iterations_std_cots: List[float]
    iterations_times: List[float]
    pheromones_matrices: List[np.ndarray]
    total_time: float
