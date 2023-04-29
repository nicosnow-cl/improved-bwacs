from typing import List
from typing_extensions import TypedDict

from ..ants.ant_solution import AntSolution


class ACOSolution(TypedDict):
    best_solutions: List[AntSolution]
    global_best_solution: AntSolution
    iterations_mean_costs: List[float]
    iterations_median_costs: List[float]
    iterations_std_cots: List[float]
    iterations_times: List[float]
    total_time: float
