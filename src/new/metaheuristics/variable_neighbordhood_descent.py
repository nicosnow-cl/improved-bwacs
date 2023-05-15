from copy import deepcopy
from math import ceil
from typing import List
import numpy as np
import random
import time

from .single_route_relocate import single_route_relocate
from .single_route_swap import single_route_swap
from .two_routes_swap import two_routes_swap
from .two_routes_relocate import two_routes_relocate
from .two_routes_exchange import two_routes_exchange
from .two_routes_swap_closest import two_routes_swap_closest
from .variable_neighbordhood_search import GeneralVNS
from ..acs import ACOSolution
from ..helpers import (
    check_if_route_load_is_valid,
    get_route_load,
    get_route_arcs,
    # get_ls_max_time,
)


class VariableNeighborhoodDescent(GeneralVNS):
    def do_descent(
        self,
        initial_solution: List[List[int]],
        initial_costs: List[float],
        neighbordhood_structures: List[callable] = [
            single_route_relocate,
            single_route_swap,
        ],
    ):
        new_solution = deepcopy(initial_solution)
        new_costs = deepcopy(initial_costs)
        # max_descent = len(max(initial_solution, key=len)) * 5

        for idx, route in enumerate(initial_solution):
            max_descent = len(route) * 6

            best_route = route
            best_cost = initial_costs[idx]
            neighbordhoods = random.choices(
                neighbordhood_structures,
                k=max_descent,
            )

            for neighborhood in neighbordhoods:
                new_route = neighborhood(best_route)
                new_cost = self.model_problem.get_route_cost(
                    new_route, self.matrix_distances
                )

                if new_cost < best_cost:
                    best_route = new_route
                    best_cost = new_cost

            new_solution[idx] = best_route
            new_costs[idx] = best_cost

        return deepcopy(new_solution), deepcopy(new_costs)

    def improve(
        self,
        initial_solution: List[List[int]],
        initial_costs: List[float],
        max_time: float = None,
        intraroute_neighborhood_structure: List[callable] = [
            single_route_relocate,
            single_route_swap,
        ],
        shake_neighborhood_structures: List[callable] = [
            two_routes_swap_closest,
            two_routes_relocate,
            two_routes_swap,
            two_routes_exchange,
        ],
        curr_iteration: int = 0,
        max_iterations: int = 100,
    ) -> ACOSolution:
        best_solution = deepcopy(initial_solution)
        best_costs = deepcopy(initial_costs)
        # print("\n", sum(best_costs))
        # best_quality = sum(best_costs)
        # tabu_list: List[List[int]] = []

        if not max_time:
            max_time = max(len(self.lst_demands) * 0.0005, 0.065)

        count = 0

        shake_intensity = min(
            round(curr_iteration / max_iterations * 3),
            3,
        )
        shakes = random.choices(
            shake_neighborhood_structures,
            k=shake_intensity,
        )
        shake_solution = best_solution[:]

        start_time = time.time()
        while time.time() - start_time < max_time:
            for shake in shakes:
                shake_solution = shake(
                    solution=best_solution[:],
                    demands=self.lst_demands,
                    distances_matrix=self.matrix_distances,
                    max_capacity=self.max_capacity,
                )

            shake_costs = [
                self.model_problem.get_route_cost(route, self.matrix_distances)
                for route in shake_solution
            ]

            descent_solution, descent_costs = self.do_descent(
                shake_solution,
                shake_costs,
                intraroute_neighborhood_structure,
            )

            if sum(descent_costs) < sum(best_costs):
                best_solution = descent_solution
                best_costs = descent_costs
                # best_quality = sum(best_costs)
                # tabu_list = []

            count += 1

        if not all(
            [
                check_if_route_load_is_valid(
                    route, self.lst_demands, self.max_capacity
                )
                for route in best_solution
            ]
        ):
            raise ValueError(
                "One route has more than the vehicle capacity: {}".format(
                    [
                        get_route_load(route, self.lst_demands)
                        for route in best_solution
                    ]
                )
            )

        if sum(shake_costs) < sum(best_costs):
            best_solution = shake_solution
            best_costs = shake_costs

        return {
            "cost": sum(best_costs),
            "routes_arcs": [
                np.array(get_route_arcs(route)) for route in best_solution
            ],
            "routes_costs": best_costs,
            "routes_loads": [
                get_route_load(route, self.lst_demands)
                for route in best_solution
            ],
            "routes": best_solution,
        }
