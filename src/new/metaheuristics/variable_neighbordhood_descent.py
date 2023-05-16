from copy import deepcopy
from math import log
from typing import List
import numpy as np
import random
import time
from itertools import permutations, combinations

from .single_route_relocate import single_route_relocate
from .single_route_swap import single_route_swap
from .two_routes_swap import two_routes_swap
from .two_routes_relocate import two_routes_relocate
from .two_routes_exchange import two_routes_exchange
from .two_routes_swap_closest import two_routes_swap_closest
from .two_routes_relocate_closest import two_routes_relocate_closest
from .variable_neighbordhood_search import GeneralVNS
from ..acs import ACOSolution
from ..helpers import (
    check_if_route_load_is_valid,
    get_route_load,
    get_route_arcs,
    # get_ls_max_time,
)

PERMUTATIONS_MAX_LENGTH = 6
SAMPLES_MAX_LENGTH = 15
BASE = 1.2


class VariableNeighborhoodDescent(GeneralVNS):
    def do_descent(
        self,
        initial_solution: List[List[int]],
        initial_costs: List[float],
        neighborhood_structures: List[callable] = [
            single_route_relocate,
            single_route_swap,
        ],
    ):
        new_solution = deepcopy(initial_solution)
        new_costs = deepcopy(initial_costs)

        for idx, route in enumerate(initial_solution):
            route_length = len(route) - 2

            if route_length <= PERMUTATIONS_MAX_LENGTH:
                best_route, best_cost = self.permutations_descent(
                    route, initial_costs[idx]
                )
            # elif route_length <= SAMPLES_MAX_LENGTH:
            #     samples_max_descent = int(
            #         route_length * log(route_length, BASE)
            #     )
            #     best_route, best_cost = self.random_samples_descent(
            #         route, initial_costs[idx], max_descent=samples_max_descent
            #     )
            else:
                neighborhood_max_descent = int(route_length**2) * 2
                best_route, best_cost = self.neighborhood_structures_descent(
                    route,
                    initial_costs[idx],
                    max_descent=neighborhood_max_descent,
                    neighborhood_structures=neighborhood_structures,
                )

            new_solution[idx] = best_route
            new_costs[idx] = best_cost

        return deepcopy(new_solution), deepcopy(new_costs)

    def do_descent_one_route(
        self,
        initial_route: List[int],
        initial_cost: float,
        neighborhood_structures: List[callable] = [
            single_route_relocate,
            single_route_swap,
        ],
    ):
        route_length = len(initial_route) - 2

        if route_length <= PERMUTATIONS_MAX_LENGTH + 1:
            best_route, best_cost = self.permutations_descent(
                deepcopy(initial_route), initial_cost
            )
        else:
            neighborhood_max_descent = int(route_length**2) * 3
            best_route, best_cost = self.neighborhood_structures_descent(
                deepcopy(initial_route),
                initial_cost,
                max_descent=neighborhood_max_descent,
                neighborhood_structures=neighborhood_structures,
            )

        return best_route, best_cost

    def permutations_descent(
        self, route: List[int], cost: float, omit_inversed: bool = True
    ):
        # start_time = time.time()

        best_route = deepcopy(route)
        best_cost = cost

        route_without_depot = best_route[1:-1]
        neighborhoods = list(permutations(route_without_depot))

        for neighborhood in neighborhoods:
            if omit_inversed and neighborhood <= neighborhood[::-1]:
                continue

            new_route = [0] + list(neighborhood) + [0]
            new_cost = self.model_problem.get_route_cost(
                new_route, self.matrix_distances
            )

            if new_cost < best_cost:
                best_route = new_route
                best_cost = new_cost
                # break

        # print("permutations:", time.time() - start_time)
        return best_route, best_cost

    def random_samples_descent(
        self, route: List[int], cost: float, max_descent: int = 75
    ):
        best_route = deepcopy(route)
        best_cost = cost

        route_without_depot = best_route[1:-1]
        route_size = len(route_without_depot)

        for _ in range(max_descent):
            neighborhood = random.sample(route_without_depot, route_size)

            new_route = [0] + neighborhood + [0]
            new_cost = self.model_problem.get_route_cost(
                new_route, self.matrix_distances
            )

            if new_cost < best_cost:
                best_route = new_route
                best_cost = new_cost
                break

        return best_route, best_cost

    def neighborhood_structures_descent(
        self,
        route: List[int],
        cost: float,
        max_descent: int = 100,
        neighborhood_structures: List[callable] = [
            single_route_relocate,
            single_route_swap,
        ],
    ):
        # start_time = time.time()

        best_route = deepcopy(route)
        best_cost = cost

        neighbordhoods = random.choices(
            neighborhood_structures,
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
                # break

        # print("neighborhoods:", time.time() - start_time)
        return best_route, best_cost

    def improve(
        self,
        initial_solution: List[List[int]],
        initial_costs: List[float],
        max_time: float = None,
        descent_neighborhood_structures: List[callable] = [
            single_route_relocate,
            single_route_swap,
        ],
        shake_neighborhood_structures: List[callable] = [
            # two_routes_relocate,
            # two_routes_swap,
            # two_routes_exchange,
            two_routes_relocate_closest,
            two_routes_swap_closest,
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
            max_time = max(len(self.lst_demands) * 0.0005, 0.1) * (
                1 - ((max_iterations - curr_iteration) / max_iterations)
            )

        intensity = min(round(curr_iteration / max_iterations * 3), 3)
        count = 0

        start_time = time.time()
        while time.time() - start_time < max_time:
            shake_solution = best_solution[:]

            shake_intensity = random.randint(0, intensity)
            shakes = random.choices(
                shake_neighborhood_structures,
                k=shake_intensity,
            )

            routes_idx = None

            shakes_done = 0
            while shakes_done < shake_intensity:
                shake = shakes[shakes_done]
                shake_solution, is_new, routes_idx = shake(
                    solution=shake_solution[:],
                    demands=self.lst_demands,
                    distances_matrix=self.matrix_distances,
                    max_capacity=self.max_capacity,
                )

                if is_new:
                    # print(is_new)
                    shakes_done += 1

            # for shake in shakes:
            #     shake_solution, is_new = shake(
            #         solution=shake_solution[:],
            #         demands=self.lst_demands,
            #         distances_matrix=self.matrix_distances,
            #         max_capacity=self.max_capacity,
            #     )

            shake_costs = [
                self.model_problem.get_route_cost(route, self.matrix_distances)
                for route in shake_solution
            ]

            if not routes_idx:
                descent_solution, descent_costs = self.do_descent(
                    shake_solution,
                    shake_costs,
                    descent_neighborhood_structures,
                )

                if sum(descent_costs) < sum(best_costs):
                    best_solution = descent_solution
                    best_costs = descent_costs
                    # best_quality = sum(best_costs)
                    # tabu_list = []
            else:
                for route_idx in routes_idx:
                    descent_route, descent_cost = self.do_descent_one_route(
                        shake_solution[route_idx],
                        shake_costs[route_idx],
                        descent_neighborhood_structures,
                    )

                    shake_solution[route_idx] = descent_route
                    shake_costs[route_idx] = descent_cost

                if sum(shake_costs) < sum(best_costs):
                    best_solution = shake_solution
                    best_costs = shake_costs
                    # best_quality = sum(best_costs)
                    # tabu_list = []

            count += 1

        # print(count)
        # print("max_time:", max_time)
        # print("Elapsed", time.time() - start_time)
        # print("\n")
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
