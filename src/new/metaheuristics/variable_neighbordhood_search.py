import random
import time
import numpy as np

from ..helpers import check_if_route_load_is_valid, get_route_load, \
    get_route_arcs, get_ls_max_time


class GeneralVNS():
    def __init__(self,
                 distances_matrix,
                 demands_array,
                 tare,
                 vehicle_capacity,
                 k_number,
                 max_iterations,
                 problem_model,
                 time_limit=None,
                 ):
        self.distances_matrix = distances_matrix
        self.demands_array = demands_array
        self.tare = tare
        self.vehicle_capacity = vehicle_capacity
        self.k_number = k_number
        self.max_iterations = max_iterations
        self.time_limit = time_limit
        self.problem_model = problem_model

    # Only for EMVRP
    # def fitness_function(self, route):
    #     route_energy = 0
    #     vehicle_weight = None
    #     prev_node = None

    #     for pos, i in enumerate(route):
    #         if pos == 0:
    #             vehicle_weight = self.tare
    #         else:
    #             route_energy += self.distances_matrix[prev_node][i] * \
    #                 vehicle_weight
    #             vehicle_weight += self.demands_array[i]

    #         prev_node = i

    #     return route_energy

    def single_route_relocate(self, solution):
        selected_route_index = random.randint(0, len(solution) - 1)
        selected_route = solution[selected_route_index].copy()

        if len(selected_route) < 4:
            return solution

        node_index1, new_index_position = random.sample(
            range(1, len(selected_route) - 1), 2)

        new_route = selected_route[:]
        new_route = selected_route[:]
        new_route.insert(new_index_position, new_route.pop(node_index1))

        new_solution = solution[:]
        new_solution[selected_route_index] = new_route

        return new_solution

    def single_route_swap(self, solution):
        selected_route_index = random.randint(0, len(solution) - 1)
        selected_route = solution[selected_route_index].copy()

        if len(selected_route) < 4:
            return solution

        node_index1, node_index2 = random.sample(
            range(1, len(selected_route) - 1), 2)

        new_route = selected_route[:]
        new_route[node_index1] = selected_route[node_index2]
        new_route[node_index2] = selected_route[node_index1]

        new_solution = solution[:]
        new_solution[selected_route_index] = new_route

        return new_solution

    def two_routes_relocate(self, solution):
        if len(solution) < 2:
            return solution

        route_index1, route_index2 = random.sample(range(len(solution)), 2)

        route1 = solution[route_index1].copy()
        route2 = solution[route_index2].copy()

        if len(route1) < 3 or len(route2) < 3:
            return solution

        pop_index = random.randint(1, len(route1) - 2)
        node = route1.pop(pop_index)

        insert_index = random.randint(1, len(route2) - 2)
        route2.insert(insert_index, node)

        is_route2_valid = check_if_route_load_is_valid(
            route2, self.demands_array, self.vehicle_capacity)

        if not is_route2_valid:
            return solution

        new_solution = solution[:]
        new_solution[route_index1] = route1
        new_solution[route_index2] = route2

        return new_solution

    def two_routes_swap(self, solution):
        if len(solution) < 2:
            return solution

        route_index1, route_index2 = random.sample(range(len(solution)), 2)

        route1 = solution[route_index1].copy()
        route2 = solution[route_index2].copy()

        if len(route1) < 3 or len(route2) < 3:
            return solution

        node_index1 = random.randint(1, len(route1) - 2)
        node_index2 = random.randint(1, len(route2) - 2)
        node1 = route1[node_index1]
        node2 = route2[node_index2]

        route1[node_index1] = node2
        route2[node_index2] = node1

        is_route1_valid = check_if_route_load_is_valid(
            route1, self.demands_array, self.vehicle_capacity)
        is_route2_valid = check_if_route_load_is_valid(
            route2, self.demands_array, self.vehicle_capacity)

        if not is_route1_valid or not is_route2_valid:
            return solution

        new_solution = solution[:]
        new_solution[route_index1] = route1
        new_solution[route_index2] = route2

        return new_solution

    def two_routes_exchange(self, solution):
        if len(solution) < 2:
            return solution

        route_index1, route_index2 = random.sample(range(len(solution)), 2)

        route1 = solution[route_index1].copy()
        route2 = solution[route_index2].copy()

        if len(route1) < 4 or len(route2) < 4:
            return solution

        route1_node_index1 = random.randint(1, len(route1) - 3)
        route1_node_index2 = route1_node_index1 + 1

        route2_node_index1 = random.randint(1, len(route2) - 3)
        route2_node_index2 = route2_node_index1 + 1

        new_route1 = route1[:]
        new_route2 = route2[:]

        new_route2[route2_node_index1] = route1[route1_node_index1]
        new_route2[route2_node_index2] = route1[route1_node_index2]

        new_route1[route1_node_index1] = route2[route2_node_index1]
        new_route1[route1_node_index2] = route2[route2_node_index2]

        is_route1_valid = check_if_route_load_is_valid(
            new_route1, self.demands_array, self.vehicle_capacity)
        is_route2_valid = check_if_route_load_is_valid(
            new_route2, self.demands_array, self.vehicle_capacity)

        if not is_route1_valid or not is_route2_valid:
            return solution

        new_solution = solution[:]
        new_solution[route_index1] = new_route1
        new_solution[route_index2] = new_route2

        return new_solution

    def improve(self, initial_solution, actual_iteration):
        best_solution = initial_solution.copy()
        best_solution_costs = self.problem_model.fitness(
            best_solution, self.distances_matrix)
        best_solution_quality = sum(best_solution_costs)

        max_time = self.time_limit if self.time_limit is not None \
            else get_ls_max_time(len(self.demands_array), actual_iteration,
                                 self.max_iterations)

        neighborhoods_samples = [self.single_route_relocate,
                                 self.single_route_swap,
                                 self.two_routes_relocate,
                                 self.two_routes_swap,
                                 self.two_routes_exchange
                                 ]
        shake = neighborhoods_samples[-1]

        neighborhoods_ranking = {}
        for neighborhood in neighborhoods_samples:
            neighborhoods_ranking[neighborhood.__name__] = 0

        start_time = time.time()
        while time.time() - start_time < max_time:
            actual_solution = shake(best_solution.copy())
            actual_solution_costs = self.problem_model.fitness(
                actual_solution, self.distances_matrix)
            actual_solution_quality = sum(actual_solution_costs)
            neighborhoods = random.choices(neighborhoods_samples,
                                           weights=(5, 4, 3, 2, 1),
                                           k=self.k_number)

            for neighborhood in neighborhoods:
                try:
                    nb_solution = neighborhood(actual_solution.copy())
                    is_nb_solution_valid = all([
                        check_if_route_load_is_valid(route,
                                                     self.demands_array,
                                                     self.vehicle_capacity)
                        for route in nb_solution])

                    if is_nb_solution_valid:
                        nb_solution_costs = self.problem_model.fitness(
                            nb_solution, self.distances_matrix)
                        nb_solution_quality = sum(nb_solution_costs)

                        if nb_solution_quality < actual_solution_quality:
                            actual_solution = nb_solution.copy()
                            actual_solution_quality = nb_solution_quality
                            neighborhoods_ranking[neighborhood.__name__] += 1

                except ValueError as e:
                    print(f'ValueError: {e} on {neighborhood.__name__}')
                    print(actual_solution)
                    print([check_if_route_load_is_valid(route,
                                                        self.demands_array,
                                                        self.vehicle_capacity)
                           for route in actual_solution])
                    continue

            if actual_solution_quality < best_solution_quality:
                best_solution = actual_solution.copy()
                best_solution_costs = \
                    self.problem_model.fitness(
                        best_solution, self.distances_matrix)
                best_solution_quality = sum(best_solution_costs)

        if not all([check_if_route_load_is_valid(route,
                                                 self.demands_array,
                                                 self.vehicle_capacity)
                    for route in best_solution]):
            raise ValueError(
                "One of the routes has more than the vehicle capacity")

        # print(neighborhoods_ranking)

        return (best_solution,
                best_solution_quality,
                [np.array(get_route_arcs(route)) for route in best_solution],
                best_solution_costs,
                [get_route_load(route, self.demands_array)
                    for route in best_solution])
