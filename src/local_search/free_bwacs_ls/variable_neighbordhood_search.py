import random
import time
# import numpy as np


class GeneralVNS():
    def __init__(self,
                 distances_matrix,
                 demands_array,
                 tare,
                 vehicle_capacity,
                 k_number,
                 max_iterations,
                 #  time_limit,
                 #  fitness_function
                 user_fitness_by_route=None
                 ):
        self.distances_matrix = distances_matrix
        self.demands_array = demands_array
        self.tare = tare
        self.vehicle_capacity = vehicle_capacity
        self.k_number = k_number
        self.max_iterations = max_iterations
        # self.time_limit = time_limit
        # self.fitness_function = fitness_function
        self.user_fitness_by_route = user_fitness_by_route

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

    # Only for VRP
    def fitness_function(self, route):
        route_cost = 0
        prev_node = None

        for pos, i in enumerate(route):
            if pos == 0:
                prev_node = i
                continue
            else:
                route_cost += self.distances_matrix[prev_node][i]

            prev_node = i

        return route_cost

    def apply_fitness_by_route(self, solution):
        return [self.fitness_function(route) for route in solution]

    def fitness(self, solution):
        if self.user_fitness_by_route is not None:
            return sum(self.user_fitness_by_route(solution))
        else:
            return sum(self.apply_fitness_by_route(solution))

    def get_route_load(self, route):
        return sum(self.demands_array[route])

    def check_if_route_load_is_valid(self, route):
        route_load = self.get_route_load(route)

        return route_load <= self.vehicle_capacity

    def shake(self, solution):
        route_index1, route_index2 = random.sample(range(len(solution)), 2)

        route1 = solution[route_index1].copy()
        route2 = solution[route_index2].copy()

        if len(route1) <= 3:
            return solution

        pop_index = random.randint(1, len(route1) - 2)
        node = route1.pop(pop_index)

        insert_index = random.randint(1, len(route2) - 2)
        route2.insert(insert_index, node)

        new_solution = solution
        new_solution[route_index1] = route1
        new_solution[route_index2] = route2

        return new_solution

    def single_route_relocate(self, solution):
        selected_route_index = random.randint(0, len(solution) - 1)
        selected_route = solution[selected_route_index].copy()

        if len(selected_route) < 4:
            return solution

        i, j = random.sample(range(1, len(selected_route) - 1), 2)
        node = selected_route[i]

        new_route = selected_route[:i] + selected_route[i + 1:j] \
            + [node] + selected_route[j:]
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
        new_route[node_index1], new_route[node_index2] = \
            new_route[node_index2], new_route[node_index1]
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

        is_route2_valid = self.check_if_route_load_is_valid(route2)

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

        is_route1_valid = self.check_if_route_load_is_valid(route1)
        is_route2_valid = self.check_if_route_load_is_valid(route2)

        if not is_route1_valid and not is_route2_valid:
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

        if len(route1) <= 3 or len(route2) <= 3:
            return solution

        route1_node_index1, route1_node_index2 = random.sample(
            range(1, len(route1)-2), 2)
        route2_node_index1, route2_node_index2 = random.sample(
            range(1, len(route2)-2), 2)

        route1_node1 = route1[route1_node_index1]
        route1_node2 = route1[route1_node_index2]

        route2_node1 = route2[route2_node_index1]
        route2_node2 = route2[route2_node_index2]

        route1[route1_node_index1] = route2_node1
        route1[route1_node_index2] = route2_node2

        route2[route2_node_index1] = route1_node1
        route2[route2_node_index2] = route1_node2

        if self.check_if_route_load_is_valid(route1) is not True \
                and self.check_if_route_load_is_valid(route2) is not True:
            return solution

        new_solution = solution[:]
        new_solution[route_index1] = route1
        new_solution[route_index2] = route2

        return new_solution

    def improve(self, initial_solution, actual_iteration):
        best_solution = initial_solution.copy()

        intensity_percentage = (actual_iteration / self.max_iterations)
        max_time = len(self.demands_array) * \
            0.0005 if intensity_percentage <= 0.85 else len(
                self.demands_array) * 0.00001
        neighborhoods = [self.single_route_relocate,
                         self.single_route_swap,
                         self.two_routes_relocate,
                         self.two_routes_swap,
                         #  self.two_routes_exchange
                         ]

        start_time = time.time()
        while time.time() - start_time < max_time:
            actual_solution = self.two_routes_swap(best_solution.copy())
            actual_solution_quality = self.fitness(actual_solution)

            for _ in range(4):
                [neighborhood] = random.choices(neighborhoods,
                                                weights=(30, 25, 15, 10),
                                                k=1)
                try:
                    nb_solution = neighborhood(
                        actual_solution.copy())
                except ValueError as e:
                    print(f'ValueError: {e} on {neighborhood.__name__}')
                    print(actual_solution)
                    print([self.check_if_route_load_is_valid(route) for
                           route in actual_solution])
                    continue

                nb_solution_quality = self.fitness(nb_solution)
                nb_solution_validate_load = all([
                    self.check_if_route_load_is_valid(route) for route in
                    nb_solution])

                if nb_solution_validate_load and \
                        len(nb_solution) == self.k_number:
                    if nb_solution_quality < actual_solution_quality:
                        actual_solution = nb_solution.copy()
                        actual_solution_quality = nb_solution_quality

            if actual_solution_quality < self.fitness(best_solution):
                best_solution = actual_solution.copy()

        if all([self.check_if_route_load_is_valid(route) for route in
                best_solution]):
            raise ValueError(
                "One of the routes has more than the vehicle capacity")

        best_solution_costs = self.apply_fitness_by_route(best_solution)
        best_solution_loads = [self.get_route_load(route) for route in
                               best_solution]

        return best_solution, best_solution_costs, best_solution_loads
