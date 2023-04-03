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

        is_route1_valid = self.check_if_route_load_is_valid(new_route1)
        is_route2_valid = self.check_if_route_load_is_valid(new_route2)

        if not is_route1_valid or not is_route2_valid:
            return solution

        new_solution = solution[:]
        new_solution[route_index1] = new_route1
        new_solution[route_index2] = new_route2

        return new_solution

    def improve(self, initial_solution, actual_iteration):
        best_solution = initial_solution.copy()
        best_solution_costs = self.apply_fitness_by_route(best_solution)
        best_solution_quality = self.fitness(best_solution)

        intensity_percentage = (actual_iteration / self.max_iterations)
        intensity_factor = 0.0005 if intensity_percentage <= 0.85 else 0.001
        max_time = len(self.demands_array) * intensity_factor

        shake = self.two_routes_exchange
        neighborhoods_samples = [self.single_route_relocate,
                                 self.single_route_swap,
                                 self.two_routes_relocate,
                                 self.two_routes_swap,
                                 self.two_routes_exchange
                                 ]

        neighborhoods_ranking = {}
        for neighborhood in neighborhoods_samples:
            neighborhoods_ranking[neighborhood.__name__] = 0

        start_time = time.time()
        while time.time() - start_time < max_time:
            actual_solution = shake(best_solution.copy())
            actual_solution_quality = self.fitness(actual_solution)
            neighborhoods = random.choices(neighborhoods_samples,
                                           weights=(5, 4, 3, 2, 1),
                                           k=5)

            for neighborhood in neighborhoods:
                try:
                    nb_solution = neighborhood(actual_solution.copy())
                    is_nb_solution_valid = all([
                        self.check_if_route_load_is_valid(route) for route in
                        nb_solution])

                    if is_nb_solution_valid:
                        nb_solution_quality = self.fitness(nb_solution)

                        if nb_solution_quality < actual_solution_quality:
                            actual_solution = nb_solution.copy()
                            actual_solution_quality = nb_solution_quality
                            neighborhoods_ranking[neighborhood.__name__] += 1

                except ValueError as e:
                    print(f'ValueError: {e} on {neighborhood.__name__}')
                    print(actual_solution)
                    print([self.check_if_route_load_is_valid(route) for
                           route in actual_solution])
                    continue

            if actual_solution_quality < best_solution_quality:
                best_solution = actual_solution.copy()
                best_solution_costs = self.apply_fitness_by_route(
                    best_solution)
                best_solution_quality = actual_solution_quality

        if not all([self.check_if_route_load_is_valid(route) for route in
                    best_solution]):
            raise ValueError(
                "One of the routes has more than the vehicle capacity")

        # print(neighborhoods_ranking)
        return best_solution, best_solution_costs, [self.get_route_load(route)
                                                    for route in best_solution]
