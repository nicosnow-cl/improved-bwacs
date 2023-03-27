import random
import time


class GeneralVNS():
    def __init__(self,
                 distances_matrix,
                 demands_arr,
                 tare,
                 vehicle_capacity,
                 k_number,
                 max_iterations,
                 #  time_limit,
                 #  fitness_function
                 ):
        self.distances_matrix = distances_matrix
        self.demands_array = demands_arr
        self.tare = tare
        self.vehicle_capacity = vehicle_capacity
        self.k_number = k_number
        self.max_iterations = max_iterations
        # self.time_limit = time_limit
        # self.fitness_function = fitness_function

    # Only for EMVRP
    def fitness_function(self, route):
        route_energy = 0
        vehicle_weight = 0
        prev_node = None

        for i in route:
            if i == 0:
                vehicle_weight = self.tare
                prev_node = i
            else:
                route_energy += self.distances_matrix[prev_node][i] * \
                    vehicle_weight
                vehicle_weight += self.demands_array[i]
                prev_node = i

        return route_energy

    def fitness(self, solution):
        total = 0

        for route in solution:
            total += self.fitness_function(route)

        return total

    def validate_route_capacity(self, route):
        route_capacity = sum(self.demands_array[route])

        if route_capacity > self.vehicle_capacity:
            return False
        else:
            return True

    def shake(self, solution):
        route_index1, route_index2 = random.sample(range(len(solution)), 2)

        route1 = solution[route_index1].copy()
        route2 = solution[route_index2].copy()

        pop_index = random.randint(1, len(route1) - 2)
        node = route1.pop(pop_index)

        insert_index = random.randint(1, len(route2) - 1)
        route2.insert(insert_index, node)

        new_solution = solution[:]
        new_solution[route_index1] = route1
        new_solution[route_index2] = route2

        return new_solution

    def single_route_relocate(self, solution):
        selected_route_index = random.randint(0, len(solution)-1)
        selected_route = solution[selected_route_index]

        if len(selected_route) <= 3:
            return solution

        i, j = sorted(random.sample(range(1, len(selected_route)-2), 2))
        node = selected_route[i]

        new_route = selected_route[:i] + \
            selected_route[i+1:j] + [node] + selected_route[j:]
        new_solution = solution[:]
        new_solution[selected_route_index] = new_route

        if self.fitness(new_solution) < self.fitness(solution):
            return new_solution
        else:
            return solution

    def single_route_swap(self, solution):
        selected_route_index = random.randint(0, len(solution) - 1)
        selected_route = solution[selected_route_index]

        node_index1 = random.randint(1, len(selected_route) - 2)
        node_index2 = random.randint(1, len(selected_route) - 2)

        while node_index2 == node_index1:
            node_index2 = random.randint(1, len(selected_route) - 2)

        new_route = selected_route[:]
        new_route[node_index1], new_route[node_index2] = new_route[node_index2], new_route[node_index1]
        new_solution = solution[:]
        new_solution[selected_route_index] = new_route

        if self.fitness(new_solution) < self.fitness(solution):
            return new_solution
        else:
            return solution

    def two_routes_relocate(self, solution):
        route_index1, route_index2 = random.sample(range(len(solution)), 2)

        route1 = solution[route_index1].copy()
        route2 = solution[route_index2].copy()

        pop_index = random.randint(1, len(route1) - 2)
        node = route1.pop(pop_index)

        insert_index = random.randint(1, len(route2) - 1)
        route2.insert(insert_index, node)

        if self.validate_route_capacity(route2) is not True:
            return solution

        new_solution = solution[:]
        new_solution[route_index1] = route1
        new_solution[route_index2] = route2

        if self.fitness(new_solution) < self.fitness(solution):
            return new_solution
        else:
            return solution

    def two_routes_swap(self, solution):
        if len(solution) < 2:
            return solution

        route_index1, route_index2 = random.sample(range(len(solution)), 2)

        route1 = solution[route_index1].copy()
        route2 = solution[route_index2].copy()

        node_index1 = random.randint(1, len(route1) - 2)
        node_index2 = random.randint(1, len(route2) - 2)
        node1 = route1[node_index1]
        node2 = route2[node_index2]

        route1[node_index1] = node2
        route2[node_index2] = node1

        if self.validate_route_capacity(route1) is not True \
                and self.validate_route_capacity(route2) is not True:
            return solution

        new_solution = solution[:]
        new_solution[route_index1] = route1
        new_solution[route_index2] = route2

        if self.fitness(new_solution) < self.fitness(solution):
            return new_solution
        else:
            return solution

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

        if self.validate_route_capacity(route1) is not True \
                and self.validate_route_capacity(route2) is not True:
            return solution

        new_solution = solution[:]
        new_solution[route_index1] = route1
        new_solution[route_index2] = route2

        if self.fitness(new_solution) < self.fitness(solution):
            return new_solution
        else:
            return solution

    def improve(self, initial_solution, actual_iteration):
        best_solution = initial_solution.copy()
        intensity_percentage = (actual_iteration / self.max_iterations)
        max_time = len(self.demands_array) * \
            0.01 if intensity_percentage <= 0.85 else len(
                self.demands_array) * 0.015
        neighborhoods = [self.single_route_relocate,
                         self.single_route_swap,
                         self.two_routes_relocate,
                         self.two_routes_swap,
                         self.two_routes_exchange]

        start_time = time.time()

        while time.time() - start_time < max_time:
            shaked_solution = self.shake(best_solution)

            [neighborhood] = random.choices(
                neighborhoods, weights=(30, 30, 20, 15, 5), k=1)
            neighborhood_solution = neighborhood(shaked_solution)

            if all([self.validate_route_capacity(route) is not True for route
                    in neighborhood_solution]):
                if self.fitness(neighborhood_solution) < \
                        self.fitness(best_solution):
                    best_solution = neighborhood_solution

        if all([self.validate_route_capacity(route) is not True for route
                in best_solution]):
            raise ValueError(
                "One of the routes has more than the vehicle capacity")

        return best_solution, self.fitness(best_solution)
