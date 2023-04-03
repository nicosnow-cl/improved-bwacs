import numpy as np
import random


class FreeAnt:
    def __init__(self, nodes, demands, max_capacity, tare, distances_matrix,
                 probabilities_matrix, q0, best_start_nodes=None):
        self.demands = demands
        self.max_capacity = max_capacity
        self.tare = tare
        self.distances_matrix = distances_matrix
        self.probabilities_matrix = probabilities_matrix
        self.q0 = q0
        self.depot = nodes[0]
        self.clients = nodes[1:]
        self.best_start_nodes = best_start_nodes

    def set_probabilities_matrix(self, probabilities_matrix):
        self.probabilities_matrix = probabilities_matrix

    def set_best_start_nodes(self, best_start_nodes):
        self.best_start_nodes = best_start_nodes

    def move_to_next_node(self, actual_node, valid_nodes):
        probabilites_of_nodes = \
            self.probabilities_matrix[actual_node][valid_nodes]

        q = random.random()

        if q >= self.q0:
            return valid_nodes[probabilites_of_nodes.argmax()]
        else:
            # THIS APPROACH IS TOO SLOW
            # probabilities = np.divide(
            #     probabilites_of_nodes, probabilites_of_nodes.sum())
            # return np.random.choice(valid_nodes, 1, p=probabilities)[0]

            # THIS APPROACH GIVE US WORST RESULTS
            # return random.choices(valid_nodes, weights=probabilites_of_nodes,
            #                       k=1)[0]

            cumsum = np.cumsum(probabilites_of_nodes)
            return random.choices(valid_nodes, cum_weights=cumsum, k=1)[0]

    def get_valid_nodes(self, unvisited_nodes, vehicle_load):
        next_vehicle_loads = vehicle_load + self.demands[unvisited_nodes]

        return unvisited_nodes[next_vehicle_loads <= self.max_capacity]

    def get_valid_nodes_sorted_by_distance(self,
                                           r,
                                           unvisited_nodes,
                                           vehicle_load):
        valid_nodes = self.get_valid_nodes(unvisited_nodes, vehicle_load)

        return valid_nodes[self.distances_matrix[r][valid_nodes].argsort()]

    def generate_route(self, unvisited_nodes):
        r = self.depot
        route = [self.depot]
        route_cost = 0
        vehicle_load = 0
        remaining_unvisited_nodes = unvisited_nodes

        if self.best_start_nodes is not None and \
                len(self.best_start_nodes) > 0:
            # s = random.choice(self.best_start_nodes)
            s = self.move_to_next_node(r, self.best_start_nodes)

            route_cost += self.distances_matrix[r][s]  # Only for VRP
            vehicle_load += self.demands[s]
            remaining_unvisited_nodes = \
                remaining_unvisited_nodes[remaining_unvisited_nodes != s]
            self.best_start_nodes = list(
                filter(lambda node: node != s, self.best_start_nodes))

            route.append(s)
            r = s

        valid_nodes = remaining_unvisited_nodes
        while valid_nodes.size:
            s = self.move_to_next_node(r, valid_nodes)

            # route_cost += self.distances_matrix[r][s] * \
            # (vehicle_load + self.tare) # Only for EMVRP
            route_cost += self.distances_matrix[r][s]  # Only for VRP
            vehicle_load += self.demands[s]
            remaining_unvisited_nodes = \
                remaining_unvisited_nodes[remaining_unvisited_nodes != s]

            valid_nodes = self.get_valid_nodes(
                remaining_unvisited_nodes, vehicle_load)

            # valid_nodes = self.get_valid_nodes_sorted_by_distance(
            #     s, remaining_unvisited_nodes, vehicle_load)

            route.append(s)
            r = s

        route.append(self.depot)
        # route_cost += self.distances_matrix[r][self.depot] * \
        # (vehicle_load + self.tare) # Only for EMVRP
        route_cost += self.distances_matrix[r][self.depot]  # Only for VRP

        return (route,
                route_cost,
                vehicle_load,
                remaining_unvisited_nodes)

    def generate_solution(self):
        solution = []
        costs = []
        loads = []

        unvisited_nodes = self.clients
        while unvisited_nodes.size:
            route, cost, vehicle_load, remaining_unvisited_nodes = \
                self.generate_route(unvisited_nodes)

            solution.append(route)
            costs.append(cost)
            loads.append(vehicle_load)

            unvisited_nodes = remaining_unvisited_nodes

        return solution, costs, loads
