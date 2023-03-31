import numpy as np
import random


class FreeAnt:
    def __init__(self, nodes, demands, max_capacity, tare, distances_matrix,
                 probabilities_matrix, q0):
        self.demands = demands
        self.max_capacity = max_capacity
        self.tare = tare
        self.distances_matrix = distances_matrix
        self.probabilities_matrix = probabilities_matrix
        self.q0 = q0
        self.depot = nodes[0]
        self.clients = nodes[1:]

    def move_to_next_node(self, actual_node, valid_nodes):
        probabilites_of_nodes = \
            self.probabilities_matrix[actual_node][valid_nodes]
        q = random.random()

        if q <= self.q0:
            return valid_nodes[probabilites_of_nodes.argmax()]
        else:
            # THIS APPROACH IS TOO SLOW
            # probabilities = np.divide(
            #     probabilites_of_nodes, probabilites_of_nodes.sum())
            # return np.random.choice(valid_nodes, 1, p=probabilities)[0]

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
        energy = 0
        vehicle_weight = self.tare
        vehicle_load = 0
        remaining_unvisited_nodes = unvisited_nodes

        valid_nodes = unvisited_nodes
        while valid_nodes.size:
            s = self.move_to_next_node(r, valid_nodes)

            energy += self.distances_matrix[r][s] * vehicle_weight
            vehicle_weight += self.demands[s]
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
        energy += self.distances_matrix[r][self.depot] * vehicle_weight

        return (route,
                energy,
                remaining_unvisited_nodes,
                vehicle_load)

    def set_probabilities_matrix(self, probabilities_matrix):
        self.probabilities_matrix = probabilities_matrix

    def generate_solution(self):
        solution = []
        energies = []
        loads = []

        unvisited_nodes = self.clients
        while unvisited_nodes.size:
            route, energy, remaining_unvisited_nodes, vehicle_load = \
                self.generate_route(unvisited_nodes)

            solution.append(route)
            energies.append(energy)
            loads.append(vehicle_load)

            unvisited_nodes = remaining_unvisited_nodes

        return solution, energies, loads
