from typing import Set
import numpy as np
import random

from ..helpers import get_route_arcs
from ..models.vehicle_model import VehicleModel


class FreeAnt:
    def __init__(self, nodes, demands, max_capacity, tare,
                 distances_matrix, probabilities_matrix, q0,
                 problem_model):
        self.demands = demands
        self.max_capacity = max_capacity
        self.tare = tare
        self.distances_matrix = distances_matrix
        self.probabilities_matrix = probabilities_matrix
        self.q0 = q0
        self.depot = nodes[0]
        self.clients = set(nodes[1:])
        self.problem_model = problem_model

    def set_probabilities_matrix(self, probabilities_matrix):
        self.probabilities_matrix = probabilities_matrix

    def set_best_start_nodes(self, best_start_nodes):
        self.best_start_nodes = best_start_nodes

    def move_to_next_node_legacy(self, actual_node, valid_nodes):
        probabilites_of_nodes = \
            self.probabilities_matrix[actual_node][valid_nodes]

        q = random.random()

        if q >= self.q0:
            return valid_nodes[probabilites_of_nodes.argmax()]
        else:
            cumsum = np.cumsum(probabilites_of_nodes)
            return random.choices(valid_nodes, cum_weights=cumsum, k=1)[0]

    def choose_next_node(self, actual_node, valid_nodes):
        probabilities_of_nodes = \
            self.probabilities_matrix[actual_node][valid_nodes]

        q = np.random.rand()

        if q >= self.q0:
            return valid_nodes[probabilities_of_nodes.argmax()]
        else:
            cum_weights = probabilities_of_nodes.cumsum()
            cum_weights /= cum_weights[-1]

            return valid_nodes[np.searchsorted(cum_weights, np.random.rand())]

    def get_valid_nodes(self, unvisited_nodes, vehicle: VehicleModel):
        return [node for node in unvisited_nodes
                if vehicle['load'] + self.demands[node]
                <= vehicle['max_capacity']]

    def get_valid_nodes_sorted_by_distance(self,
                                           r,
                                           unvisited_nodes,
                                           vehicle_load):
        valid_nodes = self.get_valid_nodes(unvisited_nodes, vehicle_load)

        return valid_nodes[self.distances_matrix[r][valid_nodes].argsort()]

    def generate_route(self,
                       unvisited_nodes: Set[int],
                       ant_best_start_nodes=None):
        r = self.depot
        route = [self.depot]
        route_cost = 0
        vehicle: VehicleModel = {'max_capacity': self.max_capacity, 'load': 0}

        valid_nodes = list(unvisited_nodes)

        if ant_best_start_nodes:
            s = self.choose_next_node(r, ant_best_start_nodes)

            route_cost += self.problem_model.get_cost_between_two_nodes(
                r, s, self.distances_matrix)
            vehicle['load'] += self.demands[s]

            valid_nodes.remove(s)

            route.append(s)
            r = s

        while valid_nodes:
            s = self.choose_next_node(r, valid_nodes)

            route_cost += self.problem_model.get_cost_between_two_nodes(
                r, s, self.distances_matrix)
            vehicle['load'] += self.demands[s]

            valid_nodes.remove(s)
            valid_nodes = self.get_valid_nodes(valid_nodes, vehicle)

            route.append(s)
            r = s

        route.append(self.depot)
        route_cost += self.problem_model.get_cost_between_two_nodes(
            r, self.depot, self.distances_matrix)

        return route, route_cost, vehicle['load']

    def generate_solution(self, ant_best_start_nodes=[]):
        solution = []
        costs = []
        loads = []
        unvisited_nodes = self.clients.copy()

        while unvisited_nodes:
            route, cost, vehicle_load = \
                self.generate_route(unvisited_nodes, ant_best_start_nodes)

            solution.append(route)
            costs.append(cost)
            loads.append(vehicle_load)

            unvisited_nodes.difference_update(route)
            ant_best_start_nodes = [
                node for node in ant_best_start_nodes if node not in route]

        fitness = sum(costs)
        routes_arcs = [np.array(get_route_arcs(route)) for route in solution]

        return solution, fitness, routes_arcs, costs, loads
