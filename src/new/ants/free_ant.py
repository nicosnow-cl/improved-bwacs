from typing import Set
import numpy as np
import random

from ..helpers import get_route_arcs
from ..models.vehicle_model import VehicleModel
from .ant_solution import AntSolution


class FreeAnt:
    def __init__(
        self,
        nodes,
        lst_demands,
        matrix_probabilities,
        matrix_costs,
        max_capacity,
        tare,
        problem_model,
        q0: float = None,
    ):
        self.lst_demands = lst_demands
        self.matrix_probabilities = matrix_probabilities
        self.matrix_costs = matrix_costs
        self.max_capacity = max_capacity
        self.tare = tare
        self.q0 = q0
        self.problem_model = problem_model
        self.depot = nodes[0]
        self.clients = set(nodes[1:])

    def set_probabilities_matrix(self, probabilities_matrix):
        self.matrix_probabilities = probabilities_matrix

    def set_best_start_nodes(self, best_start_nodes):
        self.best_start_nodes = best_start_nodes

    def choose_next_node(self, actual_node, valid_nodes):
        probabilities_of_nodes = self.matrix_probabilities[actual_node][
            valid_nodes
        ]
        probabilities = probabilities_of_nodes / probabilities_of_nodes.sum()

        if self.q0 is None:
            return random.choices(valid_nodes, probabilities, k=1)[0]

        if random.random() <= self.q0:
            return valid_nodes[probabilities_of_nodes.argmax()]
        else:
            # return random.choices(valid_nodes, probabilities, k=1)[0]
            return np.random.choice(a=valid_nodes, size=1, p=probabilities)[0]

    def get_valid_nodes(self, unvisited_nodes, vehicle: VehicleModel):
        return [
            node
            for node in unvisited_nodes
            if vehicle["load"] + self.lst_demands[node]
            <= vehicle["max_capacity"]
        ]

    def get_valid_nodes_sorted_by_distance(
        self, r, unvisited_nodes, vehicle_load
    ):
        valid_nodes = self.get_valid_nodes(unvisited_nodes, vehicle_load)

        return valid_nodes[self.matrix_costs[r][valid_nodes].argsort()]

    def generate_route(
        self,
        unvisited_nodes: Set[int],
        actual_route: int,
        ant_best_start_nodes=None,
    ):
        r = self.depot
        route = [self.depot]
        route_cost = 0
        vehicle: VehicleModel = {"max_capacity": self.max_capacity, "load": 0}

        valid_nodes = list(unvisited_nodes)
        start_on_best_nodes = ant_best_start_nodes and actual_route == 0

        if start_on_best_nodes:
            np_weights = np.array(ant_best_start_nodes)
            s = valid_nodes[np_weights[valid_nodes].argmax()]

            route_cost += self.problem_model.get_cost_between_two_nodes(
                r, s, self.matrix_costs
            )
            vehicle["load"] += self.lst_demands[s]

            valid_nodes.remove(s)

            route.append(s)
            r = s

        while valid_nodes:
            s = self.choose_next_node(r, valid_nodes)

            route_cost += self.problem_model.get_cost_between_two_nodes(
                r, s, self.matrix_costs
            )

            vehicle["load"] += self.lst_demands[s]

            valid_nodes.remove(s)
            valid_nodes = self.get_valid_nodes(valid_nodes, vehicle)

            route.append(s)
            r = s

        route.append(self.depot)
        route_cost += self.problem_model.get_cost_between_two_nodes(
            r, self.depot, self.matrix_costs
        )

        return route, route_cost, vehicle["load"]

    def generate_solution(self, ant_best_start_nodes=[]) -> AntSolution:
        routes = []
        costs = []
        loads = []
        unvisited_nodes = self.clients.copy()

        while unvisited_nodes:
            route, cost, vehicle_load = self.generate_route(
                unvisited_nodes, len(routes), ant_best_start_nodes
            )

            routes.append(route)
            costs.append(cost)
            loads.append(vehicle_load)

            unvisited_nodes.difference_update(route)

        return {
            "cost": sum(costs),
            "routes_arcs": [
                np.array(get_route_arcs(route)) for route in routes
            ],
            "routes_costs": costs,
            "routes_loads": loads,
            "routes": routes,
        }
