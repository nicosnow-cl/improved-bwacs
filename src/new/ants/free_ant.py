import numpy as np
import random

from .ant_solution import AntSolution


class FreeAnt:
    def __init__(
        self,
        nodes,
        lst_demands,
        matrix_probabilities,
        matrix_pheromones,
        matrix_heuristics,
        matrix_costs,
        max_capacity,
        tare,
        problem_model,
        q0: float = None,
    ):
        self.lst_demands = lst_demands
        self.matrix_probabilities = matrix_probabilities
        self.matrix_pheromones = matrix_pheromones
        self.matrix_heuristics = matrix_heuristics
        self.matrix_costs = matrix_costs
        self.max_capacity = max_capacity
        self.tare = tare
        self.q0 = q0
        self.problem_model = problem_model
        self.depot = nodes[0]
        self.clients = set(nodes[1:])
        self.depot_cost = self.problem_model.get_cost_between_two_nodes(
            self.depot, self.depot, self.matrix_costs
        )

    def set_probabilities_matrix(self, probabilities_matrix):
        self.matrix_probabilities = probabilities_matrix

    def set_pheromones_matrix(self, pheromones_matrix):
        self.matrix_pheromones = pheromones_matrix

    def set_heuristics_matrix(self, heuristics_matrix):
        self.matrix_heuristics = heuristics_matrix

    def set_best_start_nodes(self, best_start_nodes):
        self.best_start_nodes = best_start_nodes

    def choose_next_node(self, actual_node, valid_nodes):
        prob_of_nodes = self.matrix_probabilities[actual_node][valid_nodes]

        if self.q0 is None:
            return random.choices(valid_nodes, weights=prob_of_nodes, k=1)[0]

        if random.random() <= self.q0:
            return valid_nodes[np.argmax(prob_of_nodes)]
        else:
            return random.choices(valid_nodes, weights=prob_of_nodes, k=1)[0]

    def get_valid_nodes(self, unvisited_nodes, vehicle_load):
        remaining_capacity = self.max_capacity - vehicle_load
        return [
            node
            for node in unvisited_nodes
            if self.lst_demands[node] <= remaining_capacity
        ]

    def generate_route(
        self,
        unvisited_nodes: set,
        ant_best_start_nodes=None,
    ):
        route = [self.depot]
        route_arcs = []
        route_cost = 0
        vehicle_load = 0

        valid_nodes = list(unvisited_nodes)

        if ant_best_start_nodes:
            np_weights = np.array(ant_best_start_nodes)
            s = valid_nodes[np_weights[valid_nodes].argmax()]

            route_cost += self.matrix_costs[self.depot][s]
            vehicle_load += self.lst_demands[s]

            valid_nodes.remove(s)

            route.append(s)
            route_arcs.append((self.depot, s))

        while valid_nodes:
            s = self.choose_next_node(route[-1], valid_nodes)

            route_cost += self.matrix_costs[route[-1]][s]

            vehicle_load += self.lst_demands[s]

            valid_nodes.remove(s)
            valid_nodes = self.get_valid_nodes(valid_nodes, vehicle_load)

            route.append(s)
            route_arcs.append((route[-2], s))

        route.append(self.depot)
        route_arcs.append((route[-2], self.depot))
        route_cost += self.matrix_costs[route[-2]][self.depot]

        return route, route_arcs, route_cost, vehicle_load

    def generate_solution(self, ant_best_start_nodes=None) -> AntSolution:
        solution = {
            "cost": np.inf,
            "routes_arcs": [],
            "routes_arcs_flatten": [],
            "routes_costs": [],
            "routes_loads": [],
            "routes": [],
        }

        unvisited_nodes = self.clients.copy()

        while unvisited_nodes:
            route, route_arcs, route_cost, route_load = self.generate_route(
                unvisited_nodes, ant_best_start_nodes
            )

            solution["routes"].append(route)
            solution["routes_arcs"].append(route_arcs)
            solution["routes_arcs_flatten"].extend(route_arcs)
            solution["routes_costs"].append(route_cost)
            solution["routes_loads"].append(route_load)

            unvisited_nodes.difference_update(route)

        solution["cost"] = sum(solution["routes_costs"])
        return solution
