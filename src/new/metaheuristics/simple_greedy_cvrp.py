import numpy as np


class SimpleGreedy():
    def __init__(self, nodes, demands, max_capacity, tare, distances_matrix):
        self.demands = demands
        self.max_capacity = max_capacity
        self.tare = tare
        self.distances_matrix = distances_matrix
        self.depot = nodes[0]
        self.clients = nodes[1:]

    def generate_solution(self):
        num_customers = len(demand)
        unvisited = set(range(1, num_customers))
        routes = []

        while unvisited:
            remaining_capacity = capacity
            current_node = 0  # depot
            route = [current_node]

            while True:
                if not unvisited:
                    break

                nearest_neighbor = min(
                    unvisited, key=lambda x: distance_matrix[current_node][x])
                if demand[nearest_neighbor] > remaining_capacity:
                    break

                unvisited.remove(nearest_neighbor)
                remaining_capacity -= demand[nearest_neighbor]
                current_node = nearest_neighbor
                route.append(current_node)

            route.append(0)  # return to the depot
            routes.append(route)

        return routes
