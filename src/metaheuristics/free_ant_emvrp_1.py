class FreeAntEMVRP_1:
    def __init__(self, depot, nodes, start, combinations_matrix,
                 distances_matrix, demands_array, vehicle_capacity, tare,
                 alpha, beta, q0):
        import numpy as np
        import random

        self.depot = depot
        self.nodes = nodes
        self.start = start
        self.combinations_matrix = combinations_matrix
        self.distances_matrix = distances_matrix
        self.demands_array = demands_array
        self.vehicle_capacity = vehicle_capacity
        self.tare = tare
        self.alpha = alpha
        self.beta = beta
        self.q0 = q0
        self.np = np
        self.random = random

    def run(self):
        unvisited_nodes = self.np.array(self.nodes)
        routes_solution = []
        routes_energies = []

        while unvisited_nodes.size:
            route_solution = []
            route_energy = 0

            vehicle_capacity = 0
            vehicle_weight = self.tare

            '''
            # Elección del primer nodo como depot
            r = self.depot
            route_solution.append(self.depot)
            '''

            # Elección del primer nodo en un nodo distinto para cada hormiga
            route_solution.append(self.depot)
            if not routes_solution:
                r = self.start
            else:
                r = self.random.choice(unvisited_nodes)
            unvisited_nodes = unvisited_nodes[unvisited_nodes != r]
            route_solution.append(r)
            route_energy += self.distances_matrix[self.depot][r] * \
                vehicle_weight
            vehicle_weight += self.demands_array[r]
            vehicle_capacity += self.demands_array[r]

            unvisited_nodes_sorted = self.distances_matrix[r][unvisited_nodes].argsort(
            )
            valid_nodes = unvisited_nodes[unvisited_nodes_sorted][:int(
                len(self.nodes)/3)]

            while valid_nodes.size:
                combination = self.combinations_matrix[r][valid_nodes]
                # probabilities = self.np.divide(combination, combination.sum())

                q = self.np.random.random(1)[0]
                if q <= self.q0:
                    # s = valid_idx[probabilities.argmax()]
                    s = valid_nodes[combination.argmax()]
                else:
                    # s = self.np.random.choice(valid_idx, p = probabilities)
                    # s = self.random.choices(valid_idx, weights = probabilities, k = 1)[0]
                    cum_sum = self.np.cumsum(combination)
                    s = self.random.choices(
                        valid_nodes, cum_weights=cum_sum, k=1)[0]

                unvisited_nodes = unvisited_nodes[unvisited_nodes != s]
                route_solution.append(s)
                route_energy += self.distances_matrix[r][s] * vehicle_weight
                r = s
                vehicle_weight += self.demands_array[s]
                vehicle_capacity += self.demands_array[s]

                _demands_by_node = self.demands_array[unvisited_nodes] + \
                    vehicle_capacity
                valid_unvisited_nodes = unvisited_nodes[_demands_by_node <=
                                                        self.vehicle_capacity]
                valid_unvisited_nodes_sorted = self.distances_matrix[r][valid_unvisited_nodes].argsort(
                )
                valid_nodes = valid_unvisited_nodes[valid_unvisited_nodes_sorted][:int(
                    len(self.nodes)/3)]

            route_solution.append(self.depot)
            route_energy += self.distances_matrix[r][self.depot] * \
                vehicle_weight
            routes_solution.append(route_solution)
            routes_energies.append(route_energy)

        return routes_solution, routes_energies
