class FreeAntEMVRP_2:
    def __init__(self, depot, nodes, start, pheromones_matrix, distances_matrix, saving_matrix, cu_matrix, demands_array, vehicle_capacity,
                 tare, alpha, beta, q0):       
        import numpy as np
        import random
        
        self.depot = depot
        self.nodes = nodes
        self.start = start
        self.pheromones_matrix = pheromones_matrix
        self.distances_matrix = distances_matrix
        self.saving_matrix = saving_matrix
        self.cu_matrix = cu_matrix
        self.demands_array = demands_array
        self.vehicle_capacity = vehicle_capacity
        self.tare = tare             
        self.alpha = alpha
        self.beta = beta
        self.gamma = 3
        self._lambda = 3
        self.q0 = q0
        self.np = np
        self.random = random
 
    def run(self):
        nodes = self.np.array(self.nodes)
        unvisited_nodes = list(range(len(nodes)))
        routes_solution = []
        routes_energies = []

        # for k in range(self.k_number):
        while unvisited_nodes:
            route_solution = []
            route_energy = 0
            
            vehicle_capacity = 0
            vehicle_weight = self.tare
            
            '''
            # Elección del primer nodo como depot
            r = self.depot
            route_solution.append(self.depot)
            valid_idx = unvisited_nodes.copy()
            '''
            
            '''
            # Elección del primer nodo de manera aleatoria
            route_solution.append(self.depot)
            r = self.random.choice(unvisited_nodes)
            unvisited_nodes.remove(r)
            r = nodes[r]
            route_solution.append(r)
            route_energy += self.distances_matrix[self.depot][r] * vehicle_weight            
            vehicle_weight += self.demands_array[r]
            vehicle_capacity += self.demands_array[r]
            valid_idx = [i for i in unvisited_nodes if (vehicle_capacity + self.demands_array[nodes[i]] <= self.vehicle_capacity)]
            '''
            
            
            # Elección del primer nodo en un nodo distinto por cada hormiga
            route_solution.append(self.depot)
            if not routes_solution:
                r = self.start
            else:
                r = self.random.choice(unvisited_nodes)
            unvisited_nodes.remove(r)
            r = nodes[r]
            route_solution.append(r)
            route_energy += self.distances_matrix[self.depot][r] * vehicle_weight            
            vehicle_weight += self.demands_array[r]
            vehicle_capacity += self.demands_array[r]
            valid_idx = [i for i in unvisited_nodes if (vehicle_capacity + self.demands_array[nodes[i]] <= self.vehicle_capacity)]
            
            while valid_idx:             
                pheromones_trail = self.pheromones_matrix[r][nodes[valid_idx]]
                heuristic = self.np.power(1 / (self.distances_matrix[r][nodes[valid_idx]] * vehicle_weight), self.beta)
                savings = self.np.power(self.saving_matrix[r][nodes[valid_idx]], self.gamma)
                capacity_utilization = self.np.power((vehicle_capacity + self.demands_array[nodes[valid_idx]]) / self.vehicle_capacity,
                                                     self._lambda)
                combination = self.np.multiply(pheromones_trail, heuristic)
                combination = self.np.multiply(combination, savings)
                combination = self.np.multiply(combination, capacity_utilization)
                # probabilities = self.np.divide(combination, combination.sum())
                                
                q = self.np.random.random(1)[0]
                if q <= self.q0:
                    s = valid_idx[combination.argmax()]
                    # s = valid_idx[probabilities.argmax()]
                else:
                    cum_sum = self.np.cumsum(combination)
                    s = self.random.choices(valid_idx, cum_weights = cum_sum, k = 1)[0]
                    # s = self.random.choices(valid_idx, weights = probabilities, k = 1)[0]
                    
                unvisited_nodes.remove(s)
                s = nodes[s]
                route_solution.append(s)
                route_energy += self.distances_matrix[r][s] * vehicle_weight
                r = s
                vehicle_capacity += self.demands_array[s]
                vehicle_weight += self.demands_array[s]
                valid_idx = [i for i in unvisited_nodes if (vehicle_capacity + self.demands_array[nodes[i]] <= self.vehicle_capacity)]
            
            route_solution.append(self.depot)
            route_energy += self.distances_matrix[r][self.depot] * vehicle_weight
            routes_solution.append(route_solution)
            routes_energies.append(route_energy)            
        
        return routes_solution, routes_energies    