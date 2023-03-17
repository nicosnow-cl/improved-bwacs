class RestrictedAntEMVRP:
    def __init__(self, depot, cluster, distances_matrix, heuristic_matrix, demands_array, vehicle_capacity, tare, pheromones_matrix, alpha,
                 beta, q0, combinatorial_matrix):     
        import numpy as np
        import random
        
        self.depot = depot
        self.cluster = cluster        
        self.distances_matrix = distances_matrix
        self.demands_array = demands_array
        self.vehicle_capacity = vehicle_capacity
        self.tare = tare
        self.pheromones_matrix = pheromones_matrix
        self.heuristic_matrix = heuristic_matrix
        self.combinatorial_matrix = combinatorial_matrix
        self.alpha = alpha
        self.beta = beta
        self.q0 = q0     
        self.np = np
        self.random = random
        
    def run(self):
        _cluster = self.np.array([self.depot] + self.cluster)
        unvisited_nodes = self.np.array(list(range(len(_cluster))))
        
        route = []
        route_energy = 0
        route_distance = 0
        
        '''
        # Elección del primer nodo como depot
        r = self.depot
        route.append(r)
        vehicle_weight = self.tare         
        unvisited_nodes = unvisited_nodes[unvisited_nodes != r]
        '''
        
        
        # Elección del primer nodo de manera aleatoria
        vehicle_weight = self.tare 
        unvisited_nodes = unvisited_nodes[unvisited_nodes != self.depot]
        route.append(_cluster[self.depot])
        r = self.np.random.choice(unvisited_nodes)
        route.append(_cluster[r])
        unvisited_nodes = unvisited_nodes[unvisited_nodes != r]
        route_energy += self.distances_matrix[_cluster[self.depot]][_cluster[r]] * vehicle_weight
        vehicle_weight += self.demands_array[_cluster[r]]
        
        
        while unvisited_nodes.size:
            # heuristic = self.heuristic_matrix[_cluster[r]][_cluster[unvisited_nodes]]
            # heuristic = self.np.power((1 / (self.distances_matrix[_cluster[r]][_cluster[unvisited_nodes]] * vehicle_weight)), self.beta)
            # pheromone_trail = self.pheromones_matrix[r][unvisited_nodes]
            # combination = self.np.multiply(pheromone_trail, heuristic)
            combination = self.combinatorial_matrix[r][unvisited_nodes]
            probabilities = self.np.divide(combination, combination.sum())
            q = self.np.random.random(1)[0]

            if q <= self.q0:
                s = unvisited_nodes[probabilities.argmax()]
            else:                
                # s = self.np.random.choice(unvisited_nodes, 1, p = probabilities)[0]
                s = self.random.choices(unvisited_nodes, weights = probabilities, k = 1)[0]
    
            route.append(_cluster[s])            
            route_energy += self.distances_matrix[_cluster[r]][_cluster[s]] * vehicle_weight
            # route_distance += self.distances_matrix[_cluster[r]][_cluster[s]]
            vehicle_weight += self.demands_array[_cluster[s]]            
            
            unvisited_nodes = unvisited_nodes[unvisited_nodes != s]
            r = s
                
        route.append(self.depot)          
        route_energy += self.distances_matrix[_cluster[r]][_cluster[self.depot]] * vehicle_weight
        # route_distance += self.distances_matrix[_cluster[r]][_cluster[self.depot]]        
        
        return route, route_energy, route_distance