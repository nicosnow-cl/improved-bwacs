class KMedoidsEMVRP:
    def __init__(self, depot, nodes, demands_array, distances_matrix, vehicle_capacity, k_number):
        import numpy as np
        import math
        
        self.clusters = []
        self.medoids_list = []
        self.depot = depot
        self.nodes = nodes
        self.demands_array = demands_array
        self.distances_matrix = distances_matrix
        self.vehicle_capacity = vehicle_capacity    
        self.k_number = k_number
        self.np = np
        self.math = math
    
    def setInitialMedoids(self):        
        self.medoids_list.append((self.distances_matrix[self.depot, :] * self.demands_array).argmax())
        
        for k in range(self.k_number - 1):
            _priority = self.distances_matrix[:, [self.depot] + self.medoids_list]
            _priority = _priority.mean(axis = 1)         
            _priority[self.medoids_list] = 0
            _priority *= self.demands_array
            self.medoids_list.append(_priority.argmax())
        
        self.nodes.remove(self.depot)        
        self.nodes = self.np.array(self.nodes) 
        self.nodes = self.nodes - 1
        self.demands_array = self.demands_array[1:]
        self.distances_matrix = self.np.delete(self.distances_matrix, 0, 0)
        self.distances_matrix = self.np.delete(self.distances_matrix, 0, 1)
        
    def calculateTotalCosts(self):
        total_cost = 0
        
        for k, m in enumerate(self.medoids_list):
            cluster_energy = self.distances_matrix[m][self.clusters[k]] * self.demands_array[m]
            total_cost += cluster_energy.sum()
            
        return total_cost
        # return self.np.sum([self.distances_matrix[(m, self.clusters[k])].sum() for k, m in enumerate(self.medoids_list)])
    
    def reassignUnassignedNodes(self, unassigned_nodes):
        for node in unassigned_nodes:            
            k = self.np.random.choice(list(range(self.k_number)))    
            self.clusters[k] = self.np.append(self.clusters[k], node)
    
    def unassignedNodesAsMedoids(self, unassigned_nodes):
        for node in unassigned_nodes:
            costs_sub_matrix = self.distances_matrix[node, self.medoids_list] * self.demands_array[node]
            k = costs_sub_matrix.argsort()[0]
            self.medoids_list[k] = node
            
    def run(self):
        import time
        from copy import deepcopy
        start_time = time.time()
        
        if self.distances_matrix.shape[0] < 25:
            limit_constraint = 2
        elif self.distances_matrix.shape[0] >= 25 and self.distances_matrix.shape[0] < 50:
            limit_constraint = 2
        elif self.distances_matrix.shape[0] >= 50 and self.distances_matrix.shape[0] < 75:
            limit_constraint = 2
        elif self.distances_matrix.shape[0] >= 75 and self.distances_matrix.shape[0] < 100:
            limit_constraint = 1.5
        elif self.distances_matrix.shape[0] >= 100 and self.distances_matrix.shape[0] < 150:
            limit_constraint = 1
        elif self.distances_matrix.shape[0] >= 150 and self.distances_matrix.shape[0] < 250:
            limit_constraint = 0.8
        elif self.distances_matrix.shape[0] >= 250 and self.distances_matrix.shape[0] < 500:
            limit_constraint = 0.4
        elif self.distances_matrix.shape[0] >= 500 and self.distances_matrix.shape[0] < 1000:
            limit_constraint = 0.1
        else:
            limit_constraint = 0.05
        
        self.setInitialMedoids()   
        self.medoids_list = [idx - 1 for idx in self.medoids_list]
        
        iteration = 1
        max_iterations = self.math.ceil(len(self.nodes) * limit_constraint)
        best_clusters = None
        best_total_cost = self.np.inf
        best_unassigned_nodes = [[] for k in range(self.k_number)]
        best_constraint_clusters = None
        best_constraint_total_cost = self.np.inf
        best_constraint_unassigned_nodes = [[] for k in range(self.k_number)] 
        
        while iteration <= max_iterations:
            print('    > ITERACIÓN ' + str(iteration))
            unassigned_nodes = self.nodes.copy()
            old_medoids_list = self.medoids_list.copy()
            print('        - Medoides actuales: ' + str(old_medoids_list))
            
            costs_sub_matrix = self.distances_matrix[:, self.medoids_list]
            for i in self.nodes:
                costs_sub_matrix[i] *= self.demands_array[i]
                
            J = self.np.zeros((self.nodes.shape[0]), dtype = int)
            clusters_capacity = self.np.full((self.k_number), self.vehicle_capacity, dtype = int)

            for i, k in enumerate(self.medoids_list):
                J[k] = i
                clusters_capacity[i] -= self.demands_array[k]
                unassigned_nodes = unassigned_nodes[unassigned_nodes != k]
            
            for i in unassigned_nodes:                   
                valid_idx = [k for k in range(self.k_number) if (clusters_capacity[k] - self.demands_array[i]) >= 0]

                if valid_idx:
                    costs = costs_sub_matrix[i, :]
                    j = valid_idx[costs[valid_idx].argmin()]                                    
                    J[i] = j
                    clusters_capacity[j] -= self.demands_array[i]
                    unassigned_nodes = unassigned_nodes[unassigned_nodes != i]                  

            self.clusters = [self.np.where(J == k)[0] for k in range(self.k_number)]
            print('        - Nodos sin asignar: ' + str(unassigned_nodes) + ', ' + str(self.demands_array[unassigned_nodes]))
            print('        - Mejor costo anterior: ' + str(best_total_cost))
            
            new_total_cost = self.calculateTotalCosts()
            print('        - Costo total actual: ' + str(new_total_cost) + '\n')
            
            if (len(unassigned_nodes) < len(best_unassigned_nodes)) or (new_total_cost < best_total_cost):                
                best_clusters = deepcopy(self.clusters)                
                best_clusters = [best_clusters[k] + 1 for k in range(self.k_number)]
                best_total_cost = new_total_cost
                best_medoids_list = deepcopy(self.medoids_list)
                best_medoids_list = [medoid + 1 for medoid in best_medoids_list]
                best_unassigned_nodes = deepcopy(unassigned_nodes)
                best_unassigned_nodes += 1
                
                if (not unassigned_nodes.size):
                    best_constraint_clusters = deepcopy(best_clusters)
                    best_constraint_total_cost = best_total_cost
                    best_contraint_medoids_list = deepcopy(best_medoids_list)
                    best_constraint_unassigned_nodes = deepcopy(best_unassigned_nodes)
                else:
                    self.reassignUnassignedNodes(unassigned_nodes)
            
            for k in range(self.k_number):
                _ = self.np.ix_(self.clusters[k], self.clusters[k])                
                h = self.distances_matrix[_]      
                
                for pos, i in enumerate(self.clusters[k]):
                    h[pos] *= self.demands_array[i]
                
                J = self.np.mean(h, axis = 1)
                j = self.np.argmin(J)
                self.medoids_list[k] = self.clusters[k][j]
            
            if (self.np.array_equal(self.np.sort(old_medoids_list), self.np.sort(self.medoids_list))):
                break
            
            self.unassignedNodesAsMedoids(unassigned_nodes)
            
            iteration += 1
        
        print('    > Mejor Clusterización: ' + str(best_clusters) + ', con costo final: ' + str(best_total_cost) 
              + ', nodos sin asignar: ' + str(best_unassigned_nodes))
        print('    > Mejor Clusterización (Contraint): ' + str(best_constraint_clusters) + ', con costo final: ' 
              + str(best_constraint_total_cost) + ', nodos sin asignar: ' + str(best_constraint_unassigned_nodes))
        
        print('    --- %s seconds ---\n' % (time.time() - start_time))
        
        if best_constraint_clusters != None:
            return [list(best_constraint_clusters[k]) for k in range(self.k_number)], best_constraint_total_cost,        best_contraint_medoids_list, best_constraint_unassigned_nodes
        else:
            return [list(best_clusters[k]) for k in range(self.k_number)], best_total_cost, best_medoids_list, best_unassigned_nodes       