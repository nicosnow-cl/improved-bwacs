class BWACSClusterLS:
    def __init__(self, depot, cluster, cluster_quality, distances_matrix, demands_array, tare, vehicle_capacity, k_number):
        import numpy as np
        import random
        
        self.depot = depot
        self.cluster = cluster
        self.cluster_quality = cluster_quality
        self.distances_matrix = distances_matrix
        self.demands_array = demands_array
        self.tare = tare
        self.vehicle_capacity = vehicle_capacity
        self.k_number = k_number
        self.np = np
        self.random = random
        self.arcs_matrix = []
        
    def calculateClusterCost(self, cluster):       
        for pos, i in enumerate(cluster):
            if pos == 0:
                cluster_energy = 0
                vehicle_weight = self.tare
                before_node = self.depot
            else:
                cluster_energy += self.distances_matrix[before_node][i] * vehicle_weight
                vehicle_weight += self.demands_array[i]
                before_node = i
 
        return cluster_energy
    
    def createArcsMatrix(self, cluster):
        arcs_matrix = self.np.zeros((len(self.cluster)-1, len(self.cluster)-1))
        
        for pos, i in enumerate(cluster):
            if pos == 0:
                before_node = self.depot
            else:
                arcs_matrix[self.cluster.index(before_node)][self.cluster.index(i)] = 1
                before_node = i
        
        return arcs_matrix
    
    def improve(self):
        from itertools import permutations
        from copy import deepcopy
        import math
        
        complexity = math.factorial(len(self.cluster[1:-1])) 
        complexity /= len(self.cluster)
        
        '''
            ###############################################################
            ##  0. Dependiendo del problema, definimos una variable para ##
            ##  controlar el número máximo de intentos.                  ##
            ###############################################################
        '''        
        if complexity < 1000: limit_constraint = 1            
        elif complexity >= 1000 and complexity < 4999: limit_constraint = 0.6            
        elif complexity >= 5000 and complexity < 9999: limit_constraint = 0.1            
        elif complexity >= 10000 and complexity < 99999: limit_constraint = 0.05            
        elif complexity >= 100000 and complexity < 999999: limit_constraint = 0.005            
        elif complexity >= 1000000 and complexity < 9999999: limit_constraint = 0.0005            
        elif complexity >= 10000000 and complexity < 99999999: limit_constraint = 0.00005            
        else: limit_constraint = 0.000005            
                
        # self.cluster_quality = self.calculateClusterCost(self.cluster)
            
        '''
            ###############################################################
            ##  1. 2-Opt.                                                 ##
            ##                                                           ##
            ###############################################################
        ''' 
        
        indexs_cluster = list(range(len(self.cluster)))        
        tabu_list = []
        iteration = 1
        max_iterations = complexity * limit_constraint

        while iteration <= max_iterations:
            temp_cluster = deepcopy(self.cluster)
            pair = self.random.sample(indexs_cluster[1:-1], 2)
            nodes = tuple([self.cluster[pair[0]], self.cluster[pair[1]]])
            
            if nodes not in tabu_list:
                tabu_list.append(nodes)
                temp_cluster[pair[0]], temp_cluster[pair[1]] = self.cluster[pair[1]], self.cluster[pair[0]]
                temp_cluster_quality = self.calculateClusterCost(temp_cluster)
                
                if temp_cluster_quality < self.cluster_quality:
                    self.cluster = deepcopy(temp_cluster)
                    self.cluster_quality = temp_cluster_quality                    
                
            iteration += 1
                  
        self.arcs_matrix = self.createArcsMatrix(self.cluster)
        
        return self.cluster, self.cluster_quality, self.arcs_matrix
            
            