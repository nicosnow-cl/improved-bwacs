class KMeansLocalSearch:
    def __init__(self, depot, nodes, clusters, centroids_list, clusters_total_cost, demands_array, coords_matrix, vehicle_capacity,
                 k_number, clusters_demands, clusters_costs, unassigned_nodes, metric):
        import numpy as np
        import random
        
        self.depot = depot
        self.nodes = nodes
        self.clusters = clusters
        self.centroids_list = centroids_list
        self.clusters_total_cost = clusters_total_cost
        self.demands_array = demands_array
        self.coords_matrix = coords_matrix
        self.vehicle_capacity = vehicle_capacity
        self.k_number = k_number
        self.clusters_demands = clusters_demands
        self.clusters_costs = clusters_costs
        self.unassigned_nodes = unassigned_nodes
        self.metric = metric
        self.distances_to_centroids_matrix = None
        self.np = np
        self.random = random
    
    def calculateClustersDemands(self, clusters):        
        return self.np.array([self.demands_array[cluster].sum() for cluster in clusters]) 
    
    def recalculateCentroids(self, clusters):
        return self.np.array([self.coords_matrix[cluster].mean(axis = 0) for cluster in clusters])
       
    def calculateClustersEnergies(self, centroids_list, clusters):
        ord_ = 1 if self.metric == 'manhattan' else 2
        clusters_costs = []
        for k, c in enumerate(centroids_list):
            cluster_distances = [self.np.linalg.norm((self.coords_matrix[i] - c), ord = ord_) for i in clusters[k]]      
            cluster_energies = self.np.array(cluster_distances) * self.demands_array[clusters[k]]
            clusters_costs.append(cluster_energies.sum())
        return self.np.array(clusters_costs)
    
    def clusterRelocate(self, original_solution):
        feasible_solution = original_solution.copy()
        is_feasible = False
        
        while not is_feasible:
            idx_clusters = self.random.sample(range(len(original_solution)), 2)
            _route_a, _route_b = original_solution[idx_clusters[0]].copy(), original_solution[idx_clusters[1]].copy()            
            idx_n_i = self.random.choice(range(len(_route_a)))
            
            _route_b.insert(0, original_solution[idx_clusters[0]][idx_n_i])
            _route_a.pop(idx_n_i)
            
            route_demand = self.calculateClustersDemands([_route_b])
            
            if (route_demand <= self.vehicle_capacity): 
                feasible_solution[idx_clusters[0]], feasible_solution[idx_clusters[1]] = _route_a, _route_b
                is_feasible = True                

        return feasible_solution
    
    def clustersSwap(self, original_solution):
        feasible_solution = original_solution.copy()
        is_feasible = False
        
        while not is_feasible:
            idx_clusters = self.random.sample(range(len(original_solution)), 2)
            _cluster_a, _cluster_b = original_solution[idx_clusters[0]].copy(), original_solution[idx_clusters[1]].copy()
            
            idx_n_i = self.random.choice(range(len(_cluster_a)))
            idx_n_j = self.random.choice(range(len(_cluster_b)))            
            
            _cluster_a[idx_n_i], _cluster_b[idx_n_j] = _cluster_b[idx_n_j], _cluster_a[idx_n_i]
                        
            clusters_demands = self.calculateClustersDemands([_cluster_a, _cluster_b])
            
            if (clusters_demands[0] <= self.vehicle_capacity) and (clusters_demands[1] <= self.vehicle_capacity): 
                feasible_solution[idx_clusters[0]], feasible_solution[idx_clusters[1]] = _cluster_a, _cluster_b
                is_feasible = True            
        
        return feasible_solution    
    
    def clusters3Swap(self, org_sol):
        feasible_solution = org_sol.copy()
        is_feasible = False
        
        while not is_feasible:
            idx_clusters = self.random.sample(range(len(org_sol)), 3) 
            _c_a, _c_b, _c_c = org_sol[idx_clusters[0]].copy(), org_sol[idx_clusters[1]].copy(), org_sol[idx_clusters[2]].copy()
            
            idx_n_i = self.random.choice(range(len(_c_a)))
            idx_n_j = self.random.choice(range(len(_c_b))) 
            idx_n_k = self.random.choice(range(len(_c_c)))
            
            _c_a[idx_n_i], _c_b[idx_n_j], _c_c[idx_n_k] = _c_c[idx_n_k], _c_a[idx_n_i], _c_b[idx_n_j]
            
            clusters_demands = self.calculateClustersDemands([_c_a, _c_b, _c_c])
            
            if ((clusters_demands[0] <= self.vehicle_capacity) and (clusters_demands[1] <= self.vehicle_capacity) 
                and (clusters_demands[2] <= self.vehicle_capacity)):
                feasible_solution[idx_clusters[0]] = _c_a
                feasible_solution[idx_clusters[1]] = _c_b
                feasible_solution[idx_clusters[2]] = _c_c
                is_feasible = True
            
        return feasible_solution
            
    def clustersExchange(self, org_sol):
        feasible_solution = org_sol.copy()
        is_feasible = False
        
        while not is_feasible:
            idx_clusters = self.random.sample(range(len(org_sol)), 2)
            _c_a, _c_b = org_sol[idx_clusters[0]].copy(), org_sol[idx_clusters[1]].copy()

            idx_a = self.random.choice(list(range(len(_c_a)))[0:-2])
            idx_b = self.random.choice(list(range(len(_c_b)))[0:-2])

            _c_a[idx_a], _c_a[idx_a+1], _c_b[idx_b], _c_b[idx_b+1] = _c_b[idx_b], _c_b[idx_b+1], _c_a[idx_a], _c_a[idx_a+1]
            
            clusters_demands = self.calculateClustersDemands([_c_a, _c_b])
            
            if (clusters_demands[0] <= self.vehicle_capacity) and (clusters_demands[1] <= self.vehicle_capacity):
                feasible_solution[idx_clusters[0]], feasible_solution[idx_clusters[1]] = _c_a, _c_b
                is_feasible = True
        
        return feasible_solution
    
    def clusters3Exchange(self, org_sol):
        feasible_solution = org_sol.copy()
        is_feasible = False
        
        while not is_feasible:
            idx_clusters = self.random.sample(range(len(org_sol)), 3)
            c_a, c_b, c_c = org_sol[idx_clusters[0]].copy(), org_sol[idx_clusters[1]].copy(), org_sol[idx_clusters[2]].copy()
            
            i_a = self.random.choice(list(range(len(c_a)))[0:-2])
            i_b = self.random.choice(list(range(len(c_b)))[0:-2])
            i_c = self.random.choice(list(range(len(c_c)))[0:-2])
            
            c_a[i_a], c_a[i_a+1], c_b[i_b], c_b[i_b+1], c_c[i_c], c_c[i_c+1] =  c_c[i_c], c_c[i_c+1], c_a[i_a], c_a[i_a+1], c_b[i_b], c_b[i_b+1]
            clusters_demands = self.calculateClustersDemands([c_a, c_b, c_c])
            
            if ((clusters_demands[0] <= self.vehicle_capacity) and (clusters_demands[1] <= self.vehicle_capacity) 
                and (clusters_demands[2] <= self.vehicle_capacity)):
                feasible_solution[idx_clusters[0]] = c_a
                feasible_solution[idx_clusters[1]] = c_b
                feasible_solution[idx_clusters[2]] = c_c
                is_feasible = True
        
        return feasible_solution
    
    def shaking(self, original_solution, N):
        shaking_solution = original_solution.copy()
        
        if (N == 0):        shaking_solution = self.clustersSwap(shaking_solution) 
        elif (N == 1):      shaking_solution = self.clusters3Swap(shaking_solution)
        elif (N == 2):      shaking_solution = self.clustersExchange(shaking_solution)
        else:               shaking_solution = self.clusters3Exchange(shaking_solution)                         

        return shaking_solution
    
    def VNS(self, original_clusters, clusters_quality, N):
        from copy import deepcopy
        
        best_clusters = original_clusters.copy()
        best_centroids = self.recalculateCentroids(best_clusters)
        best_clusters_quality = clusters_quality.copy()
        
        for n in range(N):       
            # clusterRelocate()
            temp_clusters = self.clusterRelocate(best_clusters)
            temp_centroids = self.recalculateCentroids(temp_clusters)
            temp_clusters_quality = self.calculateClustersEnergies(temp_centroids, temp_clusters)
            
            if temp_clusters_quality.sum() < best_clusters_quality.sum():
                best_clusters = deepcopy(temp_clusters)
                best_centroids = deepcopy(temp_centroids)
                best_clusters_quality = deepcopy(temp_clusters_quality)
                return best_clusters, best_centroids, best_clusters_quality
            
            # clustersSwap()
            temp_clusters = self.clustersSwap(best_clusters)
            temp_centroids = self.recalculateCentroids(temp_clusters)
            temp_clusters_quality = self.calculateClustersEnergies(temp_centroids, temp_clusters)
            
            if temp_clusters_quality.sum() < best_clusters_quality.sum():
                best_clusters = deepcopy(temp_clusters)
                best_centroids = deepcopy(temp_centroids)
                best_clusters_quality = deepcopy(temp_clusters_quality)
                return best_clusters, best_centroids, best_clusters_quality          
            
            # clusters3Swap()
            temp_clusters = self.clusters3Swap(best_clusters)
            temp_centroids = self.recalculateCentroids(temp_clusters)
            temp_clusters_quality = self.calculateClustersEnergies(temp_centroids, temp_clusters)
            
            if temp_clusters_quality.sum() < best_clusters_quality.sum():
                best_clusters = deepcopy(temp_clusters)
                best_centroids = deepcopy(temp_centroids)
                best_clusters_quality = deepcopy(temp_clusters_quality)
                return best_clusters, best_centroids, best_clusters_quality
            
            # clustersExchange()
            temp_clusters = self.clustersExchange(best_clusters)
            temp_centroids = self.recalculateCentroids(temp_clusters)
            temp_clusters_quality = self.calculateClustersEnergies(temp_centroids, temp_clusters)
            
            if temp_clusters_quality.sum() < best_clusters_quality.sum():
                best_clusters = deepcopy(temp_clusters)
                best_centroids = deepcopy(temp_centroids)
                best_clusters_quality = deepcopy(temp_clusters_quality)
                return best_clusters, best_centroids, best_clusters_quality
            
            # clusters3Exchange()
            temp_clusters = self.clusters3Exchange(best_clusters)
            temp_centroids = self.recalculateCentroids(temp_clusters)
            temp_clusters_quality = self.calculateClustersEnergies(temp_centroids, temp_clusters)
            
            if temp_clusters_quality.sum() < best_clusters_quality.sum():
                best_clusters = deepcopy(temp_clusters)
                best_centroids = deepcopy(temp_centroids)
                best_clusters_quality = deepcopy(temp_clusters_quality)
                return best_clusters, best_centroids, best_clusters_quality 
               
        return best_clusters, best_centroids, best_clusters_quality
    
    def improve(self):        
        import time
        from copy import deepcopy
        start_time = time.time()
        
        best_clusters = deepcopy(self.clusters)
        best_centroids = self.recalculateCentroids(self.clusters)
        best_quality = deepcopy(self.clusters_costs)
        new_improve = False
        i = 0
        
        while i < (len(self.nodes)/2):
            for shaking_ratio in range(4):
                new_clusters = self.shaking(best_clusters, shaking_ratio)
                new_clusters, new_centroids, new_quality = self.VNS(new_clusters, best_quality, 15)      

                if new_quality.sum() < best_quality.sum():
                    print(str(time.time() - start_time) + ': ' + str(new_quality.sum()))
                    best_clusters = deepcopy(new_clusters)
                    best_centroids = deepcopy(new_centroids)
                    best_quality = deepcopy(new_quality)
                    new_improve = True
            
            if new_improve:
                i = 1
                new_improve = False
            else:
                i += 1

        return best_clusters, best_quality, best_centroids                