class RestrictedLocalGVNS:
    def __init__(self, depot, cluster, solution, solution_quality, distances_matrix, demands_array, tare, vehicle_capacity):
        import numpy as np
        import random as random
        from itertools import permutations
        from copy import deepcopy
        
        self.depot = depot
        self.cluster = cluster
        self.solution = solution
        self.solution_quality = solution_quality
        self.distances_matrix = distances_matrix
        self.demands_array = demands_array
        self.tare = tare
        self.vehicle_capacity = vehicle_capacity
        self.np = np
        self.random = random
        self.permutations = permutations
        self.deepcopy = deepcopy
        self.tabu_list = []
        
    def calculateSolutionQuality(self, solution): 
        route_energy = 0                                
            
        for pos, i in enumerate(solution):
            if pos == 0:
                vehicle_weight = self.tare
                before_node = i
            else:
                route_energy += self.distances_matrix[before_node][i] * vehicle_weight
                vehicle_weight += self.demands_array[i]
                before_node = i
                    
        return route_energy
    
    def singleRouteRelocate(self, best_sr_reloc_sol, best_sr_reloc_qual):     
        new_ = False
        if len(best_sr_reloc_sol) >= 4:
            temp_route = best_sr_reloc_sol.copy()
            idx_i, idx_j = self.random.sample(list(range(len(temp_route)))[1:-1], 2)
            temp_route.insert(idx_j, temp_route.pop(idx_i))
            
            if temp_route not in self.tabu_list:
                self.tabu_list.append(temp_route)
                temp_quality = self.calculateSolutionQuality(temp_route)

                if temp_quality < best_sr_reloc_qual:
                    best_sr_reloc_sol = temp_route
                    best_sr_reloc_qual = temp_quality
                
                new_ = True

        return best_sr_reloc_sol, best_sr_reloc_qual, new_
    
    def singleRouteSwap(self, best_sr_swap_sol, best_sr_swap_qual):             
        new_ = False
        if len(best_sr_swap_sol) >= 4:
            temp_route = best_sr_swap_sol.copy()            
            idx_i, idx_j = self.random.sample(list(range(len(temp_route)))[1:-1], 2)        
            temp_route[idx_i], temp_route[idx_j] = temp_route[idx_j], temp_route[idx_i] 

            if temp_route not in self.tabu_list:
                self.tabu_list.append(temp_route)
                temp_quality = self.calculateSolutionQuality(temp_route)

                if temp_quality < best_sr_swap_qual:
                    best_sr_swap_sol = temp_route
                    best_sr_swap_qual = temp_quality
                
                new_ = True

        return best_sr_swap_sol, best_sr_swap_qual, new_     
    
            
    def shakingSingleRouteSwap(self, best_sr_swap_sol, best_sr_swap_qual):
        if len(best_sr_swap_sol) >= 4:
            temp_route = best_sr_swap_sol.copy()
            idx_i, idx_j = self.random.sample(list(range(len(temp_route)))[1:-1], 2)        
            temp_route[idx_i], temp_route[idx_j] = temp_route[idx_j], temp_route[idx_i]                
                
            if temp_route not in self.tabu_list:
                self.tabu_list.append(temp_route)
                best_sr_swap_sol = temp_route
                    
        return best_sr_swap_sol     
    
    def shakingSingleRoute3Swap(self, best_sr_3swap_sol, best_sr_3swap_qual):
        if len(best_sr_3swap_sol) >= 5:
            temp_route = best_sr_3swap_sol.copy()
            idx_i, idx_j, idx_k = self.random.sample(list(range(len(temp_route)))[1:-1], 3)        
            temp_route[idx_i], temp_route[idx_j], temp_route[idx_k] = temp_route[idx_k], temp_route[idx_i], temp_route[idx_j]
                
            if temp_route not in self.tabu_list:
                self.tabu_list.append(temp_route)
                best_sr_3swap_sol = temp_route
                    
        return best_sr_3swap_sol
    
    def shakingSingleRoute2Exchange(self, best_sr_2exc_sol, best_sr_2exc_qual):            
        if (len(best_sr_2exc_sol) >= 7):
            temp_route = best_sr_2exc_sol.copy()
            idx_list = list(range(len(temp_route)))
            idx_n_i = self.random.choice(idx_list[1:-2])
            idx_list.pop(idx_n_i+1)            
            idx_list.pop(idx_n_i)
            idx_list.pop(idx_n_i-1)
            idx_n_j = self.random.choice(idx_list[1:-2])
            temp_route[idx_n_i], temp_route[idx_n_i+1], temp_route[idx_n_j], temp_route[idx_n_j+1] = temp_route[idx_n_j], temp_route[idx_n_j+1], temp_route[idx_n_i], temp_route[idx_n_i+1]
                
            if temp_route not in self.tabu_list:
                self.tabu_list.append(temp_route)
                best_sr_2exc_sol = temp_route
                
        return best_sr_2exc_sol
        
    
    def shaking(self, original_solution, original_quality, N):        
        if (N == 0):                     return self.shakingSingleRouteSwap(original_solution, original_quality)
        elif (N == 1):                   return self.shakingSingleRoute3Swap(original_solution, original_quality)
        else:                            return self.shakingSingleRoute2Exchange(original_solution, original_quality)

    def VNS(self, shaking_solution, best_global_quality, N):
        best_vns_solution = shaking_solution.copy()
        best_vns_quality = best_global_quality        
        
        for n in range(N):
            best_vns_solution, best_vns_quality, new_ = self.singleRouteSwap(best_vns_solution, best_vns_quality)            
            best_vns_solution, best_vns_quality, new_ = self.singleRouteRelocate(best_vns_solution, best_vns_quality)

        '''
        n = 0
        while n <= int(N * 0.75):
            best_vns_solution, best_vns_quality, new_ = self.singleRouteSwap(best_vns_solution, best_vns_quality)
            if new_ == True: 
                n += 1
                new_ = False        
             
        n = 0
        while n <= int(N * 0.50):
            best_vns_solution, best_vns_quality, new_ = self.singleRouteRelocate(best_vns_solution, best_vns_quality)
            if new_ == True: 
                n += 1
                new_ = False 
        '''        
        
        return best_vns_solution, best_vns_quality
            
    def improve(self):        
        from math import factorial
        import time
              
        best_global_solution = self.deepcopy(self.solution)
        best_global_quality = self.deepcopy(self.solution_quality)        
        new_improve = False
        iteration = 0
        max_iterations = len(self.demands_array[self.cluster])
        max_vns_search = int(len(self.demands_array[self.cluster]) * 1.5)
        max_time = len(self.demands_array[self.cluster]) * 0.00075
        n = len(self.cluster)
        r = n
        max_possibles_perms = int(factorial(n)/factorial(n - r))
        # print('Número máximo de perms: ' + str(max_possibles_perms))
        start_time = time.time()
        # print(str(time.time() - start_time) + ': ' + str(best_global_quality))
        
        while iteration <= max_iterations:
            for shaking_ratio in range(3):
                shaking_solution = self.shaking(best_global_solution.copy(), best_global_quality, shaking_ratio)
                vns_solution, vns_quality = self.VNS(shaking_solution, best_global_quality, max_vns_search)     

                if vns_quality < best_global_quality:
                    # print(str(time.time() - start_time) + ': ' + str(vns_quality))
                    best_global_solution = vns_solution
                    best_global_quality = vns_quality
                    new_improve = True
            
            if new_improve == True:
                iteration = 0
                new_improve = False 
            else:
                if (time.time() - start_time) >= max_time: break
                iteration += 1 
                                   
            # print('Tamaño de la lista tabu: ' + str(len(self.tabu_list)))
        return best_global_solution, best_global_quality