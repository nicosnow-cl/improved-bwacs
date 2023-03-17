class FreeBWACSGVN:
    def __init__(self, depot, solution, solution_quality, distances_matrix, demands_array, tare, vehicle_capacity, k_number, iteration,
                 max_iterations):
        import numpy as np
        import random as random
        from copy import deepcopy
        
        self.depot = depot
        self.solution = solution
        self.solution_quality = solution_quality
        self.distances_matrix = distances_matrix
        self.demands_array = demands_array
        self.tare = tare
        self.vehicle_capacity = vehicle_capacity
        self.k_number = k_number
        self.solution_demands = []
        self.iteration = iteration
        self.max_iterations = max_iterations
        self.intensity_percentage = 0
        self.np = np
        self.random = random
        self.deepcopy = deepcopy
    
    def calculateRoutesDemands(self, routes):
        return self.np.array([self.demands_array[route].sum() for route in routes]) 
    
    def calculateSolutionQuality(self, solution):       
        routes_energies = self.np.zeros(len(solution))
        
        for k, route in enumerate(solution):
            route_energy = 0 
            
            for pos, i in enumerate(route):
                if pos == 0:
                    vehicle_weight = self.tare
                    before_node = i
                else:
                    route_energy += self.distances_matrix[before_node][i] * vehicle_weight
                    vehicle_weight += self.demands_array[i]
                    before_node = i

            routes_energies[k] = route_energy      
            
        return routes_energies
    
    def singleRouteRelocate(self, best_sr_reloc_sol, best_sr_reloc_qual, idx_route = None, tabu_list = []):     
        if (idx_route == None): idx_route = self.random.choice(range(len(best_sr_reloc_sol)))
        
        if len(best_sr_reloc_sol[idx_route]) >= 4:
            temp_route = best_sr_reloc_sol[idx_route].copy()
            idx_i, idx_j = self.random.sample(list(range(len(temp_route)))[1:-1], 2)
            temp_route.insert(idx_j, temp_route.pop(idx_i))
            
            if temp_route not in tabu_list:
                tabu_list.append(temp_route)
                temp_quality = self.calculateSolutionQuality([temp_route])

                if temp_quality < best_sr_reloc_qual[idx_route]:
                    best_sr_reloc_sol[idx_route] = temp_route
                    best_sr_reloc_qual[idx_route] = temp_quality
                    
        return best_sr_reloc_sol, best_sr_reloc_qual, tabu_list
    
    def singleRouteSwap(self, best_sr_swap_sol, best_sr_swap_qual, idx_route = None, tabu_list = []):             
        if (idx_route == None): idx_route = self.random.choice(range(len(best_sr_swap_sol)))
        
        if len(best_sr_swap_sol[idx_route]) >= 4:
            temp_route = best_sr_swap_sol[idx_route].copy()
            idx_i, idx_j = self.random.sample(list(range(len(temp_route)))[1:-1], 2)        
            temp_route[idx_i], temp_route[idx_j] = temp_route[idx_j], temp_route[idx_i] 
            
            if temp_route not in tabu_list:
                tabu_list.append(temp_route)
                temp_quality = self.calculateSolutionQuality([temp_route])

                if temp_quality < best_sr_swap_qual[idx_route]:
                    best_sr_swap_sol[idx_route] = temp_route
                    best_sr_swap_qual[idx_route] = temp_quality
            
        return best_sr_swap_sol, best_sr_swap_qual, tabu_list    
    
    def multiRouteSwap(self, best_mr_swap_sol, best_mr_swap_qual, idx_r_i = None, idx_r_j = None):
        if (idx_r_i == None) or (idx_r_j == None): range_solution = range(len(best_mr_swap_sol))
        tabu_list = []
        is_feasible = False
        n = 1
        
        max_n = (len(self.demands_array) - 1) / 2 
        
        while (not is_feasible) and (n < max_n):
            
            if (idx_r_i == None) or (idx_r_j == None): idx_r_i, idx_r_j = self.random.sample(range_solution, 2)
            
            if (len(best_mr_swap_sol[idx_r_i]) >= 3) and (len(best_mr_swap_sol[idx_r_j]) >= 3):
                temp_route_a, temp_route_b = best_mr_swap_sol[idx_r_i].copy(), best_mr_swap_sol[idx_r_j].copy()
                idx_n_i = self.random.choice(list(range(len(temp_route_a)))[1:-1])
                idx_n_j = self.random.choice(list(range(len(temp_route_b)))[1:-1])
                temp_route_a[idx_n_i], temp_route_b[idx_n_j] = temp_route_b[idx_n_j], temp_route_a[idx_n_i]
                
                if (temp_route_a not in tabu_list) or (temp_route_b not in tabu_list):                
                    tabu_list.append(temp_route_a)
                    tabu_list.append(temp_route_b)
                    routes_demands = self.calculateRoutesDemands([temp_route_a, temp_route_b])

                    if (routes_demands[0] <= self.vehicle_capacity) and (routes_demands[1] <= self.vehicle_capacity): 
                        original_quality = self.calculateSolutionQuality([best_mr_swap_sol[idx_r_i], best_mr_swap_sol[idx_r_j]])
                        new_quality = self.calculateSolutionQuality([temp_route_a, temp_route_b])

                        if new_quality.sum() < original_quality.sum():
                            best_mr_swap_sol[idx_r_i], best_mr_swap_sol[idx_r_j] = temp_route_a, temp_route_b
                            best_mr_swap_qual[idx_r_i], best_mr_swap_qual[idx_r_j] = new_quality[0], new_quality[1]
                            is_feasible = True
            n += 1
            
        return best_mr_swap_sol, best_mr_swap_qual     
            
    def shakingMultiRouteSwap(self, best_mr_swap_sol, best_mr_swap_qual):
        range_solution = range(len(best_mr_swap_sol))
        idx_routes = []
        tabu_list = []
        is_feasible = False
        n = 1
        
        max_n = (len(self.demands_array) - 1) / 2
        
        while (not is_feasible) and (n < max_n):
            idx_routes = self.random.sample(range_solution, 2)            
            
            if (len(best_mr_swap_sol[idx_routes[0]]) >= 3) and (len(best_mr_swap_sol[idx_routes[1]]) >= 3):
                min_quality = ((best_mr_swap_qual[idx_routes[0]] + best_mr_swap_qual[idx_routes[1]]) 
                           + ((best_mr_swap_qual[idx_routes[0]] + best_mr_swap_qual[idx_routes[1]]) * 0.2))
                
                temp_route_a, temp_route_b = best_mr_swap_sol[idx_routes[0]].copy(), best_mr_swap_sol[idx_routes[1]].copy()
                idx_n_i = self.random.choice(list(range(len(temp_route_a)))[1:-1])
                idx_n_j = self.random.choice(list(range(len(temp_route_b)))[1:-1])
                temp_route_a[idx_n_i], temp_route_b[idx_n_j] = temp_route_b[idx_n_j], temp_route_a[idx_n_i]
                
                if (temp_route_a not in tabu_list) or (temp_route_b not in tabu_list):                
                    tabu_list.append(temp_route_a)
                    tabu_list.append(temp_route_b)
                    routes_demands = self.calculateRoutesDemands([temp_route_a, temp_route_b])

                    if (routes_demands[0] <= self.vehicle_capacity) and (routes_demands[1] <= self.vehicle_capacity): 
                        new_quality = self.calculateSolutionQuality([temp_route_a, temp_route_b])
                        
                        if new_quality.sum() < min_quality:
                            best_mr_swap_sol[idx_routes[0]], best_mr_swap_sol[idx_routes[1]] = temp_route_a, temp_route_b
                            best_mr_swap_qual[idx_routes[0]], best_mr_swap_qual[idx_routes[1]] = new_quality[0], new_quality[1]
                            is_feasible = True            
            n += 1
            
        return best_mr_swap_sol, best_mr_swap_qual, idx_routes     
    
    def shakingMultiRoute2Exchange(self, best_mr_2exc_sol, best_mr_2exc_qual):
        range_solution = range(len(best_mr_2exc_sol))
        idx_routes = []
        tabu_list = []
        is_feasible = False
        n = 1
        
        max_n = (len(self.demands_array) - 1) / 2
        
        while (not is_feasible) and (n < max_n):
            idx_routes = self.random.sample(range_solution, 2)            
            
            if (len(best_mr_2exc_sol[idx_routes[0]]) >= 5) and (len(best_mr_2exc_sol[idx_routes[1]]) >= 5):
                min_quality = ((best_mr_2exc_qual[idx_routes[0]] + best_mr_2exc_qual[idx_routes[1]]) 
                           + ((best_mr_2exc_qual[idx_routes[0]] + best_mr_2exc_qual[idx_routes[1]]) * 0.2))
                
                temp_route_a, temp_route_b = best_mr_2exc_sol[idx_routes[0]].copy(), best_mr_2exc_sol[idx_routes[1]].copy()
                idx_n_i = self.random.choice(list(range(len(temp_route_a)))[1:-2])
                idx_n_j = self.random.choice(list(range(len(temp_route_b)))[1:-2])
                temp_route_a[idx_n_i], temp_route_a[idx_n_i+1], temp_route_b[idx_n_j], temp_route_b[idx_n_j+1] = temp_route_b[idx_n_j], temp_route_b[idx_n_j+1], temp_route_a[idx_n_i], temp_route_a[idx_n_i+1]
                
                if (temp_route_a not in tabu_list) or (temp_route_b not in tabu_list):                
                    tabu_list.append(temp_route_a)
                    tabu_list.append(temp_route_b)
                    routes_demands = self.calculateRoutesDemands([temp_route_a, temp_route_b])

                    if (routes_demands[0] <= self.vehicle_capacity) and (routes_demands[1] <= self.vehicle_capacity): 
                        new_quality = self.calculateSolutionQuality([temp_route_a, temp_route_b])
                        
                        if new_quality.sum() < min_quality:
                            best_mr_2exc_sol[idx_routes[0]], best_mr_2exc_sol[idx_routes[1]] = temp_route_a, temp_route_b
                            best_mr_2exc_qual[idx_routes[0]], best_mr_2exc_qual[idx_routes[1]] = new_quality[0], new_quality[1]
                            is_feasible = True            
            n += 1
            
        return best_mr_2exc_sol, best_mr_2exc_qual, idx_routes 
        
    
    def shaking(self, original_solution, original_quality, N):        
        if (N == 0):                     return original_solution, original_quality, list(range(len(original_solution)))
        elif (N == 1):                   return self.shakingMultiRouteSwap(original_solution.copy(), original_quality.copy())
        else:                            return self.shakingMultiRoute2Exchange(original_solution.copy(), original_quality.copy())

    def VNS(self, shaking_solution, shaking_quality, best_global_quality, N, idx_routes):
        from itertools import permutations 
        best_vns_solution = self.deepcopy(shaking_solution)
        best_vns_quality = self.deepcopy(shaking_quality)
        
        idx_perm = list(permutations(idx_routes, 2))
        for idx_r_i, idx_r_j in idx_perm:  
            '''
            # multiRouteSwap()
            best_vns_solution, best_vns_quality = self.multiRouteSwap(best_vns_solution.copy(), best_vns_quality.copy(), 
                                                                          idx_r_i, idx_r_j)
            '''
            
            tabu_list = []
            for n in range(N):
                # singleRouteSwap()
                best_vns_solution, best_vns_quality, tabu_list = self.singleRouteSwap(best_vns_solution.copy(), best_vns_quality.copy(),
                                                                                          idx_r_i, tabu_list)
                # singleRouteRelocate()            
                best_vns_solution, best_vns_quality, tabu_list = self.singleRouteRelocate(best_vns_solution.copy(), best_vns_quality.copy(),
                                                                                          idx_r_i, tabu_list)  
                          
            tabu_list = []
            for n in range(N):
                # singleRouteSwap()
                best_vns_solution, best_vns_quality, tabu_list = self.singleRouteSwap(best_vns_solution.copy(), best_vns_quality.copy(),
                                                                                      idx_r_j, tabu_list)
                # singleRouteRelocate()            
                best_vns_solution, best_vns_quality, tabu_list = self.singleRouteRelocate(best_vns_solution.copy(), best_vns_quality.copy(),
                                                                                          idx_r_j, tabu_list)
                
        return best_vns_solution, best_vns_quality
            
    def improve(self):
        import time
        
        self.intensity_percentage = (self.iteration / self.max_iterations)        
        best_global_solution = self.deepcopy(self.solution)
        best_global_quality = self.deepcopy(self.solution_quality)
        new_improve = False        
        i = 0
        
        if self.intensity_percentage <= 0.85: 
            max_time = len(self.demands_array) * 0.0025
        else:
            max_time = len(self.demands_array) * 0.0050       
        
        start_time = time.time()
        # print(str(time.time() - start_time) + ': ' + str(best_global_quality.sum()))
        while i < self.k_number:
            for shaking_ratio in range(3):
                shaking_solution, shaking_quality, idx_routes = self.shaking(best_global_solution.copy(), best_global_quality.copy(),
                                                                             shaking_ratio)
                vns_solution, vns_quality = self.VNS(shaking_solution.copy(), shaking_quality.copy(), best_global_quality.copy(), 
                                                     int((len(self.demands_array) / 3)), idx_routes)     

                if vns_quality.sum() < best_global_quality.sum():
                    # print(str(time.time() - start_time) + ': ' + str(vns_quality.sum()))
                    best_global_solution = self.deepcopy(vns_solution)
                    best_global_quality = self.deepcopy(vns_quality)
                    new_improve = True
                    
            if new_improve == True:
                i = 0
                new_improve = False 
            else:
                i += 1
            
            if (time.time() - start_time) >= max_time:
                break
        
        return best_global_solution, best_global_quality