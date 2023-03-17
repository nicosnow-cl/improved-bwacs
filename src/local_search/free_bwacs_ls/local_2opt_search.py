class Free2OPTSearch:
    def __init__(self, depot, solution, solution_quality, distances_matrix, demands_array, tare, vehicle_capacity):
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
        # if (idx_route == None): idx_route = self.random.choice(range(len(best_sr_swap_sol)))
        
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
                
    def shakingMultiRouteRelocate(self, best_mr_reloc_sol, best_mr_reloc_qual):
        range_solution = range(len(best_mr_reloc_sol))
        idx_routes = []
        tabu_list = []
        is_feasible = False
        n = 1
        
        max_n = (len(self.demands_array) - 1) / 3
        
        while (not is_feasible) and (n < max_n):
            idx_routes = self.random.sample(range_solution, 2)
            
            if (len(best_mr_reloc_sol[idx_routes[0]]) >= 4):
                min_quality = ((best_mr_reloc_qual[idx_routes[0]] + best_mr_reloc_qual[idx_routes[1]]) 
                           + ((best_mr_reloc_qual[idx_routes[0]] + best_mr_reloc_qual[idx_routes[1]]) * 0.2))
                
                temp_route_a, temp_route_b = best_mr_reloc_sol[idx_routes[0]].copy(), best_mr_reloc_sol[idx_routes[1]].copy()
                idx_n_i = self.random.choice(list(range(len(temp_route_a)))[1:-1])
                idx_n_j = self.random.choice(list(range(len(temp_route_b)))[1:-1])                
                temp_route_b.insert(idx_n_j, temp_route_a.pop(idx_n_i))
                
                if (temp_route_b not in tabu_list):
                    tabu_list.append(temp_route_b)
                    routes_demands = self.calculateRoutesDemands([temp_route_b])
                    
                    if (routes_demands[0] <= self.vehicle_capacity):
                        new_quality = self.calculateSolutionQuality([temp_route_a, temp_route_b])
                        
                        if new_quality.sum() < min_quality:
                            best_mr_reloc_sol[idx_routes[0]], best_mr_reloc_sol[idx_routes[1]] = temp_route_a, temp_route_b
                            best_mr_reloc_qual[idx_routes[0]], best_mr_reloc_qual[idx_routes[1]] = new_quality[0], new_quality[1]
                            is_feasible = True 
            n += 1
        
        return best_mr_reloc_sol, best_mr_reloc_qual, idx_routes
                
    def shakingMultiRouteSwap(self, best_mr_swap_sol, best_mr_swap_qual):
        range_solution = range(len(best_mr_swap_sol))
        idx_routes = []
        tabu_list = []
        is_feasible = False
        n = 1
        
        max_n = (len(self.demands_array) - 1) / 3
        
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
           
    
    def shaking(self, original_solution, original_quality, N):        
        if (N == 0):                     return original_solution, original_quality, list(range(len(original_solution)))
        elif (N == 1):                   return self.shakingMultiRouteRelocate(original_solution.copy(), original_quality.copy()) 
        elif (N == 2):                   return self.shakingMultiRouteSwap(original_solution.copy(), original_quality.copy())

    def localSearch(self, shaking_solution, shaking_quality, best_global_quality, idx_routes):
        best_vns_solution = self.deepcopy(shaking_solution)
        best_vns_quality = self.deepcopy(shaking_quality)
        
        for idx_route in idx_routes:
            N = int(len(best_vns_solution[idx_route]) * 0.75)
            tabu_list = []
            
            for n in range(N):
                # singleRouteSwap()
                best_vns_solution, best_vns_quality, tabu_list = self.singleRouteSwap(best_vns_solution.copy(), best_vns_quality.copy(),
                                                                                          idx_route, tabu_list)
            for n in range(N):     
                # singleRouteRelocate()            
                best_vns_solution, best_vns_quality, tabu_list = self.singleRouteRelocate(best_vns_solution.copy(), best_vns_quality.copy(),
                                                                                          idx_route, tabu_list) 
                
        return best_vns_solution, best_vns_quality
            
    def improve(self):
        import time
           
        best_global_solution = self.deepcopy(self.solution)
        best_global_quality = self.deepcopy(self.solution_quality)
        max_iterations = 1
             
        start_time = time.time()
        # print(str(time.time() - start_time) + ': ' + str(best_global_quality.sum()))
        for iteration in range(max_iterations):
            for shaking_ratio in range(1):
                shaking_solution, shaking_quality, idx_routes = self.shaking(best_global_solution.copy(), best_global_quality.copy(),
                                                                             shaking_ratio)
                vns_solution, vns_quality = self.localSearch(shaking_solution.copy(), shaking_quality.copy(), best_global_quality.copy(),
                                                             idx_routes)     

                if vns_quality.sum() < best_global_quality.sum():
                    # print(str(time.time() - start_time) + ': ' + str(vns_quality.sum()))
                    best_global_solution = self.deepcopy(vns_solution)
                    best_global_quality = self.deepcopy(vns_quality)
            
        return best_global_solution, best_global_quality