class RestrictedGlobalGVNS:
    def __init__(self, depot, solution, solution_quality, distances_matrix, demands_array, tare, vehicle_capacity, k_number):
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
        self.np = np
        self.random = random
        self.deepcopy = deepcopy
        self.tabu_list = []
    
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
        
    def singleRouteRelocate(self, best_sr_reloc_sol, best_sr_reloc_qual, route_tabu_list):
        if len(best_sr_reloc_sol) >= 4:
            temp_route = best_sr_reloc_sol.copy()
            idx_i, idx_j = self.random.sample(list(range(len(temp_route)))[1:-1], 2)
            temp_route.insert(idx_j, temp_route.pop(idx_i))

            if temp_route not in route_tabu_list:
                route_tabu_list.append(temp_route)
                temp_quality = self.calculateSolutionQuality([temp_route])
                
                if temp_quality[0] < best_sr_reloc_qual:
                    best_sr_reloc_sol = temp_route
                    best_sr_reloc_qual = temp_quality[0]
                    
        return best_sr_reloc_sol, best_sr_reloc_qual, route_tabu_list
    
    def singleRouteSwap(self, best_sr_swap_sol, best_sr_swap_qual, route_tabu_list):
        if len(best_sr_swap_sol) >= 4:
            temp_route = best_sr_swap_sol.copy()
            idx_i, idx_j = self.random.sample(list(range(len(temp_route)))[1:-1], 2)        
            temp_route[idx_i], temp_route[idx_j] = temp_route[idx_j], temp_route[idx_i] 
            
            if temp_route not in route_tabu_list:
                route_tabu_list.append(temp_route)
                temp_quality = self.calculateSolutionQuality([temp_route])

                if temp_quality[0] < best_sr_swap_qual:
                    best_sr_swap_sol = temp_route
                    best_sr_swap_qual = temp_quality[0]
            
        return best_sr_swap_sol, best_sr_swap_qual, route_tabu_list    
    
    def multiRouteSwap(self, best_mr_swap_sol, best_mr_swap_qual, idx_r_i, idx_r_j):
        tabu_list = []
        is_feasible = False
        n = 1
        
        max_n = (len(self.demands_array) - 1) / 2 
        
        while (not is_feasible) and (n < max_n):            
            if (len(best_mr_swap_sol[idx_r_i]) >= 3) and (len(best_mr_swap_sol[idx_r_j]) >= 3):
                idx_r_i, idx_r_j = self.random.sample(range(len(best_mr_swap_sol)), 2)
                temp_route_a, temp_route_b = best_mr_swap_sol[idx_r_i].copy(), best_mr_swap_sol[idx_r_j].copy()
                idx_n_i = self.random.choice(list(range(len(temp_route_a)))[1:-1])
                idx_n_j = self.random.choice(list(range(len(temp_route_b)))[1:-1])
                temp_route_a[idx_n_i], temp_route_b[idx_n_j] = temp_route_b[idx_n_j], temp_route_a[idx_n_i]
                
                if (temp_route_a not in tabu_list) or (temp_route_b not in tabu_list):                
                    tabu_list.append(temp_route_a)
                    tabu_list.append(temp_route_b)
                    routes_demands = self.calculateRoutesDemands([temp_route_a, temp_route_b])

                    if (routes_demands[0] <= self.vehicle_capacity) and (routes_demands[1] <= self.vehicle_capacity): 
                        # original_quality = self.calculateSolutionQuality([best_mr_swap_sol[idx_r_i], best_mr_swap_sol[idx_r_j]])
                        min_quality = ((best_mr_swap_qual[idx_r_i] + best_mr_swap_qual[idx_r_j])
                                       + ((best_mr_swap_qual[idx_r_i] + best_mr_swap_qual[idx_r_j]) * 0.3))
                        new_quality = self.calculateSolutionQuality([temp_route_a, temp_route_b])

                        if new_quality.sum() < min_quality:
                            best_mr_swap_sol[idx_r_i], best_mr_swap_sol[idx_r_j] = self.deepcopy(temp_route_a), self.deepcopy(temp_route_b)
                            best_mr_swap_qual[idx_r_i], best_mr_swap_qual[idx_r_j] = self.deepcopy(new_quality[0]), self.deepcopy(new_quality[1])
                            is_feasible = True
            n += 1
            
        return best_mr_swap_sol, best_mr_swap_qual     
            
    def shakingMultiRouteSwap(self, best_mr_swap_sol, best_mr_swap_qual):
        range_solution = range(len(best_mr_swap_sol))
        idx_routes = []
        is_feasible = False              
        max_n = (len(self.demands_array) - 1) / 2
        n = 1
        
        while (not is_feasible) and (n < max_n):
            idx_routes = self.random.sample(range_solution, 2)            
            
            if (len(best_mr_swap_sol[idx_routes[0]]) >= 3) and (len(best_mr_swap_sol[idx_routes[1]]) >= 3):
                temp_route_a, temp_route_b = self.deepcopy(best_mr_swap_sol[idx_routes[0]]), self.deepcopy(best_mr_swap_sol[idx_routes[1]])
                idx_n_i = self.random.choice(list(range(len(temp_route_a)))[1:-1])
                idx_n_j = self.random.choice(list(range(len(temp_route_b)))[1:-1])
                temp_route_a[idx_n_i], temp_route_b[idx_n_j] = temp_route_b[idx_n_j], temp_route_a[idx_n_i]

                if (temp_route_a not in self.tabu_list) or (temp_route_b not in self.tabu_list):
                    routes_demands = self.calculateRoutesDemands([temp_route_a, temp_route_b])

                    if (routes_demands[0] <= self.vehicle_capacity) and (routes_demands[1] <= self.vehicle_capacity): 
                        new_quality = self.calculateSolutionQuality([temp_route_a, temp_route_b])
                        best_mr_swap_sol[idx_routes[0]], best_mr_swap_sol[idx_routes[1]] = temp_route_a, temp_route_b
                        best_mr_swap_qual[idx_routes[0]], best_mr_swap_qual[idx_routes[1]] = new_quality[0], new_quality[1]
                        is_feasible = True
                    else:
                        self.tabu_list.append(temp_route_a)
                        self.tabu_list.append(temp_route_b)
            n += 1
        
        return best_mr_swap_sol, best_mr_swap_qual, idx_routes
    
    def shakingMultiRoute3Swap(self, best_mr_3swap_sol, best_mr_3swap_qual):
        range_solution = range(len(best_mr_3swap_sol))
        idx_routes = []
        is_feasible = False              
        max_n = (len(self.demands_array) - 1) / 2
        n = 1
        
        while (not is_feasible) and (n < max_n):
            idx_routes = self.random.sample(range_solution, 3)            
            
            if ((len(best_mr_3swap_sol[idx_routes[0]]) >= 3) and (len(best_mr_3swap_sol[idx_routes[1]]) >= 3) 
                and (len(best_mr_3swap_sol[idx_routes[2]]) >= 3)) :
                temp_r_a, temp_r_b, temp_r_c = best_mr_3swap_sol[idx_routes[0]].copy(), best_mr_3swap_sol[idx_routes[1]].copy(), best_mr_3swap_sol[idx_routes[2]].copy()
                idx_n_i = self.random.choice(list(range(len(temp_r_a)))[1:-1])
                idx_n_j = self.random.choice(list(range(len(temp_r_b)))[1:-1])
                idx_n_k = self.random.choice(list(range(len(temp_r_c)))[1:-1])
                temp_r_a[idx_n_i], temp_r_b[idx_n_j], temp_r_c[idx_n_k] = temp_r_c[idx_n_k], temp_r_a[idx_n_i], temp_r_b[idx_n_j]
                
                if (temp_r_a not in self.tabu_list) or (temp_r_b not in self.tabu_list) or (temp_r_c not in self.tabu_list):
                    routes_demands = self.calculateRoutesDemands([temp_r_a, temp_r_b, temp_r_c])

                    if ((routes_demands[0] <= self.vehicle_capacity) and (routes_demands[1] <= self.vehicle_capacity) and 
                        (routes_demands[2] <= self.vehicle_capacity)): 
                        new_quality = self.calculateSolutionQuality([temp_r_a, temp_r_b, temp_r_c])
                        best_mr_3swap_sol[idx_routes[0]], best_mr_3swap_sol[idx_routes[1]], best_mr_3swap_sol[idx_routes[2]] = temp_r_a, temp_r_b, temp_r_c
                        best_mr_3swap_qual[idx_routes[0]], best_mr_3swap_qual[idx_routes[1]], best_mr_3swap_qual[idx_routes[2]] = new_quality[0], new_quality[1], new_quality[2]
                        is_feasible = True
                    else:
                        self.tabu_list.append(temp_r_a)
                        self.tabu_list.append(temp_r_b)
                        self.tabu_list.append(temp_r_c)
            n += 1
            
        return best_mr_3swap_sol, best_mr_3swap_qual, idx_routes 
    
    def shakingMultiRoute2Exchange(self, best_mr_2exc_sol, best_mr_2exc_qual):
        range_solution = range(len(best_mr_2exc_sol))
        idx_routes = []
        is_feasible = False
        max_n = (len(self.demands_array) - 1) / 2
        n = 1
                
        while (not is_feasible) and (n < max_n):
            idx_routes = self.random.sample(range_solution, 2)            
            
            if (len(best_mr_2exc_sol[idx_routes[0]]) >= 5) and (len(best_mr_2exc_sol[idx_routes[1]]) >= 5):
                temp_r_a, temp_r_b = best_mr_2exc_sol[idx_routes[0]].copy(), best_mr_2exc_sol[idx_routes[1]].copy()
                idx_n_i = self.random.choice(list(range(len(temp_r_a)))[1:-2])
                idx_n_j = self.random.choice(list(range(len(temp_r_b)))[1:-2])
                temp_r_a[idx_n_i], temp_r_a[idx_n_i+1], temp_r_b[idx_n_j], temp_r_b[idx_n_j+1] = temp_r_b[idx_n_j], temp_r_b[idx_n_j+1], temp_r_a[idx_n_i], temp_r_a[idx_n_i+1]
                
                if (temp_r_a not in self.tabu_list) and (temp_r_b not in self.tabu_list):
                    routes_demands = self.calculateRoutesDemands([temp_r_a, temp_r_b])

                    if (routes_demands[0] <= self.vehicle_capacity) and (routes_demands[1] <= self.vehicle_capacity):
                        new_quality = self.calculateSolutionQuality([temp_r_a, temp_r_b])
                        best_mr_2exc_sol[idx_routes[0]], best_mr_2exc_sol[idx_routes[1]] = temp_r_a, temp_r_b
                        best_mr_2exc_qual[idx_routes[0]], best_mr_2exc_qual[idx_routes[1]] = new_quality[0], new_quality[1]
                        is_feasible = True
                    else:
                        self.tabu_list.append(temp_r_a)
                        self.tabu_list.append(temp_r_b)
            n += 1
            
        return best_mr_2exc_sol, best_mr_2exc_qual, idx_routes
    
    def shakingMultiRoute3Exchange(self, best_mr_3exc_sol, best_mr_3exc_qual):
        range_solution = range(len(best_mr_3exc_sol))
        idx_routes = []
        is_feasible = False
        max_n = (len(self.demands_array) - 1) / 2
        n = 1
                
        while (not is_feasible) and (n < max_n):
            idx_routes = self.random.sample(range_solution, 3)            
            
            if ((len(best_mr_3exc_sol[idx_routes[0]]) >= 5) and (len(best_mr_3exc_sol[idx_routes[1]]) >= 5) 
                and (len(best_mr_3exc_sol[idx_routes[2]]) >= 5)):
                temp_r_a, temp_r_b, temp_r_c = best_mr_3exc_sol[idx_routes[0]].copy(), best_mr_3exc_sol[idx_routes[1]].copy(), best_mr_3exc_sol[idx_routes[2]].copy()
                idx_n_i = self.random.choice(list(range(len(temp_r_a)))[1:-2])
                idx_n_j = self.random.choice(list(range(len(temp_r_b)))[1:-2])
                idx_n_k = self.random.choice(list(range(len(temp_r_c)))[1:-2])
                temp_r_a[idx_n_i], temp_r_a[idx_n_i+1], temp_r_b[idx_n_j], temp_r_b[idx_n_j+1], temp_r_c[idx_n_k], temp_r_c[idx_n_k+1] = temp_r_c[idx_n_k], temp_r_c[idx_n_k+1], temp_r_a[idx_n_i], temp_r_a[idx_n_i+1], temp_r_b[idx_n_j], temp_r_b[idx_n_j+1]
                
                if (temp_r_a not in self.tabu_list) and (temp_r_b not in self.tabu_list) and (temp_r_c not in self.tabu_list):
                    routes_demands = self.calculateRoutesDemands([temp_r_a, temp_r_b, temp_r_c])

                    if ((routes_demands[0] <= self.vehicle_capacity) and (routes_demands[1] <= self.vehicle_capacity)
                       and (routes_demands[2] <= self.vehicle_capacity)):
                        new_quality = self.calculateSolutionQuality([temp_r_a, temp_r_b, temp_r_c])
                        best_mr_3exc_sol[idx_routes[0]], best_mr_3exc_sol[idx_routes[1]], best_mr_3exc_sol[idx_routes[2]] = temp_r_a, temp_r_b, temp_r_c
                        best_mr_3exc_qual[idx_routes[0]], best_mr_3exc_qual[idx_routes[1]], best_mr_3exc_qual[idx_routes[2]] = new_quality[0], new_quality[1], new_quality[2]
                        is_feasible = True
                    else:
                        self.tabu_list.append(temp_r_a)
                        self.tabu_list.append(temp_r_b)
                        self.tabu_list.append(temp_r_c)
            n += 1
            
        return best_mr_3exc_sol, best_mr_3exc_qual, idx_routes 
        
    
    def shaking(self, original_solution, original_quality, N):
        if (N == 0):                     return original_solution, original_quality, list(range(len(original_solution)))
        elif (N == 1):                   return self.shakingMultiRouteSwap(original_solution.copy(), original_quality.copy())
        elif (N == 2):                   return self.shakingMultiRoute3Swap(original_solution.copy(), original_quality.copy())
        elif (N == 3):                   return self.shakingMultiRoute2Exchange(original_solution.copy(), original_quality.copy())
        else:                            return self.shakingMultiRoute3Exchange(original_solution.copy(), original_quality.copy())

    def VNS(self, shaking_solution, shaking_quality, N, idx_routes): 
        best_vns_solution = shaking_solution.copy()
        best_vns_quality = shaking_quality.copy()        
        
        for idx_route in idx_routes:
            route_tabu_list = []
            
            for n in range(N):
                best_vns_solution[idx_route], best_vns_quality[idx_route], route_tabu_list = self.singleRouteSwap(best_vns_solution[idx_route].copy(), best_vns_quality[idx_route].copy(), route_tabu_list)           
                best_vns_solution[idx_route], best_vns_quality[idx_route], route_tabu_list = self.singleRouteRelocate(best_vns_solution[idx_route].copy(), best_vns_quality[idx_route].copy(), route_tabu_list)     
        
        return best_vns_solution, best_vns_quality
            
    def improve(self):        
        import time
        
        best_global_solution = self.deepcopy(self.solution)
        best_global_quality = self.deepcopy(self.solution_quality)
        new_improve = False
        max_time = len(self.demands_array) * 0.25      
        
        start_time = time.time()
        print(str(time.time() - start_time) + ': ' + str(best_global_quality.sum()))
        while True:
            for shaking_ratio in range(5):
                shaking_solution, shaking_quality, idx_routes = self.shaking(best_global_solution.copy(), best_global_quality.copy(),
                                                                             shaking_ratio)
                vns_solution, vns_quality = self.VNS(shaking_solution, shaking_quality, int(len(self.demands_array) / 2), idx_routes)     

                if vns_quality.sum() < best_global_quality.sum():
                    print(str(time.time() - start_time) + ': ' + str(vns_quality.sum()))
                    best_global_solution = self.deepcopy(vns_solution)
                    best_global_quality = self.deepcopy(vns_quality)
                    new_improve = True
            
            if (time.time() - start_time) >= max_time:
                break

        return best_global_solution, best_global_quality
            
            
