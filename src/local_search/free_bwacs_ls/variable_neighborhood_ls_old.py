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
                    before_node = self.depot
                else:
                    route_energy += self.distances_matrix[before_node][i] * vehicle_weight
                    vehicle_weight += self.demands_array[i]
                    before_node = i

            routes_energies[k] = route_energy      
            
        return routes_energies
    
    def generateRoutesArcs(self, solution):
        routes_arcs = []
        
        for route in solution:
            route_arcs = []
            
            for pos, i in enumerate(route):
                if pos == 0:
                    before_node = self.depot
                else:
                    route_arcs.append((before_node, i))
                    before_node = i

            routes_arcs.append(route_arcs)      
            
        return routes_arcs
    
    def calculateRoutesWeights(self, solution):
        routes_arcs_weights = []
        
        for route in solution:
            route_weights = {}
            
            for pos, i in enumerate(route):
                if pos == 0:
                    vehicle_weight = self.tare
                    before_node = self.depot
                else:
                    route_weights[(before_node, i)] = vehicle_weight
                    vehicle_weight += self.demands_array[i]
                    before_node = i

            routes_arcs_weights.append(route_weights)
         
        return routes_arcs_weights
    
    def singleRouteRelocate(self, best_sr_reloc_sol, best_sr_reloc_qual, idx_route = None):     
        if (idx_route == None): idx_route = self.random.choice(range(len(best_sr_reloc_sol)))
        
        if len(best_sr_reloc_sol[idx_route]) >= 4:
            temp_route = best_sr_reloc_sol[idx_route].copy()
            idx_i, idx_j = self.random.sample(list(range(len(temp_route)))[1:-1], 2)
            temp_route.insert(idx_j, temp_route.pop(idx_i))
            temp_quality = self.calculateSolutionQuality([temp_route])

            if temp_quality < best_sr_reloc_qual[idx_route]:
                best_sr_reloc_sol[idx_route] = temp_route
                best_sr_reloc_qual[idx_route] = temp_quality
            
        return best_sr_reloc_sol, best_sr_reloc_qual
    
    def singleRouteSwap(self, best_sr_swap_sol, best_sr_swap_qual):             
        idx_route = self.random.choice(range(len(best_sr_swap_sol)))
        
        if len(best_sr_swap_sol[idx_route]) >= 4:
            temp_route = best_sr_swap_sol[idx_route].copy()
            idx_i, idx_j = self.random.sample(list(range(len(temp_route)))[1:-1], 2)        
            temp_route[idx_i], temp_route[idx_j] = temp_route[idx_j], temp_route[idx_i] 
            temp_quality = self.calculateSolutionQuality([temp_route])

            if temp_quality < best_sr_swap_qual[idx_route]:
                best_sr_swap_sol[idx_route] = temp_route
                best_sr_swap_qual[idx_route] = temp_quality
            
        return best_sr_swap_sol, best_sr_swap_qual
    
    def multiRouteRelocate(self, best_mr_reloc_sol, best_mr_reloc_qual, idx_r_i = None, idx_r_j = None):
        if (idx_r_i == None) or (idx_r_j == None): range_solution = range(len(best_mr_reloc_sol))
        tabu_list = []
        is_feasible = False
        n = 1
        
        if self.intensity_percentage <= 0.85: max_n = (len(self.demands_array) - 1) / self.k_number
        else:                                 max_n = (len(self.demands_array) - 1) / 2       
        
        while (not is_feasible) and (n < max_n):
            if (idx_r_i == None) or (idx_r_j == None): idx_r_i, idx_r_j = self.random.sample(range_solution, 2)            
            temp_route_a, temp_route_b = best_mr_reloc_sol[idx_r_i].copy(), best_mr_reloc_sol[idx_r_j].copy()            
            idx_n_i = self.random.choice(list(range(len(temp_route_a)))[1:-1])            
            temp_route_b.insert(1, temp_route_a.pop(idx_n_i))
            
            if (temp_route_b not in tabu_list):
                tabu_list.append(temp_route_b)
                route_demand = self.calculateRoutesDemands([temp_route_b])

                if (route_demand <= self.vehicle_capacity):
                    original_quality = self.calculateSolutionQuality([best_mr_reloc_sol[idx_r_i], best_mr_reloc_sol[idx_r_j]])
                    new_quality = self.calculateSolutionQuality([temp_route_a, temp_route_b])

                    if new_quality.sum() < original_quality.sum():
                        best_mr_reloc_sol[idx_r_i], best_mr_reloc_sol[idx_r_j] = temp_route_a, temp_route_b
                        best_mr_reloc_qual[idx_r_i], best_mr_reloc_qual[idx_r_j] = new_quality[0], new_quality[1]
                        is_feasible = True
            n += 1

        return best_mr_reloc_sol, best_mr_reloc_qual
    
    def multiRouteSwap(self, best_mr_swap_sol, best_mr_swap_qual, idx_r_i = None, idx_r_j = None):
        if (idx_r_i == None) or (idx_r_j == None): range_solution = range(len(best_mr_swap_sol))
        tabu_list = []
        is_feasible = False
        n = 1
        
        if self.intensity_percentage <= 0.85: max_n = (len(self.demands_array) - 1) / self.k_number
        else:                                 max_n = (len(self.demands_array) - 1) / 2
            
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
    
    def multiRoute3Swap(self, best_mr_3swap_sol, best_mr_3swap_qual, idx_r_i = None, idx_r_j = None):
        if (idx_r_i == None) or (idx_r_j == None): range_solution = range(len(best_mr_swap_sol))
        tabu_list = []
        is_feasible = False
        n = 1
        
        if self.intensity_percentage <= 0.85: max_n = (len(self.demands_array) - 1) / self.k_number
        else:                                 max_n = (len(self.demands_array) - 1) / 2 
                  
        while (not is_feasible) and (n < max_n):
            idx_r_i, idx_r_j, idx_r_k = self.random.sample(range_solution, 3)
            
            if (len(best_mr_3swap_sol[idx_r_i]) >= 3) and (len(best_mr_3swap_sol[idx_r_j]) >= 3) and (len(best_mr_3swap_sol[idx_r_k]) >= 3):
                temp_r_i = best_mr_3swap_sol[idx_r_i].copy()
                temp_r_j = best_mr_3swap_sol[idx_r_j].copy()
                temp_r_k = best_mr_3swap_sol[idx_r_k].copy() 
                                
                idx_n_i = self.random.choice(list(range(len(temp_r_i)))[1:-1])
                idx_n_j = self.random.choice(list(range(len(temp_r_j)))[1:-1]) 
                idx_n_k = self.random.choice(list(range(len(temp_r_k)))[1:-1])

                temp_r_i[idx_n_i], temp_r_j[idx_n_j], temp_r_k[idx_n_k] = temp_r_k[idx_n_k], temp_r_i[idx_n_i], temp_r_j[idx_n_j]
                
                if (temp_r_i not in tabu_list) or (temp_r_j not in tabu_list) or (temp_r_k not in tabu_list):                
                    tabu_list.append(temp_r_i)
                    tabu_list.append(temp_r_j)
                    tabu_list.append(temp_r_k)
                    routes_demands = self.calculateRoutesDemands([temp_r_i, temp_r_j, temp_r_k])

                    if ((routes_demands[0] <= self.vehicle_capacity) and (routes_demands[1] <= self.vehicle_capacity) 
                        and (routes_demands[2] <= self.vehicle_capacity)):
                        original_quality = self.calculateSolutionQuality([best_mr_3swap_sol[idx_r_i], best_mr_3swap_sol[idx_r_j],
                                                                          best_mr_3swap_sol[idx_r_k]])
                        new_quality = self.calculateSolutionQuality([temp_r_i, temp_r_j, temp_r_k])
                            
                        if new_quality.sum() < original_quality.sum():
                            best_mr_3swap_sol[idx_r_i] = temp_r_i
                            best_mr_3swap_qual[idx_r_i] = new_quality[0]
                            best_mr_3swap_sol[idx_r_j] = temp_r_j
                            best_mr_3swap_qual[idx_r_j] = new_quality[1]
                            best_mr_3swap_sol[idx_r_k] = temp_r_k
                            best_mr_3swap_qual[idx_r_k] = new_quality[2]
                            is_feasible = True
            n += 1
            
        return best_mr_3swap_sol, best_mr_3swap_qual
    
    def shakingMultiRouteRelocate(self, best_mr_reloc_sol, best_mr_reloc_qual):
        range_solution = range(len(best_mr_reloc_sol))        
        idx_routes = []
        tabu_list = []
        is_feasible = False
        n = 1
        
        if self.intensity_percentage <= 0.85: max_n = (len(self.demands_array) - 1) / self.k_number
        else:                                 max_n = (len(self.demands_array) - 1) / 2      
        
        while (not is_feasible) and (n < max_n):
            idx_routes = self.random.sample(range_solution, 2)            

            if (len(best_mr_reloc_sol[idx_routes[0]]) >= 3):
                min_quality = ((best_mr_reloc_qual[idx_routes[0]] + best_mr_reloc_qual[idx_routes[1]]) 
                           + ((best_mr_reloc_qual[idx_routes[0]] + best_mr_reloc_qual[idx_routes[1]]) * 0.2))
                
                temp_route_a, temp_route_b = best_mr_reloc_sol[idx_routes[0]].copy(), best_mr_reloc_sol[idx_routes[1]].copy()            
                idx_n_i = self.random.choice(list(range(len(temp_route_a)))[1:-1])            
                temp_route_b.insert(1, temp_route_a.pop(idx_n_i))

                if (temp_route_b not in tabu_list):
                    tabu_list.append(temp_route_b)
                    route_demand = self.calculateRoutesDemands([temp_route_b])

                    if (route_demand <= self.vehicle_capacity):
                        new_quality = self.calculateSolutionQuality([temp_route_a, temp_route_b])
                        
                        if new_quality.sum() <= min_quality:
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
        
        if self.intensity_percentage <= 0.85: max_n = (len(self.demands_array) - 1) / self.k_number
        else:                                 max_n = (len(self.demands_array) - 1) / 2
            
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
                        
                        if new_quality.sum() <= min_quality:
                            best_mr_swap_sol[idx_routes[0]], best_mr_swap_sol[idx_routes[1]] = temp_route_a, temp_route_b
                            best_mr_swap_qual[idx_routes[0]], best_mr_swap_qual[idx_routes[1]] = new_quality[0], new_quality[1]
                            is_feasible = True            
            n += 1
            
        return best_mr_swap_sol, best_mr_swap_qual, idx_routes     
    
    def shakingMultiRoute3Swap(self, best_mr_3swap_sol, best_mr_3swap_qual):
        range_solution = range(len(best_mr_3swap_sol))
        idx_routes = []
        tabu_list = []
        is_feasible = False
        n = 1
        
        if self.intensity_percentage <= 0.85: max_n = (len(self.demands_array) - 1) / self.k_number
        else:                                 max_n = (len(self.demands_array) - 1) / 2 
                  
        while (not is_feasible) and (n < max_n):
            idx_routes = self.random.sample(range_solution, 3)            
            
            if ((len(best_mr_3swap_sol[idx_routes[0]]) >= 3) and (len(best_mr_3swap_sol[idx_routes[1]]) >= 3) 
                and (len(best_mr_3swap_sol[idx_routes[2]]) >= 3)):
                min_quality = ((best_mr_3swap_qual[idx_routes[0]] + best_mr_3swap_qual[idx_routes[1]] + best_mr_3swap_qual[idx_routes[2]]) 
                           + ((best_mr_3swap_qual[idx_routes[0]] + best_mr_3swap_qual[idx_routes[1]] + best_mr_3swap_qual[idx_routes[2]]) 
                              * 0.2))
                
                temp_r_i = best_mr_3swap_sol[idx_routes[0]].copy()
                temp_r_j = best_mr_3swap_sol[idx_routes[1]].copy()
                temp_r_k = best_mr_3swap_sol[idx_routes[2]].copy() 
                
                idx_n_i = self.random.choice(list(range(len(temp_r_i)))[1:-1])
                idx_n_j = self.random.choice(list(range(len(temp_r_j)))[1:-1]) 
                idx_n_k = self.random.choice(list(range(len(temp_r_k)))[1:-1])

                temp_r_i[idx_n_i], temp_r_j[idx_n_j], temp_r_k[idx_n_k] = temp_r_k[idx_n_k], temp_r_i[idx_n_i], temp_r_j[idx_n_j]
                
                if (temp_r_i not in tabu_list) or (temp_r_j not in tabu_list) or (temp_r_k not in tabu_list):                
                    tabu_list.append(temp_r_i)
                    tabu_list.append(temp_r_j)
                    tabu_list.append(temp_r_k)
                    routes_demands = self.calculateRoutesDemands([temp_r_i, temp_r_j, temp_r_k])

                    if ((routes_demands[0] <= self.vehicle_capacity) and (routes_demands[1] <= self.vehicle_capacity) 
                        and (routes_demands[2] <= self.vehicle_capacity)):                        
                        new_quality = self.calculateSolutionQuality([temp_r_i, temp_r_j, temp_r_k])
                        
                        if new_quality.sum() <= min_quality:
                            best_mr_3swap_sol[idx_routes[0]] = temp_r_i
                            best_mr_3swap_qual[idx_routes[0]] = new_quality[0]
                            best_mr_3swap_sol[idx_routes[1]] = temp_r_j
                            best_mr_3swap_qual[idx_routes[1]] = new_quality[1]
                            best_mr_3swap_sol[idx_routes[2]] = temp_r_k
                            best_mr_3swap_qual[idx_routes[2]] = new_quality[2]
                            is_feasible = True            
            n += 1
            
        return best_mr_3swap_sol, best_mr_3swap_qual, idx_routes
    
    def shakingMultiRoute3Exchange(self, best_mr_3Excg_sol, best_mr_3Excg_qual):
        range_solution = range(len(best_mr_3Excg_sol))
        idx_routes = []
        tabu_list = []
        is_feasible = False
        n = 1
        
        if self.intensity_percentage <= 0.85: max_n = (len(self.demands_array) - 1) / self.k_number
        else:                                 max_n = (len(self.demands_array) - 1) / 2
        
        while (not is_feasible) and (n < max_n):
            idx_routes = self.random.sample(range_solution, 3)
            
            if ((len(best_mr_3Excg_sol[idx_routes[0]]) >= 4) and (len(best_mr_3Excg_sol[idx_routes[1]]) >= 4) 
                and (len(best_mr_3Excg_sol[idx_routes[2]]) >= 4)):
                min_quality = ((best_mr_3Excg_qual[idx_routes[0]] + best_mr_3Excg_qual[idx_routes[1]] + best_mr_3Excg_qual[idx_routes[2]]) 
                           + ((best_mr_3Excg_qual[idx_routes[0]] + best_mr_3Excg_qual[idx_routes[1]] + best_mr_3Excg_qual[idx_routes[2]]) 
                              * 0.2))
            
                temp_r_i = best_mr_3Excg_sol[idx_routes[0]].copy()
                temp_r_j = best_mr_3Excg_sol[idx_routes[1]].copy()
                temp_r_k = best_mr_3Excg_sol[idx_routes[2]].copy()

                idx_n_i = self.random.choice(list(range(len(temp_r_i)))[1:-2])
                idx_n_j = self.random.choice(list(range(len(temp_r_j)))[1:-2]) 
                idx_n_k = self.random.choice(list(range(len(temp_r_k)))[1:-2])

                temp_r_i[idx_n_i], temp_r_i[idx_n_i + 1], temp_r_j[idx_n_j], temp_r_j[idx_n_j + 1], temp_r_k[idx_n_k], temp_r_k[idx_n_k + 1] =  temp_r_k[idx_n_k], temp_r_k[idx_n_k + 1], temp_r_i[idx_n_i], temp_r_i[idx_n_i + 1], temp_r_j[idx_n_j], temp_r_j[idx_n_j + 1]

                if (temp_r_i not in tabu_list) or (temp_r_j not in tabu_list) or (temp_r_k not in tabu_list):
                    tabu_list.append(temp_r_i)
                    tabu_list.append(temp_r_j)
                    tabu_list.append(temp_r_k)

                    routes_demands = self.calculateRoutesDemands([temp_r_i, temp_r_j, temp_r_k])

                    if ((routes_demands[0] <= self.vehicle_capacity) and (routes_demands[1] <= self.vehicle_capacity) 
                            and (routes_demands[2] <= self.vehicle_capacity)):
                        new_quality = self.calculateSolutionQuality([temp_r_i, temp_r_j, temp_r_k])
                        
                        if new_quality.sum() <= min_quality:
                            best_mr_3Excg_sol[idx_routes[0]] = temp_r_i
                            best_mr_3Excg_qual[idx_routes[0]] = new_quality[0]
                            best_mr_3Excg_sol[idx_routes[1]] = temp_r_j
                            best_mr_3Excg_qual[idx_routes[1]] = new_quality[1]
                            best_mr_3Excg_sol[idx_routes[2]] = temp_r_k
                            best_mr_3Excg_qual[idx_routes[2]] = new_quality[2]
                            is_feasible = True 
            n += 1
        
        return best_mr_3Excg_sol, best_mr_3Excg_qual, idx_routes
                        
    def shaking(self, original_solution, original_quality, N):        

        
        if self.intensity_percentage <= 0.85:
            if (N <= 1):            return original_solution, original_quality, self.random.sample(range(len(original_solution)), 2)
            else:                   return self.shakingMultiRouteSwap(original_solution, original_quality)
            # else:                 return self.shakingMultiRoute3Swap(original_solution, original_quality)
        else:
            if (N == 0):          return original_solution, original_quality, self.random.sample(range(len(original_solution)), 2)          
            elif (N == 1):        return self.shakingMultiRoute3Swap(original_solution, original_quality)
            else:                 return self.shakingMultiRoute3Exchange(original_solution, original_quality)
           
        
    def VNS(self, shaking_solution, shaking_quality, best_global_quality, N, idx_routes):        
        from itertools import permutations 
        best_vns_solution = shaking_solution.copy()
        best_vns_quality = shaking_quality.copy()
        
        idx_perm = list(permutations(idx_routes, 2))
        for idx_r_i, idx_r_j in idx_perm:          
            if self.intensity_percentage <= 0.85:            
                for n in range(int(N * 0.50)):    
                    # multiRouteSwap()
                    best_vns_solution, best_vns_quality = self.multiRouteSwap(best_vns_solution, best_vns_quality, 
                                                                              idx_r_i, idx_r_j)          
            else:            
                for n in range(int(N * 0.85)):   
                    # multiRouteSwap()
                    best_vns_solution, best_vns_quality = self.multiRouteSwap(best_vns_solution, best_vns_quality, 
                                                                              idx_r_i, idx_r_j)            
            for n in range(N):
                # singleRouteRelocate()            
                best_vns_solution, best_vns_quality = self.singleRouteRelocate(best_vns_solution, best_vns_quality, 
                                                                               idx_r_i)
            for n in range(N):
                # singleRouteRelocate()            
                best_vns_solution, best_vns_quality = self.singleRouteRelocate(best_vns_solution, best_vns_quality, 
                                                                               idx_r_j)
            
            if best_vns_quality.sum() < best_global_quality.sum():
                return best_vns_solution, best_vns_quality
            
            
            '''
            if self.intensity_percentage <= 0.85:
                if best_vns_quality.sum() < shaking_quality.sum():
                    return best_vns_solution, best_vns_quality
            else:
                if best_vns_quality.sum() < best_global_quality.sum():
                    return best_vns_solution, best_vns_quality
            '''
            
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
        while i < self.k_number:
            for shaking_ratio in range(3):
                shaking_solution, shaking_quality, idx_routes = self.shaking(best_global_solution.copy(), best_global_quality.copy(),
                                                                             shaking_ratio)
                vns_solution, vns_quality = self.VNS(shaking_solution, shaking_quality, best_global_quality, 
                                                     int((len(self.demands_array) / 3)), idx_routes)     

                if vns_quality.sum() < best_global_quality.sum():
                    # print(str(time.time() - start_time) + ': ' + str(vns_quality.sum()))
                    best_global_solution = vns_solution
                    best_global_quality = vns_quality
                    new_improve = True
                    
            if new_improve:
                i = 0
                new_improve = False 
            else:
                i += 1
            
            if (time.time() - start_time) >= max_time:
                break

        for k, route in enumerate(best_global_solution):
            temp_route = route[::-1]
            temp_quality = self.calculateSolutionQuality([temp_route])
            if temp_quality < best_global_quality[k]:
                best_global_solution[k] = temp_route.copy()
                best_global_quality[k] = temp_quality.copy()
        
        best_global_routes_arcs = self.generateRoutesArcs(best_global_solution)
        best_global_routes_arcs_weights = self.calculateRoutesWeights(best_global_solution)

        return best_global_solution, best_global_routes_arcs, best_global_quality, best_global_routes_arcs_weights