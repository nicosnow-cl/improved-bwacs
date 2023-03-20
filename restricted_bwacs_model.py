class RestrictedAntEMVRP:
    def __init__(self, depot, cluster, start, combinations_matrix, distances_matrix, demands_array, vehicle_capacity, tare, start_ant_on_best_nodes, q0):
        import numpy as np
        import random

        self.depot = depot
        self.cluster = cluster
        self.start = start
        self.combinations_matrix = combinations_matrix
        self.distances_matrix = distances_matrix
        self.demands_array = demands_array
        self.vehicle_capacity = vehicle_capacity
        self.tare = tare
        self.start_ant_on_best_nodes = start_ant_on_best_nodes
        self.q0 = q0
        self.np = np
        self.random = random

    def run(self):
        _cluster = [self.depot] + self.cluster
        unvisited_nodes = self.np.array(list(range(len(_cluster))))
        route_solution = []
        route_energy = 0
        vehicle_weight = self.tare

        if self.start_ant_on_best_nodes == 1:
            # Elección del primer nodo de manera aleatoria
            route_solution.append(self.depot)
            unvisited_nodes = unvisited_nodes[unvisited_nodes != 0]
            r = _cluster.index(self.start)
            route_solution.append(_cluster[r])
            unvisited_nodes = unvisited_nodes[unvisited_nodes != r]
            route_energy += self.distances_matrix[_cluster[self.depot]
                                                  ][_cluster[r]] * vehicle_weight
            vehicle_weight += self.demands_array[_cluster[r]]
        else:
            # Elección del primer nodo como depot
            r = self.depot
            route_solution.append(r)
            unvisited_nodes = unvisited_nodes[unvisited_nodes != 0]

        while unvisited_nodes.size:
            combination = self.combinations_matrix[r][unvisited_nodes]
            # probabilities = self.np.divide(combination, combination.sum())

            q = self.np.random.random(1)[0]
            if q <= self.q0:
                # s = unvisited_nodes[probabilities.argmax()]
                s = unvisited_nodes[combination.argmax()]
            else:
                # s = self.np.random.choice(unvisited_nodes, 1, p = probabilities)[0]
                # s = self.random.choices(unvisited_nodes, weights = probabilities, k = 1)[0]
                cum_sum = self.np.cumsum(combination)
                s = self.random.choices(
                    unvisited_nodes, cum_weights=cum_sum, k=1)[0]

            route_solution.append(_cluster[s])
            route_energy += self.distances_matrix[_cluster[r]
                                                  ][_cluster[s]] * vehicle_weight
            vehicle_weight += self.demands_array[_cluster[s]]

            unvisited_nodes = unvisited_nodes[unvisited_nodes != s]
            r = s

        route_solution.append(self.depot)
        route_energy += self.distances_matrix[_cluster[r]
                                              ][_cluster[self.depot]] * vehicle_weight

        return route_solution, route_energy


class RestrictedBWACS:
    def __init__(self,
                 instance,
                 max_nodes=9999,
                 ant_type='distance',
                 cluster_type='kmedoids',
                 metric='manhattan',
                 tare=0.15,
                 max_iterations=None,
                 max_ants=None,
                 start_ant_on_best_nodes=1,
                 alpha=1,
                 beta=1,
                 gamma=1,
                 delta=1,
                 eta=1,
                 mi=1,
                 pheromone_updating_strategy=1,
                 local_ant_update_pheromones=0,
                 best_iteration_update_pheromones=1,
                 best_global_update_pheromones=1,
                 penalize_worst_solution=1,
                 mutate_pheromones_matrix=0,
                 p=0.02,
                 Pm=0.3,
                 sigma=2,
                 q0=0.3,
                 Q=10,
                 heuristic_type=3,
                 ls_ant_solution=1,
                 ls_final_solution=0,
                 print_clusters=0,
                 output_sol_img='solution.png'
                 ):
        import numpy as np
        import random as random
        import math as math
        from itertools import permutations
        import matplotlib.pyplot as plt
        import pandas as pd
        from src.readers import ReaderCVRPLIB
        from copy import deepcopy

        '''
            ######################################
            ##  1. Parametros de la instancia.  ##
            ######################################
        '''
        self.instance = instance
        self.max_nodes = max_nodes
        self.ant_type = ant_type
        self.cluster_type = cluster_type
        self.metric = metric
        self.tare = tare
        self.max_iterations = max_iterations
        self.max_ants = max_ants
        self.heuristic_type = heuristic_type

        '''
            ######################################
            ##  2. Parametros clásicos de ACS.  ##
            ######################################
        '''
        self.start_ant_on_best_nodes = start_ant_on_best_nodes
        self.alpha = alpha
        self.beta = beta
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.eta = eta
        self.mi = mi
        self.pheromone_updating_strategy = pheromone_updating_strategy
        self.local_ant_update_pheromones = local_ant_update_pheromones
        self.best_iteration_update_pheromones = best_iteration_update_pheromones
        self.best_global_update_pheromones = best_global_update_pheromones
        self.penalize_worst_solution = penalize_worst_solution
        self.mutate_pheromones_matrix = mutate_pheromones_matrix
        self.p = p
        self.k_number = 0
        self.vehicle_capacity = 0
        self.depot = 0
        self.clients = []
        self.nodes = []
        self.demands = {}
        self.demands_array = None
        self.total_demand = 0
        self.loc_x = {}
        self.loc_y = {}
        self.coords_matrix = None
        self.distances_matrix = None
        self.energies_matrix = None
        self.saving_matrix = None
        self.combination_matrix = None
        self.pheromones_matrices = []

        '''
            ######################################
            ##  3. Parametros de BWACS.         ##
            ######################################
        '''
        self.tightness_ratio = 0
        self.t_min = 0.00001
        self.t_max = 0.0001
        self.Pm = Pm
        self.sigma = sigma
        self.Q = Q
        self.q0 = q0
        self.closest_nodes_in_clusters = []

        '''
            ######################################
            ##  4. Librerias, Modelos y Otros.  ##
            ######################################
        '''
        self.reader = ReaderCVRPLIB(self.instance, max_nodes=self.max_nodes)
        self.cluster_model = None
        self.clusters = []
        self.ant_model = None
        self.np = np
        self.pd = pd
        self.random = random
        self.math = math
        self.permutations = permutations
        self.plt = plt
        self.color_palette = []
        self.deepcopy = deepcopy
        self.ls_ant_solution = ls_ant_solution
        self.ls_final_solution = ls_final_solution
        self.print_clusters = print_clusters
        self.OUTPUT_SOLUTION_IMG = output_sol_img

    def printClusters(self, centers_list):
        self.plt.rc('font', size=16)
        self.plt.rc('figure', titlesize=18)
        self.plt.figure(figsize=(15, 15))

        for k, cluster in enumerate(self.clusters):
            for i in cluster:
                if i not in centers_list:
                    self.plt.plot(self.coords_matrix[i][0], self.coords_matrix[i][1], c=self.color_palette[k],
                                  marker='o', markersize=12)
                    self.plt.annotate('$q_{%d}=%d$' % (
                        i, self.demands_array[i]), (self.coords_matrix[i][0]+1, self.coords_matrix[i][1]-2))

        for k, j in enumerate(centers_list):
            if self.cluster_type == 'kmedoids':
                self.plt.plot(self.coords_matrix[j][0], self.coords_matrix[j]
                              [1], c=self.color_palette[k], marker='*', markersize=25)
                self.plt.annotate('$medoid_{%d}$' % (
                    k + 1), (self.coords_matrix[j][0]*0.1, self.coords_matrix[j][1]*(-0.2)))
            else:
                self.plt.plot(j[0], j[1], c=self.color_palette[k],
                              marker='*', markersize=30)
                self.plt.annotate('$centroid_{%d}$' % (
                    k + 1), (j[0]*0.1, j[1]*(-0.2)))

        self.plt.plot(
            self.loc_x[self.depot], self.loc_y[self.depot], c='r', marker='s', markersize=16)
        self.plt.annotate(
            'DEPOT', (self.loc_x[self.depot]+1, self.loc_y[self.depot]-2))
        # self.plt.savefig('cluster.png')
        self.plt.show()

    def printSolution(self, solution, routes_arcs, solution_energies, solution_distances):
        self.plt.rc('font', size=16)
        self.plt.rc('figure', titlesize=18)
        self.plt.figure(figsize=(15, 15))
        legend_content = []
        legend_colors = []

        for k, route in enumerate(solution):
            for i in route:
                if i != 0:
                    self.plt.plot(self.coords_matrix[i][0], self.coords_matrix[i]
                                  [1], c=self.color_palette[k], marker='o', markersize=17)
                    self.plt.annotate('$q_{%d}=%d$' % (
                        i, self.demands_array[i]), (self.coords_matrix[i][0], self.coords_matrix[i][1]-2))
            for x, y in routes_arcs[k]:
                self.plt.plot([self.loc_x[x], self.loc_x[y]], [
                              self.loc_y[x], self.loc_y[y]], c=self.color_palette[k], alpha=0.5, linewidth=4)
            legend_content.append('Route ' + str(k + 1) + '\nEnergy: ' + str(round(solution_energies[k], 2)) + '\nDistance: ' +
                                  str(round(solution_distances[k], 2)))
            legend_colors.append(self.color_palette[k])

        self.plt.plot(
            self.loc_x[self.depot], self.loc_y[self.depot], c='r', marker='s', markersize=17)
        self.plt.annotate(
            'DEPOT', (self.loc_x[self.depot]-1.5, self.loc_y[self.depot]-3))
        self.plt.title(label='Total Energy: ' + str(round(sum(solution_energies), 2)) + ', Total Distances: ' +
                       str(round(sum(solution_distances), 2)), fontweight=10, pad="2.0")
        self.plt.legend(legend_content, loc='lower center', bbox_to_anchor=(0.5, -0.1), ncol=10, labelcolor=legend_colors,
                        fontsize=10)
        self.plt.savefig(self.OUTPUT_SOLUTION_IMG)
        self.plt.show()

    def printDistancesMatrix(self):
        from IPython.display import display, HTML
        pd_distances_matrix = self.pd.DataFrame(self.distances_matrix)
        display(HTML(pd_distances_matrix.to_html()))

    def printEnergiesMatrix(self):
        from IPython.display import display, HTML
        pd_energies_matrix = self.pd.DataFrame(self.energies_matrix)
        display(HTML(pd_energies_matrix.to_html()))

    def printSavingMatrix(self):
        from IPython.display import display, HTML
        pd_saving_matrix = self.pd.DataFrame(self.saving_matrix)
        display(HTML(pd_saving_matrix.to_html()))

    def printCombinationMatrix(self):
        from IPython.display import display, HTML
        pd_combination_matrix = self.pd.DataFrame(self.combination_matrix)
        display(HTML(pd_combination_matrix.to_html()))

    def printPheromonesMatrix(self, k=None):
        from IPython.display import display, HTML

        if k == None:
            for k in range(len(self.clusters)):
                pd_pheromones_matrix = self.pd.DataFrame(
                    self.np.power(self.pheromones_matrices[k], self.alpha))
                display(HTML(pd_pheromones_matrix.to_html()))
        else:
            pd_pheromones_matrix = self.pd.DataFrame(
                self.np.power(self.pheromones_matrices[k], self.alpha))
            display(HTML(pd_pheromones_matrix.to_html()))

    def printArcsMatrix(self, arcs_matrix):
        from IPython.display import display, HTML
        pd_arcs_matrix = self.pd.DataFrame(arcs_matrix)
        display(HTML(pd_arcs_matrix.to_html()))

    def setInitialParameters(self):
        import matplotlib.cm as cm
        if self.max_iterations == None:
            self.max_iterations = self.math.ceil(
                (self.math.sqrt(len(self.nodes)))*(1/len(self.nodes))*100)
        if self.max_ants == None:
            self.max_ants = self.math.ceil(
                self.max_iterations * self.math.sqrt(self.max_iterations))
        self.nodes, self.demands_array = [
            self.depot] + self.clients, self.np.array(list(self.demands.values()))
        self.coords_matrix = self.np.array(
            [(self.loc_x[i], self.loc_y[i]) for i in self.nodes])
        self.color_palette = cm.jet(self.np.linspace(0, 1, self.k_number + 1))
        self.tare = self.vehicle_capacity * self.tare

    def createDistancesMatrix(self):
        self.distances_matrix = self.np.zeros(
            (len(self.nodes), len(self.nodes)))
        ord_ = 1 if self.metric == 'manhattan' else 2

        for i in self.nodes:
            for j in self.nodes:
                if i != j:
                    self.distances_matrix[i][j] = self.np.linalg.norm(
                        (self.coords_matrix[i] - self.coords_matrix[j]), ord=ord_)

    def createEnergiesMatrix(self):
        self.energies_matrix = self.np.zeros(
            (len(self.nodes), len(self.nodes)))

        for i in self.nodes:
            if i != self.depot:
                self.energies_matrix[i] = self.np.multiply(
                    self.distances_matrix[i], (self.demands_array[i] + self.tare))
            else:
                self.energies_matrix[i] = self.np.multiply(
                    self.distances_matrix[i], self.tare)

    def createSavingMatrix(self):
        self.saving_matrix = self.np.zeros((len(self.nodes), len(self.nodes)))

        for i in self.nodes[self.depot + 1:]:
            for j in self.nodes[self.depot + 1:]:
                if i != j:
                    s_i0 = self.distances_matrix[i][self.depot]
                    s_0j = self.distances_matrix[self.depot][j]
                    s_ij = self.distances_matrix[i][j]
                    saving = s_i0 + s_0j - s_ij
                    self.saving_matrix[i][j] = saving

        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        scaler.fit(self.saving_matrix)
        self.saving_matrix = scaler.transform(self.saving_matrix)

        for i in self.nodes:
            if i != self.depot:
                self.saving_matrix[self.depot][i] = 1
                self.saving_matrix[i][self.depot] = 1

    def createSavingEnergiesMatrix(self):
        self.saving_energies_matrix = self.np.zeros(
            (len(self.nodes), len(self.nodes)))

        for i in self.nodes[self.depot + 1:]:
            for j in self.nodes[self.depot + 1:]:
                if i != j:
                    s_i0 = self.energies_matrix[i][self.depot]
                    s_0j = self.energies_matrix[self.depot][j]
                    s_ij = self.energies_matrix[i][j]
                    saving = s_i0 + s_0j - s_ij
                    self.saving_energies_matrix[i][j] = saving

        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        scaler.fit(self.saving_energies_matrix)
        self.saving_energies_matrix = scaler.transform(
            self.saving_energies_matrix)

        for i in self.nodes:
            if i != self.depot:
                self.saving_energies_matrix[self.depot][i] = 1
                self.saving_energies_matrix[i][self.depot] = 1

    def createCapacityUtilizationMatrix(self):
        self.cu_matrix = self.np.zeros((len(self.nodes), len(self.nodes)))

        max_demand = self.demands_array.max()
        for i in self.nodes:
            for j in self.nodes:
                self.cu_matrix[i][j] = (
                    self.demands_array[i] + self.demands_array[j]) / self.vehicle_capacity

    def combineHeuristicMatrices(self):
        self.combination_matrix = self.np.zeros(
            (len(self.nodes), len(self.nodes)))

        if (self.heuristic_type == 0):
            _distance_matrix = self.np.divide(1, self.distances_matrix)
            self.combination_matrix = self.np.power(
                _distance_matrix, self.beta)
        elif (self.heuristic_type == 1):
            _energy_matrix = self.np.divide(1, self.energies_matrix)
            self.combination_matrix = self.np.power(_energy_matrix, self.gamma)
        elif (self.heuristic_type == 2):
            _distance_matrix = self.np.divide(1, self.distances_matrix)
            _energy_matrix = self.np.divide(1, self.energies_matrix)
            self.combination_matrix = self.np.multiply(self.np.power(
                _distance_matrix, self.beta), self.np.power(_energy_matrix, self.gamma))
        elif (self.heuristic_type == 3):
            self.combination_matrix = self.np.power(
                self.saving_matrix, self.delta)
        elif (self.heuristic_type == 4):
            self.combination_matrix = self.np.power(
                self.saving_energies_matrix, self.eta)
        elif (self.heuristic_type == 5):
            _distance_matrix = self.np.power(
                self.np.divide(1, self.distances_matrix), self.beta)
            _saving_matrix = self.np.power(self.saving_matrix, self.delta)
            self.combination_matrix = self.np.multiply(
                _distance_matrix, _saving_matrix)
        elif (self.heuristic_type == 6):
            _energy_matrix = self.np.power(
                self.np.divide(1, self.energies_matrix), self.gamma)
            _saving_energies_matrix = self.np.power(
                self.saving_energies_matrix, self.eta)
            self.combination_matrix = self.np.multiply(
                _energy_matrix, _saving_energies_matrix)
        elif (self.heuristic_type == 7):
            _distance_matrix = self.np.power(
                self.np.divide(1, self.distances_matrix), self.beta)
            _energy_matrix = self.np.power(
                self.np.divide(1, self.energies_matrix), self.gamma)
            _saving_matrix = self.np.power(self.saving_matrix, self.delta)
            self.combination_matrix = self.np.multiply(
                _distance_matrix, _energy_matrix)
            self.combination_matrix = self.np.multiply(
                self.combination_matrix, _saving_matrix)
        elif (self.heuristic_type == 8):
            _distance_matrix = self.np.power(
                self.np.divide(1, self.distances_matrix), self.beta)
            _energy_matrix = self.np.power(
                self.np.divide(1, self.energies_matrix), self.gamma)
            _saving_energies_matrix = self.np.power(
                self.saving_energies_matrix, self.eta)
            self.combination_matrix = self.np.multiply(
                _distance_matrix, _energy_matrix)
            self.combination_matrix = self.np.multiply(
                self.combination_matrix, _saving_energies_matrix)
        elif (self.heuristic_type == 9):
            _distance_matrix = self.np.power(
                self.np.divide(1, self.distances_matrix), self.beta)
            _energy_matrix = self.np.power(
                self.np.divide(1, self.energies_matrix), self.gamma)
            _saving_matrix = self.np.power(self.saving_matrix, self.delta)
            _cu_matrix = self.np.power(self.cu_matrix, self.mi)
            self.combination_matrix = self.np.multiply(
                _distance_matrix, _energy_matrix)
            self.combination_matrix = self.np.multiply(
                self.combination_matrix, _saving_matrix)
            self.combination_matrix = self.np.multiply(
                self.combination_matrix, _cu_matrix)
        elif (self.heuristic_type == 10):
            _distance_matrix = self.np.power(
                self.np.divide(1, self.distances_matrix), self.beta)
            _energy_matrix = self.np.power(
                self.np.divide(1, self.energies_matrix), self.gamma)
            _saving_energies_matrix = self.np.power(
                self.saving_energies_matrix, self.eta)
            _cu_matrix = self.np.power(self.cu_matrix, self.mi)
            self.combination_matrix = self.np.multiply(
                _distance_matrix, _energy_matrix)
            self.combination_matrix = self.np.multiply(
                self.combination_matrix, _saving_energies_matrix)
            self.combination_matrix = self.np.multiply(
                self.combination_matrix, _cu_matrix)
        elif (self.heuristic_type == 11):
            _distance_matrix = self.np.power(
                self.np.divide(1, self.distances_matrix), self.beta)
            _energy_matrix = self.np.power(
                self.np.divide(1, self.energies_matrix), self.gamma)
            _saving_matrix = self.np.power(self.saving_matrix, self.delta)
            _saving_energies_matrix = self.np.power(
                self.saving_energies_matrix, self.eta)
            _cu_matrix = self.np.power(self.cu_matrix, self.mi)
            self.combination_matrix = self.np.multiply(
                _distance_matrix, _energy_matrix)
            self.combination_matrix = self.np.multiply(
                self.combination_matrix, _saving_matrix)
            self.combination_matrix = self.np.multiply(
                self.combination_matrix, _saving_energies_matrix)
            self.combination_matrix = self.np.multiply(
                self.combination_matrix, _cu_matrix)

        self.combination_matrix[self.np.isnan(self.combination_matrix)] = 0
        self.combination_matrix[self.combination_matrix == self.np.inf] = 0

    def createClustersArcsMatrix(self):
        self.clusters_arcs = self.np.zeros((len(self.nodes), len(self.nodes)))

        for cluster in self.clusters:
            cluster_arcs = list(self.permutations(cluster, 2))
            for i, j in cluster_arcs:
                self.clusters_arcs[i][j] = 1

    def getClosestNodesFromDepot(self):
        for cluster in self.clusters:
            closests_nodes_to_depot = self.energies_matrix[self.depot][cluster].argsort(
            )
            self.closest_nodes_in_clusters.append(
                cluster[closests_nodes_to_depot[0]])

    def createCandidateList(self, k, best_nodes=[]):
        max_nodes = int((len(self.clusters[k]) * 50) / 100)
        candidate_list = [] + [self.closest_nodes_in_clusters[k]]
        candidate_list = candidate_list + best_nodes

        possible_random_nodes = [
            node for node in self.clusters[k] if node not in candidate_list]
        random_nodes = self.random.sample(
            possible_random_nodes, max_nodes - len(candidate_list))

        candidate_list = candidate_list + random_nodes
        return candidate_list

    def setInitialPheromones(self):
        self.pheromones_matrices = [self.np.full(
            (len(cluster) + 1, len(cluster) + 1), self.t_max[k] / 2) for k, cluster in enumerate(self.clusters)]

    def restartPhermonesMatrix(self, k):
        self.pheromones_matrices[k].fill(self.t_max[k] / 2)

    def calculateEnergies(self, solution):
        routes_energies = self.np.zeros(len(solution))

        for k, route in enumerate(solution):
            route_energy = 0

            for pos, i in enumerate(route):
                if pos == 0:
                    vehicle_weight = self.tare
                    before_node = self.depot
                else:
                    route_energy += self.distances_matrix[before_node][i] * \
                        vehicle_weight
                    vehicle_weight += self.demands_array[i]
                    before_node = i

            routes_energies[k] = route_energy

        return routes_energies

    def calculateDistances(self, solution):
        routes_distances = self.np.zeros(len(solution))

        for k, route in enumerate(solution):
            route_distance = 0

            for pos, i in enumerate(route):
                if pos == 0:
                    before_node = self.depot
                else:
                    route_distance += self.distances_matrix[before_node][i]
                    before_node = i

            routes_distances[k] = route_distance

        return routes_distances

    def generateArcs(self, solution):
        solution_arcs = []
        for route in solution:
            route_arcs = []
            for pos, i in enumerate(route):
                if pos == 0:
                    before_node = i
                else:
                    route_arcs.append((before_node, i))
                    before_node = i
            solution_arcs.append(route_arcs)
        return solution_arcs

    def calculateTminTmax(self):
        self.t_min = []
        self.t_max = []

        for k, cluster in enumerate(self.clusters):
            energies = self.distances_matrix[self.depot][cluster] * \
                self.demands_array[cluster]
            k_t_min = self.Q / (energies.sum() * 2)
            k_t_max = (self.Q / energies.sum()) * self.k_number

            self.t_min.append(k_t_min)
            self.t_max.append(k_t_max)

        print(self.t_min)
        print(self.t_max)

    def evaporatePheromones(self, k):
        self.pheromones_matrices[k] *= (1 - self.p)

    def ruleUpdatePheromones(self, k, ant_solution, ant_quality, max_ants):
        if self.pheromone_updating_strategy == 0:
            pheromones_quantity = self.Q / ant_quality
            ant_solution_arcs = self.generateSolutionArcs(ant_solution)

            for route_arcs in ant_solution_arcs:
                for i, j in route_arcs:
                    self.pheromones_matrices[k][i][j] += pheromones_quantity / max_ants
        else:
            ant_solution_arcs = self.generateSolutionArcs(ant_solution)
            total_energy_by_route = []
            energy_arcs_by_route = []
            for route_arcs in ant_solution_arcs:
                energy_by_arc = {}
                route_total_energy = 0
                for i, j in route_arcs:
                    route_total_energy += self.energies_matrix[i][j]
                    energy_by_arc[(i, j)] = self.energies_matrix[i][j]
                energy_arcs_by_route.append(energy_by_arc)
                total_energy_by_route.append(route_total_energy)

            for k, route_energy in enumerate(energy_arcs_by_route):
                route_total_energy = total_energy_by_route[k]

                for i, j in route_energy:
                    numerator = route_total_energy - route_energy[(i, j)]
                    denominator = len(ant_solution[k]) * route_total_energy
                    local_arc_quality = numerator / denominator
                    pheromones_quantity = (
                        self.Q / route_total_energy) * local_arc_quality
                    self.pheromones_matrices[k][i][j] += pheromones_quantity / max_ants

    def currentWorstUpdatePheromones(self, k, cw_cluster_solution, cw_index_arcs, gb_cluster_index_arcs):
        for pos, arc in enumerate(cw_index_arcs):
            if arc not in gb_cluster_index_arcs:
                self.pheromones_matrices[k][arc[0]][arc[1]] *= (1 - self.p)

    def currentBestUpdatePheromones(self, k, cb_cluster_solution, cb_route_arcs, cb_index_arcs, cb_quality):
        if self.pheromone_updating_strategy == 0:
            pheromones_quantity = self.Q / cb_quality

            for i, j in cb_index_arcs:
                self.pheromones_matrices[k][i][j] += pheromones_quantity
        else:
            energy_by_arc = {}
            route_total_energy = 0
            for i, j in cb_route_arcs:
                route_total_energy += self.energies_matrix[i][j]
                energy_by_arc[(i, j)] = self.energies_matrix[i][j]

            for pos, arc in enumerate(cb_route_arcs):
                numerator = route_total_energy - \
                    energy_by_arc[(arc[0], arc[1])]
                denominator = len(cb_cluster_solution) * route_total_energy
                local_arc_quality = numerator / denominator
                pheromones_quantity = local_arc_quality
                self.pheromones_matrices[k][cb_index_arcs[pos][0]
                                            ][cb_index_arcs[pos][1]] += pheromones_quantity

    def globalUpdatePheromones(self, k, gb_cluster_solution, gb_cluster_arcs, gb_cluster_index_arcs, gb_quality):
        if self.pheromone_updating_strategy == 0:
            pheromones_quantity = self.Q / gb_quality

            for i, j in gb_cluster_index_arcs:
                self.pheromones_matrices[k][i][j] += pheromones_quantity
        else:
            energy_by_arc = {}
            route_total_energy = 0
            for i, j in gb_cluster_arcs:
                route_total_energy += self.energies_matrix[i][j]
                energy_by_arc[(i, j)] = self.energies_matrix[i][j]

            for pos, arc in enumerate(gb_cluster_arcs):
                numerator = route_total_energy - \
                    energy_by_arc[(arc[0], arc[1])]
                denominator = len(gb_cluster_solution) * route_total_energy
                local_arc_quality = numerator / denominator
                pheromones_quantity = local_arc_quality
                self.pheromones_matrices[k][gb_cluster_index_arcs[pos][0]
                                            ][gb_cluster_index_arcs[pos][1]] += pheromones_quantity

    def mutatePheromonesMatrix(self, k, cb_solution_arcs, iteration, last_restart):
        cb_arcs_matrix = self.np.zeros((len(self.nodes), len(self.nodes)))
        for route_arcs in cb_solution_arcs:
            for i, j in route_arcs:
                cb_arcs_matrix[i][j] = 1

        t_threshold = (self.pheromones_matrix[cb_arcs_matrix == 1].mean())
        mutation_1 = (iteration - last_restart) / \
            (self.max_iterations - last_restart)
        mutation_2 = self.sigma * t_threshold
        mutation = mutation_1 * mutation_2
        mutation /= self.k_number

        for i in range(len([self.depot] + self.clusters[k])):
            z = self.np.random.random()  # Variable con valor entre 0 y 1

            if z <= self.Pm:  # Pm es una constante que se define al comienzo
                _ = self.np.random.randint(2)

                if _ == 0:  # Cuando es 0 se suma en toda la fila i el valor mutacion
                    self.pheromones_matrices[k][i] += mutation
                else:  # Cuando es 1 se resta en toda la fila i el valor mutacion
                    self.pheromones_matrices[k][i] -= mutation

    def generateRouteIndexArcs(self, k, route_solution):
        _cluster = [self.depot] + self.clusters[k]
        route_arcs = []
        index_arcs = []

        for pos, i in enumerate(route_solution):
            if pos == 0:
                before_node = i
            else:
                route_arcs.append((before_node, i))
                index_arcs.append(
                    (_cluster.index(before_node), _cluster.index(i)))
                before_node = i

        return route_arcs, index_arcs

    def generateSolutionArcs(self, solution):
        solution_arcs = []

        for route in solution:
            route_arcs = []

            for pos, i in enumerate(route):
                if pos == 0:
                    before_node = i
                else:
                    route_arcs.append((before_node, i))
                    before_node = i

            solution_arcs.append(route_arcs)

        return solution_arcs

    def solve(self):
        import time
        from copy import deepcopy

        print('• INICIALIZANDO ALGORITMO •')
        print('------------------------------\n')

        start_time = time.time()
        # Leemos los datos de la instancia
        self.depot, self.clients, self.loc_x, self.loc_y, self.demands, self.total_demand, self.vehicle_capacity, self.k_number, self.tightness_ratio = self.reader.read()
        print('• PARAMETROS DE LA INSTANCIA: \n')
        print('    > Nodos y demandas: ' + str(self.demands))
        print('    > Demanda total: ' + str(self.total_demand))
        print('    > Capacidad vehiculo: ' + str(self.vehicle_capacity))
        print('    > K-Optimo: ' + str(self.k_number))
        print('    > Tightnessratio: ' + str(self.tightness_ratio) + '\n')

        self.setInitialParameters()
        self.createDistancesMatrix()
        self.createEnergiesMatrix()
        self.createSavingMatrix()
        self.createSavingEnergiesMatrix()
        self.createCapacityUtilizationMatrix()
        self.combineHeuristicMatrices()

        if self.cluster_type == 'kmedoids':  # cluster_type nos permite elegir si queremos realizar un cluster con KMedoids o Kmeans
            from src.clustering import KMedoidsEMVRP
            self.cluster_model = KMedoidsEMVRP(self.depot, self.nodes.copy(), self.demands_array.copy(), self.distances_matrix.copy(),
                                               self.vehicle_capacity, self.k_number)
        else:
            from src.clustering import KMeansEMVRP
            self.cluster_model = KMeansEMVRP(self.depot, self.nodes.copy(), self.demands_array.copy(), self.coords_matrix.copy(),
                                             self.distances_matrix.copy(), self.vehicle_capacity, self.k_number, self.metric)

        print('• INICIANDO CLUSTERIZACIÓN: \n')
        print('    > CLUSTER TYPE: ' + self.cluster_type.upper())
        self.clusters, clusters_total_cost, centers_list, unassigned_nodes = self.cluster_model.run()
        if len(unassigned_nodes) > 0:
            print(
                f'Ha ocurrido un ERROR: The following nodes are unassigned: {str(unassigned_nodes)}')
            raise Exception(
                f'Ha ocurrido un ERROR: The following nodes are unassigned: {str(unassigned_nodes)}')

        print('    > CLUSTERS')
        for k in range(self.k_number):
            print('         - Cluster ' + str(k) + ': ' + str(self.clusters[k]) + ', con demanda final: '
                  + str(sum([self.demands_array[i] for i in self.clusters[k]])))
        print('         - Nodos totales: ' +
              str(sum([len(cluster) for cluster in self.clusters])))
        if self.print_clusters == 1:
            self.printClusters(centers_list)

        print('• INICIANDO ACO(BWACS): ')

        self.getClosestNodesFromDepot()
        self.calculateTminTmax()
        self.setInitialPheromones()

        # self.printDistancesMatrix()
        # self.printEnergiesMatrix()
        # self.printSavingMatrix()
        # self.printCombinationMatrix()
        # self.printPheromonesMatrix()

        gb_solution = [[] for k in range(len(self.clusters))]
        gb_solution_arcs = [[] for k in range(len(self.clusters))]
        gb_solution_index_arcs = [[] for k in range(len(self.clusters))]
        gb_clusters_energies = self.np.full(len(self.clusters), self.np.inf)
        gb_clusters_distances = self.np.full(len(self.clusters), self.np.inf)

        last_restart = 0
        stagnation = 0

        for k in range(len(self.clusters)):
            print('\n    • CLUSTER ' + str(k + 1))

            candidate_list = self.createCandidateList(k)
            for iteration in range(self.max_iterations):
                print('        > Iteración ' + str(iteration + 1))

                mean_quality = 0

                cb_cluster_solution = None
                cb_route_arcs = []
                cb_index_arcs = []
                cb_route_energy = self.np.inf
                cb_route_distance = self.np.inf

                cw_cluster_solution = None
                cw_route_arcs = []
                cw_index_arcs = []
                cw_route_energy = 0
                cw_route_distance = 0

                # self.printPheromonesMatrix(k)
                _cluster = [self.depot] + self.clusters[k]
                self._pheromones_matrix = self.np.power(
                    self.pheromones_matrices[k], self.alpha)
                self._combinations_matrix = self.combination_matrix[self.np.ix_(
                    _cluster, _cluster)]
                self._combinations_matrix = self.np.multiply(
                    self._pheromones_matrix, self._combinations_matrix)

                for ant in candidate_list:
                    best_k_quality = [self.np.inf for i in range(
                        int((len(self.clusters[k]) * 20) / 100))]
                    best_k_nodes = [0 for i in range(
                        int((len(self.clusters[k]) * 20) / 100))]

                    self.ant_model = RestrictedAntEMVRP(self.depot, self.clusters[k], ant, self._combinations_matrix, self.distances_matrix, self.demands_array,
                                                        self.vehicle_capacity, self.tare, self.start_ant_on_best_nodes, self.q0)
                    ant_solution, route_energy = self.ant_model.run()
                    mean_quality += route_energy

                    # self.ruleUpdatePheromones(k, ant_solution, route_arcs, route_arcs_weights, route_energy)
                    # print('            - Hormiga ' + str(ant + 1) + ': ENER. ' + str(route_energy))

                    if route_energy < cb_route_energy:
                        cb_cluster_solution = ant_solution
                        cb_route_energy = route_energy
                        cb_route_arcs, cb_index_arcs = self.generateRouteIndexArcs(
                            k, cb_cluster_solution)
                    elif route_energy > cw_route_energy:
                        cw_cluster_solution = ant_solution
                        cw_route_energy = route_energy
                        cw_route_arcs, cw_index_arcs = self.generateRouteIndexArcs(
                            k, cb_cluster_solution)

                    for idx, quality in enumerate(best_k_quality):
                        if (route_energy < quality) and (ant not in self.closest_nodes_in_clusters):
                            best_k_quality[idx] = route_energy
                            best_k_nodes[idx] = ant
                            break

                candidate_list = self.createCandidateList(k, best_k_nodes)
                mean_quality = mean_quality / len(candidate_list)
                # print('            - Solution´s Mean en cluster ' + str(k + 1) + ' : ' + str(mean_quality))

                num_arcs_t_max = self.np.count_nonzero(
                    self.pheromones_matrices[k][1:, 1:] >= self.t_max[k])
                print(num_arcs_t_max)

                if self.ls_ant_solution == 1:
                    from src.local_search import RestrictedLocalGVNS
                    local_search_model = RestrictedLocalGVNS(self.depot, self.clusters[k], cb_cluster_solution, cb_route_energy,
                                                             self.distances_matrix, self.demands_array, self.tare, self.vehicle_capacity)
                    ls_solution, ls_energy = local_search_model.improve()

                    if ls_energy < cb_route_energy:
                        cb_cluster_solution = ls_solution
                        cb_route_energy = ls_energy
                        cb_route_distance = 0
                        cb_route_arcs, cb_index_arcs = self.generateRouteIndexArcs(
                            k, ls_solution)

                if cb_route_energy < gb_clusters_energies[k]:
                    print('            - Mejor costo para cluster ' +
                          str(k + 1) + ': ' + str(cb_route_energy))
                    gb_solution[k] = cb_cluster_solution
                    gb_solution_arcs[k] = cb_route_arcs
                    gb_solution_index_arcs[k] = cb_index_arcs
                    gb_clusters_energies[k] = cb_route_energy
                    gb_clusters_distances[k] = cb_route_distance
                    stagnation = 0
                else:
                    stagnation += 1

                if (stagnation >= int((self.max_iterations * 30) / 100)):
                    # print('            !!! Reinicializando Matriz de feromonas !!!')
                    self.restartPhermonesMatrix(k)
                    last_restart = iteration
                    stagnation = 0
                    # self.currentBestUpdatePheromones(k, cb_cluster_solution, cb_route_arcs, cb_index_arcs, cb_route_energy)
                    # self.globalUpdatePheromones(k, gb_solution[k], gb_solution_arcs[k], gb_solution_index_arcs[k], gb_clusters_energies[k])
                else:
                    self.evaporatePheromones(k)
                    self.currentWorstUpdatePheromones(
                        k, cw_cluster_solution, cw_index_arcs, gb_solution_index_arcs[k])
                    self.currentBestUpdatePheromones(
                        k, cb_cluster_solution, cb_route_arcs, cb_index_arcs, cb_route_energy)
                    self.globalUpdatePheromones(
                        k, gb_solution[k], gb_solution_arcs[k], gb_solution_index_arcs[k], gb_clusters_energies[k])

                    self.pheromones_matrices[k][self.pheromones_matrices[k]
                                                < self.t_min[k]] = self.t_min[k]
                    self.pheromones_matrices[k][self.pheromones_matrices[k]
                                                > self.t_max[k]] = self.t_max[k]

        print('\n RUTAS OBTENIDAS POR LA COLONIA')
        for k, route in enumerate(gb_solution):
            print('> RUTA ' + str(k) + ': ' + str(route) +
                  ', con demanda final: ' + str(self.demands_array[route].sum()))
        print('> Energia total final: ' + str(gb_clusters_energies.sum()))
        print('> Distancia total final: ' + str(gb_clusters_distances.sum()))
        print("\n--- %s seconds ---" % (time.time() - start_time))
        arcs = self.generateArcs(gb_solution)

        if self.ls_final_solution == 1:
            from src.local_search import RestrictedGlobalGVNS
            local_search_model = RestrictedGlobalGVNS(self.depot, gb_solution, gb_clusters_energies, self.distances_matrix,
                                                      self.demands_array, self.tare, self.vehicle_capacity, self.k_number)
            ls_solution, ls_energies = local_search_model.improve()
            ls_distances = self.calculateDistances(ls_solution)
            print('\n RUTAS OBENIDAS DESPUES DE LS')
            for k, route in enumerate(ls_solution):
                print('> RUTA ' + str(k) + ': ' + str(route) +
                      ', con demanda final: ' + str(self.demands_array[route].sum()))
            print('> Energia total final: ' + str(ls_energies.sum()))
            print("\n--- %s seconds ---" % (time.time() - start_time))
            ls_arcs = self.generateArcs(ls_solution)
            gb_clusters_energies = ls_energies
            gb_clusters_distances = ls_distances

        return sum(gb_clusters_energies), gb_solution, sum(gb_clusters_distances), time.time() - start_time
