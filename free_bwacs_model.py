class FreeAntEMVRP_1:
    def __init__(self, depot, nodes, start, combinations_matrix, distances_matrix, demands_array, vehicle_capacity, tare, start_ant_on_best_nodes, q0):
        import numpy as np
        import random

        self.depot = depot
        self.nodes = nodes
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
        unvisited_nodes = self.np.array(self.nodes)
        routes_solution = []
        routes_energies = []

        while unvisited_nodes.size:
            route_solution = []
            route_energy = 0

            vehicle_capacity = 0
            vehicle_weight = self.tare

            if self.start_ant_on_best_nodes == 1:
                # Elección del primer nodo como un nodo distinto para cada hormiga
                route_solution.append(self.depot)
                if not routes_solution:
                    r = self.start
                else:
                    r = self.random.choice(unvisited_nodes)
                unvisited_nodes = unvisited_nodes[unvisited_nodes != r]
                route_solution.append(r)
                route_energy += self.distances_matrix[self.depot][r] * \
                    vehicle_weight
                vehicle_weight += self.demands_array[r]
                vehicle_capacity += self.demands_array[r]
            else:
                # Elección del primer nodo como depot
                r = self.depot
                route_solution.append(self.depot)

            valid_nodes = unvisited_nodes

            while valid_nodes.size:
                combination = self.combinations_matrix[r][valid_nodes]
                # probabilities = self.np.divide(combination, combination.sum())

                q = self.np.random.random(1)[0]
                if q <= self.q0:
                    # s = valid_nodes[probabilities.argmax()]
                    s = valid_nodes[combination.argmax()]
                else:
                    # s = self.np.random.choice(valid_nodes, p = probabilities)
                    # s = self.random.choices(valid_nodes, weights = probabilities, k = 1)[0]
                    cum_sum = self.np.cumsum(combination)
                    s = self.random.choices(
                        valid_nodes, cum_weights=cum_sum, k=1)[0]

                unvisited_nodes = unvisited_nodes[unvisited_nodes != s]
                route_solution.append(s)
                route_energy += self.distances_matrix[r][s] * vehicle_weight
                r = s
                vehicle_weight += self.demands_array[s]
                vehicle_capacity += self.demands_array[s]

                # _demands_by_node = self.demands_array[unvisited_nodes] + vehicle_capacity
                # valid_nodes = unvisited_nodes[_demands_by_node <= self.vehicle_capacity]

                max_node = int(len(self.nodes)/3)
                _demands_by_node = self.demands_array[unvisited_nodes] + \
                    vehicle_capacity
                valid_unvisited_nodes = unvisited_nodes[_demands_by_node <=
                                                        self.vehicle_capacity]
                valid_unvisited_nodes_sorted = self.distances_matrix[r][valid_unvisited_nodes].argsort()[
                    :max_node]
                valid_nodes = valid_unvisited_nodes[valid_unvisited_nodes_sorted]

            route_solution.append(self.depot)
            route_energy += self.distances_matrix[r][self.depot] * \
                vehicle_weight
            routes_solution.append(route_solution)
            routes_energies.append(route_energy)

        return routes_solution, routes_energies


class FreeBWACS:
    def __init__(self,
                 instance='',
                 max_nodes=9999,
                 cluster_type='kmeans',
                 metric='euclidian',
                 tare_percentage=0.15,
                 max_iterations=None,
                 total_ant_divider=2,
                 only_k_optimum=True,
                 start_ant_on_best_nodes=True,
                 heuristic_type=3,
                 alpha=1,
                 beta=1,
                 gamma=1,
                 delta=1,
                 eta=1,
                 mi=1,
                 pheromone_updating_strategy=0,
                 local_ant_update_pheromones=False,
                 best_iteration_ant_update_pheromones=True,
                 best_global_ant_update_pheromones=True,
                 penalize_worst_solution=True,
                 mutate_pheromones_matrix=False,
                 p=0.02,
                 Pm=0.2,
                 sigma=2,
                 l0=0.2,
                 H=1,
                 ls_ant_solution=False,
                 ls_best_iteration=False,
                 ls_best_global=True,
                 use_normalized_matrix=False,
                 print_instance=False,
                 print_clusters=False,
                 print_solution=False,
                 print_distance_matrix=False,
                 print_energy_matrix=False,
                 print_distance_saving_matrix=False,
                 print_energy_saving_matrix=False,
                 print_combination_matrix=False,
                 print_pheromone_matrix=False,
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
        self.INSTANCE = instance
        self.MAX_NODES = max_nodes
        self.CLUSTER_TYPE = cluster_type
        self.METRIC = metric
        self.TARE = 0
        self.TARE_PERCENTAGE = tare_percentage
        self.MAX_ITERATIONS = max_iterations
        self.MAX_ANTS = None
        self.TOTAL_ANT_DIVIDER = total_ant_divider
        self.ONLY_K_OPTIMUM = only_k_optimum
        self.HEURISTIC_TYPE = heuristic_type

        '''
            ######################################
            ##  2. Parametros clásicos de ACS.  ##
            ######################################
        '''
        self.START_ANT_ON_BEST_NODES = start_ant_on_best_nodes
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.eta = eta
        self.mi = mi
        self.PHEROMONE_UPDATING_STRATEGY = pheromone_updating_strategy
        self.LOCAL_ANT_UPDATE_PHEROMONES = local_ant_update_pheromones
        self.BEST_ITERATION_ANT_UPDATE_PHEROMONES = best_iteration_ant_update_pheromones
        self.BEST_GLOBAL_ANT_UPDATE_PHEROMONES = best_global_ant_update_pheromones
        self.PENALIZE_WORST_SOLUTION = penalize_worst_solution
        self.MUTATE_PHEROMONES_MATRIX = mutate_pheromones_matrix
        self.USE_NORMALIZED_MATRIX = use_normalized_matrix
        self.p = p
        self.K_NUMBER = 0
        self.VEHICLE_CAPACITY = 0
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
        self.distances_matrix_normalized = None
        self.energies_matrix = None
        self.energies_matrix_normalized = None
        self.saving_matrix = None
        self.saving_matrix_normalized = None
        self.combination_matrix = None
        self.pheromones_matrix = None

        '''
            ######################################
            ##  3. Parametros de BWACS.         ##
            ######################################
        '''
        self.tightness_ratio = 0
        self.t_min = 0.00001
        self.t_max = 0.00002
        self.t_0 = 0.000015
        self.Pm = Pm
        self.sigma = sigma
        self.H = H
        self.l0 = l0
        self.clusters_arcs = []
        self.closest_nodes_in_clusters = []

        '''
            ######################################
            ##  4. Librerias, Modelos y Otros.  ##
            ######################################
        '''
        self.reader = ReaderCVRPLIB(self.INSTANCE, max_nodes=self.MAX_NODES)
        self.cluster_model = None
        self.clusters = []
        self.np = np
        self.pd = pd
        self.random = random
        self.math = math
        self.permutations = permutations
        self.plt = plt
        self.color_palette = []
        self.deepcopy = deepcopy
        self.LS_ANT_SOLUTION = ls_ant_solution
        self.LS_BEST_ITERATION = ls_best_iteration
        self.LS_BEST_GLOBAL = ls_best_global
        self.PRINT_INSTANCE = print_instance
        self.PRINT_CLUSTERS = print_clusters
        self.PRINT_SOLUTION = print_solution
        self.PRINT_DISTANCE_MATRIX = print_distance_matrix
        self.PRINT_ENERGY_MATRIX = print_energy_matrix
        self.PRINT_DISTANCE_SAVING_MATRIX = print_distance_saving_matrix
        self.PRINT_ENERGY_SAVING_MATRIX = print_energy_saving_matrix
        self.PRINT_COMBINATION_MATRIX = print_combination_matrix
        self.PRINT_PHEROMONE_MATRIX = print_pheromone_matrix
        self.OUTPUT_SOLUTION_IMG = output_sol_img

    '''
        ############################################
        ##  1. FUNCTIONS SECTION                  ##
        ## Here are all the functions defined for ##
        ## the operation of the main algorithm    ##
        ############################################
    '''

    '''
         1.1 PRINT FUNCTIONS: 
         These functions allow the program to graph different models of the problem.
    '''
    # This function print all the nodes in a 2-D cartesian plane.

    def printInstance(self):
        self.plt.rc('font', size=16)
        self.plt.rc('figure', titlesize=18)
        self.plt.figure(figsize=(15, 15))

        for i in self.clients:
            self.plt.plot(self.coords_matrix[i][0] * 0.2, self.coords_matrix[i]
                          [1] * 0.2, c=self.color_palette[0], marker='o', markersize=17)
            self.plt.annotate('$q_{%d}=%d$' % (i, self.demands_array[i]), (
                self.coords_matrix[i][0] * 0.2, (self.coords_matrix[i][1]-3) * 0.2))

        self.plt.plot(self.loc_x[self.depot] * 0.2, self.loc_y[self.depot]
                      * 0.2, c='r', marker='s', markersize=17)
        self.plt.annotate(
            'DEPOT', ((self.loc_x[self.depot]-1.5) * 0.2, (self.loc_y[self.depot]-3) * 0.2))
        self.plt.savefig('instance.png')
        self.plt.show()

    # This function print all the clusters, with diferents colors for each node, in a 2-D cartesian plane.
    def printClusters(self, centers_list):
        self.plt.rc('font', size=16)
        self.plt.rc('figure', titlesize=18)
        self.plt.figure(figsize=(15, 15))

        for k, cluster in enumerate(self.clusters):
            for i in cluster:
                if i not in centers_list:
                    self.plt.plot(self.coords_matrix[i][0], self.coords_matrix[i]
                                  [1], c=self.color_palette[k], marker='o', markersize=17)
                    self.plt.annotate('$q_{%d}=%d$' % (
                        i, self.demands_array[i]), (self.coords_matrix[i][0], self.coords_matrix[i][1]-2))

        for k, j in enumerate(centers_list):
            if self.CLUSTER_TYPE == 'kmedoids':
                self.plt.plot(self.coords_matrix[j][0], self.coords_matrix[j]
                              [1], c=self.color_palette[k], marker='*', markersize=30)
                self.plt.annotate('$medoid_{%d}$' % (
                    k + 1), (self.coords_matrix[j][0], self.coords_matrix[j][1]-2))
            else:
                self.plt.plot(j[0], j[1], c=self.color_palette[k],
                              marker='*', markersize=30)
                self.plt.annotate('$centroid_{%d}$' % (k + 1), (j[0], j[1]-2))

        self.plt.plot(
            self.loc_x[self.depot], self.loc_y[self.depot], c='r', marker='s', markersize=17)
        self.plt.annotate(
            'DEPOT', (self.loc_x[self.depot]-1.5, self.loc_y[self.depot]-2))
        self.plt.show()

    # This function print the solution for the INSTANCE. Require a list with all the routes and a list with the arcs of the solution.
    def printSolution(self, solution, routes_arcs, solution_energies, solution_distances):
        self.plt.rc('font', size=16)
        self.plt.rc('figure', titlesize=18)
        self.plt.figure(figsize=(15, 15))
        legend_content = []
        legend_colors = []

        for k, route in enumerate(solution):
            for i in route:
                if i != 0:
                    self.plt.plot(self.coords_matrix[i][0] * 0.2, self.coords_matrix[i]
                                  [1] * 0.2, c=self.color_palette[k], marker='o', markersize=17)
                    self.plt.annotate('$q_{%d}=%d$' % (i, self.demands_array[i]), (
                        self.coords_matrix[i][0] * 0.2, (self.coords_matrix[i][1]-3) * 0.2))
                    # self.plt.annotate(str(self.demands_array[i]), (self.coords_matrix[i][0] * 0.2, (self.coords_matrix[i][1]-2.5) * 0.2))
            for x, y in routes_arcs[k]:
                self.plt.plot([self.loc_x[x] * 0.2, self.loc_x[y] * 0.2], [self.loc_y[x] *
                              0.2, self.loc_y[y] * 0.2], c=self.color_palette[k], alpha=0.5, linewidth=4)
            legend_content.append('Route ' + str(k + 1) + '\nEnergy: ' + str(round(solution_energies[k], 2)) + '\nDistance: ' +
                                  str(round(solution_distances[k], 2)))
            legend_colors.append(self.color_palette[k])

        self.plt.plot(self.loc_x[self.depot] * 0.2, self.loc_y[self.depot]
                      * 0.2, c='r', marker='s', markersize=17)
        self.plt.annotate(
            'DEPOT', ((self.loc_x[self.depot]-1.5) * 0.2, (self.loc_y[self.depot]-3) * 0.2))
        self.plt.title(label='Total Energy: ' + str(round(sum(solution_energies), 2)) + ', Total Distances: ' +
                       str(round(sum(solution_distances), 2)), fontweight=10, pad="2.0")
        self.plt.legend(legend_content, loc='lower center', bbox_to_anchor=(0.5, -0.1), ncol=10, labelcolor=legend_colors,
                        fontsize=10)
        self.plt.savefig(self.OUTPUT_SOLUTION_IMG)
        self.plt.show()

    # This print (with HTML) a PANDAS dataframe who represents the Distances matrix.
    def printDistancesMatrix(self):
        from IPython.display import display, HTML
        pd_distances_matrix = self.pd.DataFrame(self.distances_matrix)
        display(HTML(pd_distances_matrix.to_html()))

    def printDistancesMatrixNormalized(self):
        from IPython.display import display, HTML
        pd_distances_matrix = self.pd.DataFrame(
            self.distances_matrix_normalized)
        display(HTML(pd_distances_matrix.to_html()))

    # This print (with HTML) a PANDAS dataframe who represents the Energies matrix.
    def printEnergiesMatrix(self):
        from IPython.display import display, HTML
        pd_energies_matrix = self.pd.DataFrame(self.energies_matrix)
        display(HTML(pd_energies_matrix.to_html()))

    def printEnergiesMatrixNormalized(self):
        from IPython.display import display, HTML
        pd_energies_matrix = self.pd.DataFrame(self.energies_matrix_normalized)
        display(HTML(pd_energies_matrix.to_html()))

    # This print (with HTML) a PANDAS dataframe who represents the Energies Savings matrix.
    def printSavingMatrix(self):
        from IPython.display import display, HTML
        pd_saving_matrix = self.pd.DataFrame(self.saving_matrix)
        display(HTML(pd_saving_matrix.to_html()))

    # This print (with HTML) a PANDAS dataframe who represents the Energies Savings matrix.
    def printSavingEnergyMatrix(self):
        from IPython.display import display, HTML
        pd_saving_energies_matrix = self.pd.DataFrame(
            self.saving_energies_matrix)
        display(HTML(pd_saving_energies_matrix.to_html()))

    # This print (with HTML) a PANDAS dataframe who represents the final Combination matrix.
    def printCombinationMatrix(self):
        from IPython.display import display, HTML
        pd_combination_matrix = self.pd.DataFrame(self.combination_matrix)
        display(HTML(pd_combination_matrix.to_html()))

    # This print (with HTML) a PANDAS dataframe who represents the Pheromone matrix.
    def printPheromonesMatrix(self):
        from IPython.display import display, HTML
        pd_pheromones_matrix = self.pd.DataFrame(
            self.np.power(self.pheromones_matrix, self.alpha))
        display(HTML(pd_pheromones_matrix.to_html()))

    '''
         1.1 MULTIPORPOUSE FUNCTIONS: 
         These functions allow the program to resolve and determine a lot of problems and define values for variables.
    '''
    # Here we define some essential variables for the initialization of the ACO model.

    def setInitialParameters(self):
        import matplotlib.cm as cm
        if self.MAX_ITERATIONS == None:
            self.MAX_ITERATIONS = self.math.ceil(
                (self.math.sqrt(len(self.nodes)))*(1/len(self.nodes))*100)
        self.nodes, self.demands_array = [
            self.depot] + self.clients, self.np.array(list(self.demands.values()))
        self.MAX_ANTS = self.math.ceil(
            len(self.clients) / self.TOTAL_ANT_DIVIDER)
        self.coords_matrix = self.np.array(
            [(self.loc_x[i], self.loc_y[i]) for i in self.nodes])
        # Define a colors palette for the graphs functions.
        self.color_palette = cm.jet(self.np.linspace(0, 1, self.K_NUMBER + 1))
        # Define the weight of an empty vehicle.
        self.TARE = self.VEHICLE_CAPACITY * self.TARE_PERCENTAGE

    # Function to create a 'ndarray' that represents the matrix of DISTANCES between nodes.
    def createDistancesMatrix(self):
        self.distances_matrix = self.np.zeros(
            (len(self.nodes), (len(self.nodes))))
        ord_ = 1 if self.METRIC == 'manhattan' else 2

        for i in self.nodes:
            for j in self.nodes:
                if i != j:
                    self.distances_matrix[i][j] = self.np.linalg.norm(
                        self.coords_matrix[i] - self.coords_matrix[j], ord=ord_)
                # else:
                #     self.distances_matrix[i][j] = self.np.inf

    def createNormalizedDistancesMatrix(self):
        # Here we normalice the values between 0 and 1.
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        self.distances_matrix_normalized = self.np.divide(
            1, self.distances_matrix)
        self.distances_matrix_normalized[self.distances_matrix_normalized == self.np.inf] = 0
        scaler.fit(self.distances_matrix_normalized)
        self.distances_matrix_normalized = scaler.transform(
            self.distances_matrix_normalized)

    # Function to create a 'ndarray' that represents the matrix of ENERGIES between nodes.
    def createEnergiesMatrix(self):
        self.energies_matrix = self.np.zeros(
            (len(self.nodes), len(self.nodes)))

        for i in self.nodes:
            if i == self.depot:
                self.energies_matrix[i] = self.np.multiply(
                    self.distances_matrix[i], self.TARE)
            else:
                self.energies_matrix[i] = self.np.multiply(
                    self.distances_matrix[i], (self.demands_array[i] + self.TARE))

    def createNormalizedEnergiesMatrix(self):
        # Here we normalice the values between 0 and 1.
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        self.energies_matrix_normalized = self.np.divide(
            1, self.energies_matrix)
        self.energies_matrix_normalized[self.energies_matrix_normalized ==
                                        self.np.inf] = 0
        scaler.fit(self.energies_matrix_normalized)
        self.energies_matrix_normalized = scaler.transform(
            self.energies_matrix_normalized)

    # Function to create a 'ndarray' that represents the matrix of SAVINGS DISTANCES between two nodes and the depot.
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

        # Here we normalice the values between 0 and 1.
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        scaler.fit(self.saving_matrix)
        self.saving_matrix = scaler.transform(self.saving_matrix)

        for i in self.nodes:
            if i != self.depot:
                self.saving_matrix[self.depot][i] = 1
                self.saving_matrix[i][self.depot] = 1

    # Function to create a 'ndarray' that represents the matrix of SAVINGS ENERGIES between two nodes and the depot.
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

        # Here we normalice the values between 0 and 1
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        scaler.fit(self.saving_energies_matrix)
        self.saving_energies_matrix = scaler.transform(
            self.saving_energies_matrix)

        for i in self.nodes:
            if i != self.depot:
                self.saving_energies_matrix[self.depot][i] = 1
                self.saving_energies_matrix[i][self.depot] = 1

    # Function to create a 'ndarray' that represents the matrix of CAPACITY UTILIZATION of the vehicle for every node in the problem.
    def createCapacityUtilizationMatrix(self):
        self.cu_matrix = self.np.zeros((len(self.nodes), len(self.nodes)))
        for i in self.nodes:
            for j in self.nodes:
                self.cu_matrix[i][j] = (
                    self.demands_array[i] + self.demands_array[j]) / self.VEHICLE_CAPACITY

    def normalizePheromonesMatrix(self):
        # Here we normalice the values between 0 and 1.
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        pheromones_matrix_powered = self.np.power(
            self.pheromones_matrix, self.alpha)
        scaler.fit(pheromones_matrix_powered)
        return scaler.transform(pheromones_matrix_powered)

    def normalizeCombinationMatrix(self):
        # Here we normalice the values between 0 and 1.
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        combination_matrix = self.np.multiply(
            self._pheromones_matrix, self.combination_matrix)
        scaler.fit(combination_matrix)
        return scaler.transform(combination_matrix)

    # Here we combine the heuristics information to create the global heuristic matrix using the constant HEURISTIC_TYPE to control it.
    def combineHeuristicMatrices(self):
        self.combination_matrix = self.np.zeros(
            (len(self.nodes), len(self.nodes)))

        if (self.HEURISTIC_TYPE == 0):
            _distance_matrix = self.np.divide(1, self.distances_matrix)
            self.combination_matrix = self.np.power(
                _distance_matrix, self.beta)

        elif (self.HEURISTIC_TYPE == 1):
            _energy_matrix = self.np.divide(1, self.energies_matrix)
            self.combination_matrix = self.np.power(_energy_matrix, self.gamma)

        elif (self.HEURISTIC_TYPE == 2):
            if self.USE_NORMALIZED_MATRIX:
                _distances_matrix_normalized = self.np.power(
                    self.distances_matrix_normalized, self.beta)
                _energies_matrix_normalized = self.np.power(
                    self.energies_matrix_normalized, self.gamma)
                self.combination_matrix = self.np.multiply(
                    _distances_matrix_normalized, _energies_matrix_normalized)
            else:
                _distance_matrix = self.np.power(
                    self.np.divide(1, self.distances_matrix), self.beta)
                _energy_matrix = self.np.power(
                    self.np.divide(1, self.energies_matrix), self.gamma)
                self.combination_matrix = self.np.multiply(
                    _distance_matrix, _energy_matrix)

        elif (self.HEURISTIC_TYPE == 3):
            self.combination_matrix = self.np.power(
                self.saving_matrix, self.delta)

        elif (self.HEURISTIC_TYPE == 4):
            self.combination_matrix = self.np.power(
                self.saving_energies_matrix, self.eta)

        elif (self.HEURISTIC_TYPE == 5):
            _distance_matrix = self.np.power(
                self.np.divide(1, self.distances_matrix), self.beta)
            _saving_matrix = self.np.power(self.saving_matrix, self.delta)
            self.combination_matrix = self.np.multiply(
                _distance_matrix, _saving_matrix)

        elif (self.HEURISTIC_TYPE == 6):
            _energy_matrix = self.np.power(
                self.np.divide(1, self.energies_matrix), self.gamma)
            _saving_energies_matrix = self.np.power(
                self.saving_energies_matrix, self.eta)
            self.combination_matrix = self.np.multiply(
                _energy_matrix, _saving_energies_matrix)

        elif (self.HEURISTIC_TYPE == 7):
            _distance_matrix = self.np.power(
                self.np.divide(1, self.distances_matrix), self.beta)
            _energy_matrix = self.np.power(
                self.np.divide(1, self.energies_matrix), self.gamma)
            _saving_matrix = self.np.power(self.saving_matrix, self.delta)
            self.combination_matrix = self.np.multiply(
                _distance_matrix, _energy_matrix)
            self.combination_matrix = self.np.multiply(
                self.combination_matrix, _saving_matrix)

        elif (self.HEURISTIC_TYPE == 8):
            if self.USE_NORMALIZED_MATRIX:
                _distances_matrix_normalized = self.np.power(
                    self.distances_matrix_normalized, self.beta)
                _energies_matrix_normalized = self.np.power(
                    self.energies_matrix_normalized, self.gamma)
                _saving_energies_matrix = self.np.power(
                    self.saving_energies_matrix, self.eta)
                self.combination_matrix = self.np.multiply(
                    _distances_matrix_normalized, _energies_matrix_normalized)
                self.combination_matrix = self.np.multiply(
                    self.combination_matrix, _saving_energies_matrix)
            else:
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

        elif (self.HEURISTIC_TYPE == 9):
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

        elif (self.HEURISTIC_TYPE == 10):
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

        elif (self.HEURISTIC_TYPE == 11):
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

    # This function creates the arcs for the nodes within the same cluster. The mutate function depends of this function.
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
            # self.closest_nodes_in_clusters.append(cluster[closests_nodes_to_depot[1]]) # Uncomment this line if you want more than 1 node.
        '''self.closest_nodes_in_clusters = self.np.array(self.closest_nodes_in_clusters)
        self.closest_nodes_in_clusters -= 1
        self.closest_nodes_in_clusters = self.closest_nodes_in_clusters.tolist()'''
        self.closest_nodes_in_clusters = [
            idx - 1 for idx in self.closest_nodes_in_clusters]

    # This function create the new candidate_list.
    def createCandidateList(self, best_nodes=[]):
        # First select all the closest node to depot for every cluster.
        candidate_list = [] + self.closest_nodes_in_clusters
        # best_nodes is the list with the first nodes in the K best solutions in an iteration.
        candidate_list = candidate_list + best_nodes
        possible_random_nodes = [
            node for node in self.clients if node not in candidate_list]
        # Create a list with random nodes.
        random_nodes = self.random.sample(
            possible_random_nodes, self.MAX_ANTS - len(candidate_list))
        candidate_list = candidate_list + random_nodes
        return candidate_list

    '''
         1.2 Specific ACO model functions: 
         These functions allow the program to resolve and determine a lot of problems and define values for variables.
    '''
    # Calculate the t_min, t_max and t_0 value.

    def calculateTminTmax(self):
        energy = self.distances_matrix[self.depot, :] * (self.demands_array)
        # Define the lower bound for the pheromones value.
        self.t_min = self.H / (energy.sum() * 2)
        # Define the upper bound for the pheromones value.
        self.t_max = (self.H / energy.sum()) * self.K_NUMBER
        # This is the value for the initial pheromones, a middle point between t_min and t_max.
        self.t_0 = (self.t_min + self.t_max) / 2

    # This function set the initial values for the pheromone's matrix.
    # Each arc of nodes in the same cluster will have a pheromone value higher than that of intercluster nodes.
    def setInitialPheromones(self):
        # Start the pheromone's matrix with only t_0 values.
        self.pheromones_matrix = self.np.full(
            (len(self.nodes), len(self.nodes)), self.t_0)
        for cluster in self.clusters:
            cluster_arcs = list(self.permutations(cluster, 2))
            print(cluster_arcs)
            for i, j in cluster_arcs:
                # self.pheromones_matrix[i][j] = (self.t_max / 2) * 1.25 # A little formula to define de value for intracluster's arcs of nodes.
                self.pheromones_matrix[i][j] = self.t_max
    # When the stagnation condition is True, this function reset the Pheromone's matrix to the initial values but preserve a little information
    # of the best global solution.

    def restartPheromonesMatrix(self, gb_solution, gb_solution_arcs):
        self.pheromones_matrix.fill(self.t_0)
        for cluster in self.clusters:
            cluster_arcs = list(self.permutations(cluster, 2))
            for i, j in cluster_arcs:
                # self.pheromones_matrix[i][j] = (self.t_max / 2)
                self.pheromones_matrix[i][j] = self.t_max

#         total_energy_by_route = []
#         energy_arcs_by_route = []
#         for route_arcs in gb_solution_arcs:
#             energy_by_arc = {}
#             route_total_energy = 0
#             for i, j in route_arcs:
#                 route_total_energy += self.energies_matrix[i][j]
#                 energy_by_arc[(i, j)] = self.energies_matrix[i][j]
#             energy_arcs_by_route.append(energy_by_arc)
#             total_energy_by_route.append(route_total_energy)

#         for k, route_energy in enumerate(energy_arcs_by_route):
#             route_total_energy = total_energy_by_route[k]

#             for i, j in route_energy:
#                 numerator = route_total_energy - route_energy[(i, j)]
#                 denominator = len(gb_solution[k]) * route_total_energy
#                 local_arc_quality = numerator / denominator
#                 pheromones_quantity = (self.H / route_total_energy) * local_arc_quality
#                 self.pheromones_matrix[i][j] += pheromones_quantity * 0.85

    # Create a list with all the arcs between nodes for every route in a solution.
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

    # The objetive function of EMVRP.
    def calculateEnergies(self, solution):
        routes_energies = self.np.zeros(len(solution))
        for k, route in enumerate(solution):
            route_energy = 0
            for pos, i in enumerate(route):
                if pos == 0:
                    vehicle_weight = self.TARE
                    before_node = i
                else:
                    route_energy += self.distances_matrix[before_node][i] * \
                        vehicle_weight
                    vehicle_weight += self.demands_array[i]
                    before_node = i
            routes_energies[k] = route_energy
        return routes_energies

    # The objetive function of VRP.
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

    # Evaporate every single arc of pheromones.
    def evaporatePheromones(self):
        self.pheromones_matrix *= (1 - self.p)

    # This function do a local update of the pheromones with the single ant solution.
    def localUpdatePheromones(self, ant_solution, ant_quality):
        if self.PHEROMONE_UPDATING_STRATEGY == 0:
            pheromones_quantity = self.H / ant_quality
            ant_solution_arcs = self.generateSolutionArcs(ant_solution)

            for route_arcs in ant_solution_arcs:
                for i, j in route_arcs:
                    self.pheromones_matrix[i][j] += pheromones_quantity / \
                        self.MAX_ANTS
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
                        self.H / route_total_energy) * local_arc_quality
                    self.pheromones_matrix[i][j] += pheromones_quantity / \
                        self.MAX_ANTS

    # This function do a penalization in the pheromone matrix for all the arcs in the current worst solution.
    def currentWorstUpdatePheromones(self, cw_solution_arcs):
        for route_arcs in cw_solution_arcs:
            for i, j in route_arcs:
                self.pheromones_matrix[i][j] *= (1 - self.p)

    # Update the pheromone matrix with the current best solution.
    def currentBestUpdatePheromones(self, cb_solution, cb_solution_arcs, cb_quality):
        if self.PHEROMONE_UPDATING_STRATEGY == 0:
            pheromones_quantity = self.H / cb_quality
            for route_arcs in cb_solution_arcs:
                for i, j in route_arcs:
                    self.pheromones_matrix[i][j] += pheromones_quantity
        else:
            total_energy_by_route = []
            energy_arcs_by_route = []
            for route_arcs in cb_solution_arcs:
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
                    denominator = len(cb_solution[k]) * route_total_energy
                    local_arc_quality = numerator / denominator
                    pheromones_quantity = (
                        self.H / route_total_energy) * local_arc_quality
                    self.pheromones_matrix[i][j] += pheromones_quantity

    # Update the pheromone matrix with the global best solution.
    def globalBestUpdatePheromones(self, gb_solution, gb_solution_arcs, gb_quality):
        if self.PHEROMONE_UPDATING_STRATEGY == 0:
            pheromones_quantity = self.H / gb_quality
            for route_arcs in gb_solution_arcs:
                for i, j in route_arcs:
                    self.pheromones_matrix[i][j] += pheromones_quantity
        else:
            total_energy_by_route = []
            energy_arcs_by_route = []
            for route_arcs in gb_solution_arcs:
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
                    denominator = len(gb_solution[k]) * route_total_energy
                    local_arc_quality = numerator / denominator
                    pheromones_quantity = (
                        self.H / route_total_energy) * local_arc_quality
                    self.pheromones_matrix[i][j] += pheromones_quantity

    # A modified mutation function inspired in the BWAS mutation function.
    def mutatePheromonesMatrix(self, cb_solution_arcs, iteration, last_restart):
        cb_arcs_matrix = self.np.zeros((len(self.nodes), len(self.nodes)))
        for route_arcs in cb_solution_arcs:
            for i, j in route_arcs:
                cb_arcs_matrix[i][j] = 1

        t_threshold = (self.pheromones_matrix[cb_arcs_matrix == 1].mean())
        mutation_1 = (iteration - last_restart) / \
            (self.MAX_ITERATIONS - last_restart)
        mutation_2 = self.sigma * t_threshold
        mutation = mutation_1 * mutation_2
        mutation /= self.K_NUMBER

        for i in self.nodes:
            z = self.np.random.random()  # z is a value between 0 and 1.

            if z <= self.Pm:
                # We calculate a value between [0, 1, 2, 3].
                a = self.np.random.randint(4)
                # When a equal 3, we add the mutation value in the i row, but only with the clusters arcs column.
                if a == 3:
                    self.pheromones_matrix[i][self.clusters_arcs[i]
                                              == 1] += mutation
                # When a equal 2, we add the mutation value in the entire i row.
                elif a == 2:
                    self.pheromones_matrix[i] += mutation
                # When a equal 1, we substract the mutation value in the entire i row.
                elif a == 1:
                    self.pheromones_matrix[i] -= mutation
                # When a equal 0, we add the mutation value in the i row, but only with the clusters arcs column.
                else:
                    self.pheromones_matrix[i][self.clusters_arcs[i]
                                              == 0] -= mutation

    '''
        ##############################################
        ##  2. MAIN FUNCTION SECTION                ##
        ## Here are all the instructions for        ##
        ## the operation of the main ACO algorithm  ##
        ##############################################
    '''

    def solve(self):
        import time
        from copy import deepcopy

        print('• INITIALIZING ALGORITHM •')
        print('------------------------------\n')

        print('• 0. Reading instance file')
        print('------------------------------\n')

        # We break this assignament en three lines for better reading.
        result_list = list(self.reader.read())
        self.depot, self.clients, self.loc_x, self.loc_y, self.demands, self.total_demand, self.VEHICLE_CAPACITY, self.K_NUMBER, self.tightness_ratio =\
            result_list[0], result_list[1], result_list[2], result_list[3], result_list[
                4], result_list[5], result_list[6], result_list[7], result_list[8]

        self.setInitialParameters()
        if self.PRINT_INSTANCE:
            print('    - Instance draw:')
            self.printInstance()

        if self.MAX_ANTS > len(self.clients):
            print('Asegurese de definir un valor mayor para el parametro "TOTAL_ANT_DIVIDER", de tal'
                  + ' forma que el número total de hormigas sea igual o menor al número de nodos clientes.')
            raise Exception('Ha ocurrido un ERROR: El número total de hormigas es de ' + str(self.MAX_ANTS)
                            + ' y el número total de nodos clientes es de ' + str(len(self.clients)) + '.')

        # Creation of the heuristics matrices.
        self.createDistancesMatrix()
        self.createNormalizedDistancesMatrix()
        self.createEnergiesMatrix()
        self.createNormalizedEnergiesMatrix()
        self.createSavingMatrix()
        self.createSavingEnergiesMatrix()
        self.createCapacityUtilizationMatrix()
        self.combineHeuristicMatrices()

        print('    - Parameters:')
        print('        > Instance name: ' + str(self.INSTANCE))
        print('        > Number of nodes: ' + str(len(self.nodes)))
        print('        > Nodes and demands: ' + str(self.demands))
        print('        > Total demand: ' + str(self.total_demand) + ' units')
        print('        > Vehicles capacity: ' +
              str(self.VEHICLE_CAPACITY) + ' units')
        print('        > K-Optimum: ' + str(self.K_NUMBER))
        print('        > Tightness ratio: ' + str(self.tightness_ratio))
        print('        > Total of iterations: ' + str(self.MAX_ITERATIONS))
        print('        > Number of ants per iteration: ' +
              str(self.MAX_ANTS) + '\n')

        # Want to see this tables?
        if self.PRINT_DISTANCE_MATRIX:
            print('    - Distance matrix in a html table: ')
            self.printDistancesMatrix()
            self.printDistancesMatrixNormalized()
        if self.PRINT_ENERGY_MATRIX:
            print('    - Energy matrix in a html table: ')
            self.printEnergiesMatrix()
            self.printEnergiesMatrixNormalized()
        if self.PRINT_DISTANCE_SAVING_MATRIX:
            print('    - Distance saving matrix in a html table: ')
            self.printSavingMatrix()
        if self.PRINT_ENERGY_SAVING_MATRIX:
            print('    - Energy saving matrix in a html table: ')
            self.printSavingEnergyMatrix()
        if self.PRINT_COMBINATION_MATRIX:
            print('    - Combination matrix in a html table: ')
            self.printCombinationMatrix()

        print('• 1. Starting clustering process')
        print('------------------------------\n')
        print('    > CLUSTER TYPE: ' + self.CLUSTER_TYPE.upper())

        # Here we load the Cluster Model.
        if self.CLUSTER_TYPE == 'kmedoids':
            from src.clustering import KMedoidsEMVRP
            if self.USE_NORMALIZED_MATRIX:
                self.cluster_model = KMedoidsEMVRP(self.depot, self.nodes.copy(), self.demands_array.copy(), self.distances_matrix_normalized.copy(),
                                                   self.VEHICLE_CAPACITY, self.K_NUMBER)
            else:
                self.cluster_model = KMedoidsEMVRP(self.depot, self.nodes.copy(), self.demands_array.copy(), self.distances_matrix.copy(),
                                                   self.VEHICLE_CAPACITY, self.K_NUMBER)
        else:
            from src.clustering import KMeansEMVRP
            if self.USE_NORMALIZED_MATRIX:
                self.cluster_model = KMeansEMVRP(self.depot, self.nodes.copy(), self.demands_array.copy(), self.coords_matrix.copy(),
                                                 self.distances_matrix_normalized.copy(), self.VEHICLE_CAPACITY, self.K_NUMBER, self.METRIC)
            else:
                self.cluster_model = KMeansEMVRP(self.depot, self.nodes.copy(), self.demands_array.copy(), self.coords_matrix.copy(),
                                                 self.distances_matrix.copy(), self.VEHICLE_CAPACITY, self.K_NUMBER, self.METRIC)

        # Start the clustering process.
        self.clusters, clusters_total_cost, centers_list, unassigned_nodes = self.cluster_model.run()
        if len(unassigned_nodes) > 0:
            print(
                f'Ha ocurrido un ERROR: The following nodes are unassigned: {str(unassigned_nodes)}')
            raise Exception(
                f'Ha ocurrido un ERROR: The following nodes are unassigned: {str(unassigned_nodes)}')

        if self.PRINT_CLUSTERS:
            print('    - Clusters draw:')
            # Want to print the graph of clusters?
            self.printClusters(centers_list)
            print(self.clusters)
        print('    - Generated Clusters')
        for k in range(self.K_NUMBER):
            total_demand_on_k_cluster = sum(
                [self.demands_array[i] for i in self.clusters[k]])
            print('         > Cluster ' + str(k) + ': ' + str(self.clusters[k]) + ', with total demand: '
                  + str(total_demand_on_k_cluster))
            if total_demand_on_k_cluster > self.VEHICLE_CAPACITY:
                print('Ha ocurrido un ERROR: The total demand: ' + str(total_demand_on_k_cluster) + ', on the cluster ' + str(k + 1)
                      + ' is higher than the vehicle capacity: ' + str(self.VEHICLE_CAPACITY) + '.')
                print('The program shutdown inmediately.')
                return None
        print('         > Total nodes (without depot): ' +
              str(sum([len(cluster) for cluster in self.clusters])) + '\n')

        # Start the ACO algorithm.
        print('• 2. Starting BWAS algorithm (Free Ant)')
        print('------------------------------\n')
        start_time = time.time()
        # We load the Free Ant model and the VNS Local Search models.
        from src.metaheuristics import FreeAntEMVRP_2
        # Restricted VNS Local Search only to a single ant solution (only intracluster operators)
        from src.local_search import Free2OPTSearch
        # General VNS Local Search (intercluster and intracluster operators)
        from src.local_search import FreeLocalGVNS

        if self.MUTATE_PHEROMONES_MATRIX:
            # Mutation function depends of this new matrix.
            self.createClustersArcsMatrix()
        # We use this for the creation of the candidate_list.
        self.getClosestNodesFromDepot()
        self.calculateTminTmax()
        self.setInitialPheromones()
        # Create the list of candidate nodes, every node represent the deploy of an Ant.
        candidate_list = self.createCandidateList()

        if self.PRINT_PHEROMONE_MATRIX:
            print('    - Pheromone matrix in a html table: ')
            self.printPheromonesMatrix()

        print('    • Preliminar parameters:')
        print('        > Tmin value: ' + str(self.t_min**self.alpha))
        print('        > Tmax value: ' + str(self.t_max**self.alpha))
        print('        > Initial candidate list: ' + str(candidate_list) + '\n')

        # Here we define the variables where we will store the results.
        gb_solution = None
        gb_solution_quality = self.np.inf
        gb_solution_energies = []
        gb_solution_distances = []
        gb_solution_arcs = []

        # Some control's variables
        last_restart = 0
        # This variable controls when a reinitialization of the pheromone matrix should be done.
        stagnation = 0
        # In which iteration will the local search process begin?
        start_local_search = int(self.MAX_ITERATIONS / 3)

        for iteration in range(self.MAX_ITERATIONS):
            print('\n    • Iteration ' + str(iteration + 1))
            # We use this variable to show the mean quality for the all iteration solutions.
            mean_quality = 0

            # Variables to save the Current Best Solution.
            cb_solution = None
            cb_solution_quality = self.np.inf
            cb_solution_energies = []
            cb_solution_arcs = []

            # Variables to save the Current Worst Solution.
            cw_solution = None
            cw_solution_quality = 0
            cw_solution_energies = []
            cw_solution_arcs = []

            # Here we create the final values of Pheromones and Heuristic Combinations matrices for the current iteration.
            # self._pheromones_matrix = self.np.power(normalizePheromonesMatrix()self.pheromones_matrix, self.alpha)
            if self.USE_NORMALIZED_MATRIX:
                # self._pheromones_matrix = self.normalizePheromonesMatrix()
                # self._combinations_matrix = self.normalizeCombinationMatrix()
                self._pheromones_matrix = self.np.power(
                    self.pheromones_matrix, self.alpha)
                self._combinations_matrix = self.np.multiply(
                    self._pheromones_matrix, self.combination_matrix)
                from sklearn.preprocessing import MinMaxScaler
                scaler = MinMaxScaler()
                scaler.fit(self._combinations_matrix)
                self._combinations_matrix = scaler.transform(
                    self._combinations_matrix)
            else:
                self._pheromones_matrix = self.np.power(
                    self.pheromones_matrix, self.alpha)
                self._combinations_matrix = self.np.multiply(
                    self._pheromones_matrix, self.combination_matrix)

            # We use the nodes in candidate_list to deploy every ant in one of them to start its solution.
            for node in candidate_list:
                # The next 25% of candidate list are the best solution founds.
                best_k_quality = [self.np.inf for i in range(
                    int((self.MAX_ANTS * 25) / 100))]
                best_k_nodes = [0 for i in range(
                    int((self.MAX_ANTS * 25) / 100))]

                # We load the FreeAntEMVRP Model and generate a solution for the problem.
                self.ant_model = FreeAntEMVRP_1(self.depot, self.clients, node, self._combinations_matrix, self.distances_matrix,
                                                self.demands_array, self.VEHICLE_CAPACITY, self.TARE, self.START_ANT_ON_BEST_NODES, self.l0)
                ant_solution, routes_energies = self.ant_model.run()

                # Want to do a Local Search for the solution of every ant?
                if self.LS_ANT_SOLUTION:
                    local_search_model = Free2OPTSearch(self.depot, ant_solution, self.np.array(routes_energies), self.distances_matrix,
                                                        self.demands_array, self.TARE, self.VEHICLE_CAPACITY)
                    ant_solution, routes_energies = local_search_model.improve()
                    # We save the first node in the first route of the new solution.
                    node = ant_solution[0][1]

                # Want to do an update of the pheromone's matrix with the single ant solution?
                if self.LOCAL_ANT_UPDATE_PHEROMONES:
                    self.localUpdatePheromones(
                        ant_solution, sum(routes_energies))

                mean_quality += sum(routes_energies)

                # print('        > Hormiga ' + str(k + 1) + ': ENER. ' + str(sum(routes_energies)))

                # By this conditional statements, we storage the best current solution and the worst current solution.
                # The ONLY_K_OPTIMUM boolean variable define if the user want to storage solutions with higher number
                # of routes than K_NUMBER variable.
                if self.ONLY_K_OPTIMUM:
                    if sum(routes_energies) < cb_solution_quality and len(ant_solution) <= self.K_NUMBER:
                        cb_solution = self.deepcopy(ant_solution)
                        cb_solution_quality = sum(routes_energies)
                        cb_solution_energies = self.deepcopy(routes_energies)
                        cb_solution_arcs = self.generateSolutionArcs(
                            cb_solution)
                    elif sum(routes_energies) > cw_solution_quality:
                        cw_solution = self.deepcopy(ant_solution)
                        cw_solution_quality = sum(routes_energies)
                        cw_solution_energies = self.deepcopy(routes_energies)
                        cw_solution_arcs = self.generateSolutionArcs(
                            cw_solution)
                else:
                    if sum(routes_energies) < cb_solution_quality:
                        cb_solution = self.deepcopy(ant_solution)
                        cb_solution_quality = sum(routes_energies)
                        cb_solution_energies = self.deepcopy(routes_energies)
                        cb_solution_arcs = self.generateSolutionArcs(
                            cb_solution)
                    elif sum(routes_energies) > cw_solution_quality:
                        cw_solution = self.deepcopy(ant_solution)
                        cw_solution_quality = sum(routes_energies)
                        cw_solution_energies = self.deepcopy(routes_energies)
                        cw_solution_arcs = self.generateSolutionArcs(
                            cw_solution)

                # We update the best solutions found. We do this because we are interested in
                # knowing the initial nodes of these solutions, to feed the candidate_list.
                for idx, quality in enumerate(best_k_quality):
                    if sum(routes_energies) < quality and node not in self.closest_nodes_in_clusters:
                        best_k_quality[idx] = sum(routes_energies)
                        best_k_nodes[idx] = node
                        break

            # Want to do a Local Search to the best solution of this iteration?
            if self.LS_BEST_ITERATION:
                if iteration >= start_local_search:
                    # You can modify this statement. By this conditional statement we can choice between doing a LS to best iteration
                    # solution or do the LS to best global solution.
                    if self.np.random.random(1)[0] < 0.8:
                        local_search_model = FreeLocalGVNS(self.depot, cb_solution, self.np.array(cb_solution_energies), self.distances_matrix,
                                                           self.demands_array, self.TARE, self.VEHICLE_CAPACITY, self.K_NUMBER, iteration + 1,
                                                           self.MAX_ITERATIONS)
                    else:
                        local_search_model = FreeLocalGVNS(self.depot, gb_solution, self.np.array(gb_solution_energies), self.distances_matrix,
                                                           self.demands_array, self.TARE, self.VEHICLE_CAPACITY, self.K_NUMBER, iteration + 1,
                                                           self.MAX_ITERATIONS)
                    ls_solution, ls_energies = local_search_model.improve()

                    # We save the LS solution if is better than the current best solution.
                    if ls_energies.sum() < cb_solution_quality:
                        cb_solution = ls_solution
                        cb_solution_quality = ls_energies.sum()
                        cb_solution_energies = ls_energies
                        cb_solution_arcs = self.generateSolutionArcs(
                            cb_solution)

                        # We update the best solutions found. We do this because we are interested in
                        # knowing the initial nodes of these solutions, to feed the candidate_list.
                        for idx, quality in enumerate(best_k_quality):
                            if cb_solution_quality < quality and node not in self.closest_nodes_in_clusters:
                                best_k_quality[idx] = sum(routes_energies)
                                best_k_nodes[idx] = ls_solution[0][1]
                                break

            # We update the candidate_list with the best_k_nodes of this iteration
            candidate_list = self.createCandidateList(best_k_nodes)
            print('        > Mean quality of the solutions in this iteration : ' +
                  str(mean_quality / self.MAX_ANTS))

            # If the current best solution is better than the global best solution, we replace it.
            if cb_solution_quality < gb_solution_quality:
                print('        + New best energy: ' + str(int(cb_solution_quality)) + ', with the next distance: '
                      + str(int(sum(self.calculateDistances(cb_solution)))))
                gb_solution = self.deepcopy(cb_solution)
                gb_solution_quality = cb_solution_quality
                gb_solution_energies = self.deepcopy(cb_solution_energies)
                gb_solution_distances = self.calculateDistances(gb_solution)
                gb_solution_arcs = self.deepcopy(cb_solution_arcs)
                stagnation = 0  # Set the stagnation counter to 0.
            else:
                # Increase the stagnation counter in 1 (iteration).
                stagnation += 1

            # If the number of stagnations is higher than the 30% of MAX_ITERATIONS, then we do a restart of the pheromone matrix.
            if stagnation > int((self.MAX_ITERATIONS * 30) / 100):
                print('        - The Pheromone Matrix has been restarted.')
                self.restartPheromonesMatrix(gb_solution, gb_solution_arcs)
                last_restart = iteration
                stagnation = 0  # Set the stagnation counter to 0.
            # Else, we update the values of the pheromone's matrix.
            else:
                self.evaporatePheromones()
                if self.PENALIZE_WORST_SOLUTION:
                    self.currentWorstUpdatePheromones(cw_solution_arcs)
                if self.BEST_ITERATION_ANT_UPDATE_PHEROMONES:
                    self.currentBestUpdatePheromones(
                        cb_solution, cb_solution_arcs, cb_solution_quality)
                if self.BEST_GLOBAL_ANT_UPDATE_PHEROMONES:
                    self.globalBestUpdatePheromones(
                        gb_solution, gb_solution_arcs, gb_solution_quality)
                if self.MUTATE_PHEROMONES_MATRIX:
                    self.mutatePheromonesMatrix(
                        cb_solution_arcs, iteration, last_restart)

            # Set all the pheromones arcs between the range [t_min, t_max]
            self.pheromones_matrix[self.pheromones_matrix <
                                   self.t_min] = self.t_min
            self.pheromones_matrix[self.pheromones_matrix >
                                   self.t_max] = self.t_max

        print('\n    • Global best routes:')
        for k, route in enumerate(gb_solution):
            print('        - Route ' + str(k) + ': ' + str(route) +
                  ', with final total demand: ' + str(self.demands_array[route].sum()))

        print('        - Final total energy: ' +
              str(sum(gb_solution_energies)))
        print('        - Final total distance: ' +
              str(sum(gb_solution_distances)))
        print('        - Total time: %s seconds.' % (time.time() - start_time))

        if self.PRINT_SOLUTION:
            print('        - Solution draw:')
            self.printSolution(gb_solution, gb_solution_arcs,
                               gb_solution_energies, gb_solution_distances)
            for k, route in enumerate(gb_solution):
                print("Ruta : " + str(k))
                print("Peso inicial: " + str(self.TARE))
                route_energy = 0
                for pos, i in enumerate(route):
                    if pos == 0:
                        vehicle_weight = self.TARE
                        before_node = i
                    else:
                        route_energy += self.distances_matrix[before_node][i] * \
                            vehicle_weight
                        vehicle_weight += self.demands_array[i]
                        before_node = i
                        print(route_energy)
                print("\n")

        if self.LS_BEST_GLOBAL:
            from src.local_search import FreeBWACSGVN
            local_search_model = FreeBWACSGVN(self.depot, deepcopy(gb_solution), deepcopy(gb_solution_energies), self.distances_matrix,
                                              self.demands_array, self.tare, self.vehicle_capacity, self.k_number, iteration + 1,
                                              self.max_iterations)
            ls_solution, ls_routes_arcs, ls_energies, ls_routes_arcs_weights = local_search_model.improve()
            # print('\n RUTAS FINALES')
            # for k, route in enumerate(ls_solution):
            # print('> RUTA ' + str(k) + ': ' + str(route) + ', con demanda final: ' + str(self.demands_array[route].sum()))
            # print('> Energia total final: ' + str(ls_energies.sum()))
            # print("\n--- %s seconds ---" % (time.time() - start_time))
            # arcs = self.generateArcs(ls_solution)
            # self.printSolution(ls_solution, arcs)
            gb_solution_energies = ls_energies

        print('\n')
        return sum(gb_solution_energies), gb_solution, sum(gb_solution_distances), time.time() - start_time
