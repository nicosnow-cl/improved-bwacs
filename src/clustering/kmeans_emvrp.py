class KMeansEMVRP:
    def __init__(self, depot, nodes, demands_array, coords_matrix, distances_matrix, vehicle_capacity, k_number, metric):
        import numpy as np
        import math

        self.clusters = []
        self.centroids_list = []
        self.D = []
        self.depot = depot
        self.nodes = nodes
        self.demands_array = demands_array
        self.coords_matrix = coords_matrix
        self.distances_matrix = distances_matrix
        self.vehicle_capacity = vehicle_capacity
        self.k_number = k_number
        self.metric = metric
        self.distances_to_centroids_matrix = None
        self.weights = None
        self.np = np
        self.math = math

    def getDList(self):
        idx_centroids_list = []
        idx_centroids_list.append(
            (self.distances_matrix[self.depot, :] * self.demands_array).argmax())
        # idx_centroids_list.append(self.distances_matrix[self.depot, :].argmax())

        for k in range(self.k_number - 1):
            _priority = self.distances_matrix[:, [
                self.depot] + idx_centroids_list]
            _priority = _priority.mean(axis=1)
            _priority[idx_centroids_list] = 0
            _priority *= self.demands_array
            idx_centroids_list.append(_priority.argmax())

        self.D = self.coords_matrix[idx_centroids_list]
        self.nodes.remove(self.depot)
        self.nodes = self.np.array(self.nodes)
        self.nodes = self.nodes - 1
        self.demands_array = self.demands_array[1:]
        self.distances_matrix = self.np.delete(self.distances_matrix, 0, 0)
        self.distances_matrix = self.np.delete(self.distances_matrix, 0, 1)
        self.coords_matrix = self.np.delete(self.coords_matrix, 0, 0)

    def calculateDistancesToCentroids(self):
        self.distances_to_centroids_matrix = self.np.zeros(
            (self.k_number, len(self.nodes)))
        ord_ = 1 if self.metric == 'manhattan' else 2

        for k, c in enumerate(self.D):
            self.distances_to_centroids_matrix[k] = [self.np.linalg.norm(
                self.coords_matrix[i] - c, ord=ord_) for i in self.nodes]

    def calculateTotalCost(self, clusters):
        total_cost = 0

        for k, cluster in enumerate(clusters):
            cluster_energy = self.distances_to_centroids_matrix[k][cluster] * \
                self.demands_array[cluster]
            # cluster_energy = self.distances_to_centroids_matrix[k][cluster]
            total_cost += cluster_energy.sum()

        return total_cost

    def calculateNewCentroids(self, unassigned_nodes, clusters):
        D_list = self.np.array(
            [self.coords_matrix[cluster].mean(axis=0) for cluster in clusters])

        for node in unassigned_nodes:
            M = self.distances_to_centroids_matrix[:,
                                                   node] * self.demands_array[node]
            # M = self.distances_to_centroids_matrix[:, node]
            k = M.argsort()[0]
            D_list[k] = self.coords_matrix[node]

        return D_list

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

        self.getDList()
        iteration = 1
        max_iterations = self.math.ceil(len(self.nodes) * limit_constraint)
        best_clusters = None
        best_total_cost = self.np.inf
        best_unassigned_nodes = [[] for k in range(self.k_number)]
        best_constraint_clusters = None
        best_constraint_total_cost = self.np.inf
        best_constraint_unassigned_nodes = [[] for k in range(self.k_number)]
        max_stagnation = 0

        while (iteration <= max_iterations) or (best_total_cost == self.np.inf):
            print('    > ITERACIÓN ' + str(iteration))
            unassigned_nodes = self.demands_array.argsort()[::-1]
            self.calculateDistancesToCentroids()
            self.weights = self.np.zeros((self.k_number))
            self.clusters = [[] for c in range(self.k_number)]
            print('        - Centroides actuales: ' +
                  str([coords for coords in self.D]))

            for ri in self.nodes:
                if ri in unassigned_nodes:
                    M = self.distances_to_centroids_matrix[:,
                                                           ri] * self.demands_array[ri]
                    M = M.argsort()

                    for m in M:
                        cost_submatrix = self.distances_to_centroids_matrix[m,
                                                                            unassigned_nodes]

                        for pos, i in enumerate(unassigned_nodes):
                            cost_submatrix[pos] *= self.demands_array[i]

                        priority_matrix = cost_submatrix / \
                            self.demands_array[unassigned_nodes]
                        mask = self.np.where([True if (self.demands_array[i] + self.weights[m] <= self.vehicle_capacity) else False
                                              for i in unassigned_nodes])[0]

                        if len(mask) > 0:
                            ri_ = unassigned_nodes[mask[priority_matrix[mask].argmin(
                            )]]
                            self.clusters[m].append(ri_)
                            self.weights[m] += self.demands_array[ri_]
                            unassigned_nodes = unassigned_nodes[unassigned_nodes != ri_]

                            if ri not in unassigned_nodes:
                                break

            print('        - Nodos sin asignar: ' + str(unassigned_nodes) +
                  ', ' + str(self.demands_array[unassigned_nodes]))
            print('        - Mejor costo anterior: ' + str(best_total_cost))

            new_total_cost = self.calculateTotalCost(self.clusters)
            print('        - Costo total actual: ' + str(new_total_cost) + '\n')

            self.D = self.calculateNewCentroids(
                unassigned_nodes, self.clusters)

            if (new_total_cost < best_total_cost):
                best_clusters = deepcopy(self.clusters)
                for k in range(self.k_number):
                    best_clusters[k] = self.np.array(best_clusters[k])
                    best_clusters[k] += 1
                best_total_cost = new_total_cost
                best_centroids = deepcopy(self.D)
                best_unassigned_nodes = deepcopy(unassigned_nodes)
                best_unassigned_nodes += 1
            else:
                max_stagnation += 1

            if (not unassigned_nodes.size) and (new_total_cost < best_constraint_total_cost):
                best_constraint_clusters = deepcopy(self.clusters)
                for k in range(self.k_number):
                    best_constraint_clusters[k] = self.np.array(
                        best_constraint_clusters[k])
                    best_constraint_clusters[k] += 1
                best_constraint_total_cost = new_total_cost
                best_contraint_centroids = deepcopy(self.D)
                best_constraint_unassigned_nodes = deepcopy(unassigned_nodes)
                max_stagnation = 0

            if (max_stagnation >= int((len(self.nodes)/10) * limit_constraint)) and (not unassigned_nodes.size):
                break
            else:
                iteration += 1

        print('    > Mejor Clusterización: ' + str(best_clusters) + ', con costo final: ' + str(best_total_cost)
              + ', nodos sin asignar: ' + str(best_unassigned_nodes))
        print('    > Mejor Clusterización (Contraint): ' + str(best_constraint_clusters) + ', con costo final: '
              + str(best_constraint_total_cost) + ', nodos sin asignar: ' + str(best_constraint_unassigned_nodes))

        print('    --- %s seconds ---\n' % (time.time() - start_time))

        if best_constraint_clusters != None:
            return [cluster.tolist() for cluster in best_constraint_clusters], best_constraint_total_cost, best_contraint_centroids, best_constraint_unassigned_nodes
        else:
            return [cluster.tolist() for cluster in best_clusters], best_total_cost, best_centroids, best_unassigned_nodes
