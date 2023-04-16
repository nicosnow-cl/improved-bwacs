import math
import time
from typing import List
import numpy as np
import itertools


class KMeans:
    D: np.ndarray
    demands: np.ndarray
    k_optimal: int
    matrix_coords: np.ndarray
    matrix_distances_to_centroids: np.ndarray
    matrix_distances: np.ndarray
    max_capacity: float
    max_iterations: int
    metric: str
    nodes: List[int]

    def __init__(self, **kwargs):
        self.metric = 'euclidean'
        self.max_iterations = None

        self.__dict__.update(kwargs)

        self.depot = self.nodes[0]

    def get_d_list(self):
        centroids_list_idx = []
        # TODO: change to ProblemModel method
        centroids_list_idx.append(
            self.matrix_distances[self.depot, :].argmax())

        for _ in range(self.k_optimal - 1):
            priority = self.matrix_distances[:, [
                self.depot] + centroids_list_idx]
            priority = priority.mean(axis=1)
            priority[centroids_list_idx] = 0
            centroids_list_idx.append(priority.argmax())

        D = self.matrix_coords[centroids_list_idx]
        self.nodes = self.nodes[1:]
        self.nodes = np.array(self.nodes)
        self.nodes = self.nodes - 1
        self.demands = self.demands[1:]
        self.matrix_distances = np.delete(self.matrix_distances, 0, 0)
        self.matrix_distances = np.delete(self.matrix_distances, 0, 1)
        self.matrix_coords = np.delete(self.matrix_coords, 0, 0)

        return D

    def get_distances_to_centroids(self):
        matrix_distances_to_centroids = np.zeros(
            (self.k_optimal, len(self.nodes)))
        l_norm = 1 if self.metric == 'manhattan' else 2

        for k, c in enumerate(self.D):
            matrix_distances_to_centroids[k] = [np.linalg.norm(
                self.matrix_coords[i] - c, ord=l_norm) for i in self.nodes]

        return matrix_distances_to_centroids

    def get_total_cost(self, clusters):
        total_cost = 0

        for k, cluster in enumerate(clusters):
            total_cost += self.matrix_distances_to_centroids[k][cluster].sum()

        return total_cost

    def get_new_centroids(self, unassigned_nodes, clusters):
        D = np.array(
            [self.matrix_coords[cluster].mean(axis=0) for cluster in clusters])

        for node in unassigned_nodes:
            M = self.matrix_distances_to_centroids[:, node]
            k = M.argsort()[0]
            D[k] = self.matrix_coords[node]

        return D

    def run(self):
        if self.max_iterations is None:
            if self.matrix_distances.shape[0] < 25:
                limit_constraint = 2
            elif self.matrix_distances.shape[0] >= 25 and \
                    self.matrix_distances.shape[0] < 50:
                limit_constraint = 2
            elif self.matrix_distances.shape[0] >= 50 and \
                    self.matrix_distances.shape[0] < 75:
                limit_constraint = 2
            elif self.matrix_distances.shape[0] >= 75 and \
                    self.matrix_distances.shape[0] < 100:
                limit_constraint = 1.5
            elif self.matrix_distances.shape[0] >= 100 and \
                    self.matrix_distances.shape[0] < 150:
                limit_constraint = 1
            elif self.matrix_distances.shape[0] >= 150 and \
                    self.matrix_distances.shape[0] < 250:
                limit_constraint = 0.8
            elif self.matrix_distances.shape[0] >= 250 and \
                    self.matrix_distances.shape[0] < 500:
                limit_constraint = 0.4
            elif self.matrix_distances.shape[0] >= 500 and \
                    self.matrix_distances.shape[0] < 1000:
                limit_constraint = 0.1
            else:
                limit_constraint = 0.05

            self.max_iterations = math.ceil(
                len(self.nodes) * limit_constraint)

        best_clusters = None
        best_constraint_clusters = None
        best_constraint_total_cost = np.inf
        best_constraint_unassigned_nodes = [[] for k in range(self.k_optimal)]
        best_solutions = []
        best_constraint_solutions = []
        best_total_cost = np.inf
        best_unassigned_nodes = [[] for k in range(self.k_optimal)]
        iteration = 1
        max_stagnation = int((len(self.nodes)/10) * limit_constraint)
        stagnation = 0

        start_time = time.time()
        self.D = self.get_d_list()

        while iteration <= self.max_iterations or best_total_cost == np.inf:
            print(f'    > Iteration: {iteration}/{self.max_iterations}')

            self.matrix_distances_to_centroids = \
                self.get_distances_to_centroids()

            unass_nodes = self.demands.argsort()[::-1].copy()
            clusters = [[] for c in range(self.k_optimal)]
            clusters_load = np.zeros((self.k_optimal))

            for ri in self.nodes:
                if ri in unass_nodes:
                    M = self.matrix_distances_to_centroids[:, ri]
                    M = M.argsort()

                    for m in M:
                        submatrix_costs = \
                            self.matrix_distances_to_centroids[m,
                                                               unass_nodes]

                        priority_matrix = submatrix_costs / \
                            self.demands[unass_nodes]
                        mask = np.where([
                            True if (
                                self.demands[i] + clusters_load[m]
                                <= self.max_capacity)
                            else False for i in unass_nodes])[0]

                        if len(mask) > 0:
                            ri_ = unass_nodes[mask[
                                priority_matrix[mask].argmin()]]
                            clusters[m].append(ri_)
                            clusters_load[m] += self.demands[ri_]
                            unass_nodes = unass_nodes[unass_nodes != ri_]

                            if ri not in unass_nodes:
                                break

            new_total_cost = self.get_total_cost(clusters)

            print('        - Unassigned nodes: ' + str(unass_nodes) +
                  ', ' + str(self.demands[unass_nodes]))
            print('        - Best previous total cost: '
                  + str(best_total_cost))
            print('        - New total cost: ' + str(new_total_cost) + '\n')

            self.D = self.get_new_centroids(unass_nodes, clusters)

            if new_total_cost < best_total_cost:
                best_clusters = clusters[:]

                for k in range(self.k_optimal):
                    best_clusters[k] = np.array(best_clusters[k])
                    best_clusters[k] += 1

                best_total_cost = new_total_cost
                best_centroids = self.D[:]
                best_unassigned_nodes = unass_nodes[:]
                best_unassigned_nodes += 1

                best_solutions.append(clusters[:])
            else:
                stagnation += 1

            if not unass_nodes.size and \
                    new_total_cost < best_constraint_total_cost:
                best_constraint_clusters = clusters[:]

                for k in range(self.k_optimal):
                    best_constraint_clusters[k] = np.array(
                        best_constraint_clusters[k])
                    best_constraint_clusters[k] += 1

                best_constraint_total_cost = new_total_cost
                best_contraint_centroids = self.D[:]
                best_constraint_unassigned_nodes = unass_nodes[:]
                stagnation = 0

                best_constraint_solutions.append(clusters[:])
            if stagnation >= max_stagnation and not unass_nodes.size:
                break
            else:
                iteration += 1

        print('    > Best clusters:  {} , with final cost: {}'
              .format(str(best_clusters), str(best_total_cost))
              + ', unassigned nodes: {}'
              .format(str(best_unassigned_nodes)))
        print('    > Best clusters (Constraint):  {} , with final cost: {}'
              .format(str(best_constraint_clusters),
                      str(best_constraint_total_cost))
              + ', unassigned nodes: {}'
              .format(str(best_constraint_unassigned_nodes)))
        print('    --- %s seconds ---\n' % (time.time() - start_time))

        if best_constraint_clusters is not None:
            clusters_lst = [cluster.tolist()
                            for cluster in best_constraint_clusters]
            clusters_arcs = [list(itertools.combinations(
                cluster, 2)) for cluster in clusters_lst]

            return (clusters_lst,
                    clusters_arcs,
                    best_constraint_total_cost,
                    best_contraint_centroids,
                    best_constraint_unassigned_nodes,
                    best_constraint_solutions)
        else:
            clusters_lst = [cluster.tolist()
                            for cluster in best_clusters]
            clusters_arcs = [list(itertools.combinations(
                cluster, 2)) for cluster in clusters_lst]

            return (clusters_lst,
                    clusters_arcs,
                    best_total_cost,
                    best_centroids,
                    best_unassigned_nodes,
                    best_solutions)
