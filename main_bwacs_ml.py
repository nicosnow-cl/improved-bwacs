from itertools import permutations
from threading import Thread
import numpy as np


from src.new.aco import FreeAnt
from src.new.acs import BWACS
from src.new.helpers import get_coords_matrix, get_distances_matrix, get_saving_matrix, get_saving_matrix_2015
from src.new.machine_learning import KMeans
from src.new.metaheuristics import GeneralVNS
from src.new.models import VRPModel
from src.readers import ReaderCVRPLIB


ALPHA = 2
BETA = 2
DELTA = 4
INSTANCE = 'instances/CVRPLIB/CMT/CMT3'
MAX_ITERATIONS = 200
P = 0.2
P_M = 0.3
Q_0 = 0.8
SIMILARITY_PERCENTAGE_TO_DO_RESTART = 55
TARE_PERCENTAGE = 0.15


reader = ReaderCVRPLIB(INSTANCE)
depot, clients, loc_x, loc_y, demands, total_demand, max_capacity, k, \
    tightness_ratio = reader.read()

nodes = [depot] + clients
loc_x_lst = [loc_x[node] for node in nodes]
loc_y_lst = [loc_y[node] for node in nodes]
demands_array = [demands[node] for node in demands]
matrix_coords = get_coords_matrix(nodes, loc_x_lst, loc_y_lst)
matrix_distances = get_distances_matrix(nodes, matrix_coords)
saving_matrix = get_saving_matrix(depot, nodes, matrix_distances)
matrix_heuristics = matrix_distances * saving_matrix


parameters_kmeans = {
    'demands': np.array(demands_array),
    'k_optimal': k,
    'matrix_coords': matrix_coords[:],
    'matrix_distances': matrix_distances[:],
    'max_capacity': max_capacity,
    'nodes': nodes[:],
}

kmeans = KMeans(**parameters_kmeans)
clusters, arcs_clusters_lst, _, _, _, solutions = kmeans.run()

best_solutions_clusters = solutions[:]
best_solutions_clusters.reverse()
best_solutions_clusters = best_solutions_clusters[:int(k/2)]
best_solutions_clusters_arcs = []
for solution_clusters in best_solutions_clusters:
    clusters_arcs = [list(permutations(cluster, 2))
                     for cluster in solution_clusters]
    best_solutions_clusters_arcs.append(clusters_arcs)

parameters_ants = {
    'alpha': ALPHA,
    'ants_num': len(clients),
    'arcs_clusters_importance': .4,  # t_deta[i][j] *= (1 + 0.4)
    'arcs_clusters_lst': best_solutions_clusters_arcs,
    'beta': BETA,
    'delta': DELTA,
    'demands_array': demands_array,
    'ipynb': True,
    'k_optimal': k,
    # 'local_pheromone_update': True,
    'matrix_costs': matrix_distances,
    'matrix_heuristics': matrix_heuristics,
    'max_capacity': max_capacity,
    'max_iterations': MAX_ITERATIONS,
    'model_ant': FreeAnt,
    'model_ls_it': GeneralVNS,
    'model_problem': VRPModel,
    'nodes': nodes,
    'p_m': P_M,
    'p': P,
    'percentage_of_similarity': SIMILARITY_PERCENTAGE_TO_DO_RESTART,
    'q0': Q_0,
    'tare': max_capacity * TARE_PERCENTAGE,
    # 'work_with_candidate_nodes': True,
}

bwacs = BWACS(**parameters_ants)
bwacs.run()

# Run the algorithm in a new thread
# thread = Thread(target=bwacs.run)
# thread.start()
