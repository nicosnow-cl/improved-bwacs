from itertools import permutations
from threading import Thread
from math import ceil
import numpy as np

from src.new.ants import FreeAnt
from src.new.acs import BWAS
from src.new.helpers import get_distances_matrix
from src.new.machine_learning import KMeans
from src.new.metaheuristics import GeneralVNS
from src.new.models import VRPModel
from src.readers import ReaderCVRPLIB
from src.new.heuristics import HeuristicModel


ALPHA = 1  # 0.75, 1, 1.05, 1.1, 1.25, 1.5, 1.75, 2
BETA = 3  # 2, 2.5, 3,  3.5
GAMMA = 2  # 1, 1.5 2
DELTA = 2  # 1, 2, 3, 4
# INSTANCE = 'instances/CVRPLIB/CMT/CMT1'
INSTANCE = 'instances/CVRPLIB/CMT/CMT2'
MIN_ITERATIONS = 200
MAX_ITERATIONS = 500
ANTS_NUM_RELATION = 2
P = 0.15  # 0.05, 0.1, 0.15, 0.2, 0.25, 0.3
P_M = 0.2
Q_0 = 0.8
SIMILARITY_OF_ARCS_TO_DO_RESTART = 0.75  # 0.60, 0.70, 0.75, 0.80
# 0.885, 0.89, 0.9, 0.92, 0.95, 0.99
SIMILARITY_OF_QUALITIES_TO_DO_RESTART = None
TARE_PERCENTAGE = 0.15
PROBABILITIES_MATRIX_TYPE = 'normal'
THREAD = False
WORK_WITH_CANDIDATE_NODES = False


reader = ReaderCVRPLIB(INSTANCE)
depot, clients, loc_x, loc_y, demands, total_demand, max_capacity, k, \
    tightness_ratio = reader.read()

nodes, demands, matrix_coords = VRPModel.get_normalize_instance_parameters(
    depot,
    clients,
    demands,
    loc_x,
    loc_y)

errors = VRPModel.validate_instance(nodes, demands, max_capacity)
if errors:
    raise Exception(errors)

iterations = max(round(len(nodes), -2), MIN_ITERATIONS)
matrix_distances = get_distances_matrix(nodes, matrix_coords)
k_optimal = ceil(sum(demands) / max_capacity)

parameters_heuristics = {
    'demands': demands,
    'importance_distances': BETA,
    'importance_savings': GAMMA,
    'matrix_coords': matrix_coords,
    'nodes': nodes,
}

heuristics = HeuristicModel(**parameters_heuristics)
matrix_heuristics = heuristics.get_heuristic_matrix(['distance', 'saving'])

parameters_kmeans = {
    'demands': np.array(demands),
    'k_optimal': k,
    'matrix_coords': matrix_coords[:],
    'matrix_distances': matrix_distances[:],
    'max_capacity': max_capacity,
    'nodes': nodes[:],
}


kmeans = KMeans(**parameters_kmeans)
clusters, arcs_clusters_lst, best_cost, _, _, solutions = kmeans.run()

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
    'ants_num': ceil(len(clients) / ANTS_NUM_RELATION),
    'arcs_clusters_importance': .5,  # t_delta[i][j] *= (1 + 0.5)
    'arcs_clusters_lst': [arcs_clusters_lst],
    'beta': BETA,
    'delta': DELTA,
    'demands_array': demands,
    'ipynb': True,
    'k_optimal': k_optimal,
    'local_pheromone_update': True,
    'matrix_costs': matrix_distances,
    'matrix_heuristics': matrix_heuristics,
    'max_capacity': max_capacity,
    'max_iterations': min(iterations, MAX_ITERATIONS),
    'model_ant': FreeAnt,
    'model_ls_it': GeneralVNS,
    'model_problem': VRPModel,
    'nodes': nodes,
    'p_m': P_M,
    'p': P,
    'percent_arcs_limit': SIMILARITY_OF_ARCS_TO_DO_RESTART,
    'percent_quality_limit': SIMILARITY_OF_QUALITIES_TO_DO_RESTART,
    'q0': Q_0,
    'tare': max_capacity * TARE_PERCENTAGE,
    'work_with_candidate_nodes': WORK_WITH_CANDIDATE_NODES,
}

bwacs = BWAS(**parameters_ants)

if THREAD:
    thread = Thread(target=bwacs.run)
    thread.start()
else:
    bwacs.run()
