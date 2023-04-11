import numpy as np


from src.new.aco import FreeAnt
from src.new.acs import BWACS
from src.new.helpers import get_coords_matrix, get_distances_matrix
from src.new.machine_learning import KMeans
from src.new.metaheuristics import GeneralVNS
from src.new.models import VRPModel
from src.readers import ReaderCVRPLIB


ALPHA = 2
BETA = 2
DELTA = 2
INSTANCE = 'instances/CVRPLIB/CMT/CMT1'
MAX_ITERATIONS = 300
P = 0.2
P_M = 0.3
Q_0 = 0.8
SIMILARITY_PERCENTAGE_TO_DO_RESTART = 45
TARE_PERCENTAGE = 0.15


reader = ReaderCVRPLIB(INSTANCE)
depot, clients, loc_x, loc_y, demands_array, total_demand, max_capacity, k, \
    tightness_ratio = reader.read()

nodes = [depot] + clients
loc_x_lst = [loc_x[node] for node in nodes]
loc_y_lst = [loc_y[node] for node in nodes]
matrix_coords = get_coords_matrix(nodes, loc_x_lst, loc_y_lst)
matrix_costs = get_distances_matrix(nodes, matrix_coords)

parameters_kmeans = {
    'demands': np.array([demands_array[node] for node in demands_array]),
    'k_optimal': k,
    'matrix_coords': matrix_coords[:],
    'matrix_distances': matrix_costs[:],
    'max_capacity': max_capacity,
    'nodes': nodes[:],
}

kmeans = KMeans(**parameters_kmeans)
clusters, arcs_clusters, _, _, _ = kmeans.run()

parameters_ants = {
    'alpha': ALPHA,
    'ants_num': len(clients),
    # 'arcs_clusters': arcs_clusters,
    'beta': BETA,
    'delta': DELTA,
    'demands_array': np.array([demands_array[node] for node in demands_array]),
    'ipynb': True,
    'k_optimal': k,
    'matrix_costs': matrix_costs,
    'matrix_heuristics': matrix_costs,
    'max_capacity': max_capacity,
    'max_iterations': MAX_ITERATIONS,
    'model_ant': FreeAnt,
    'model_problem': VRPModel,
    'nodes': nodes,
    'p_m': P_M,
    'p': P,
    'percentage_of_similarity': SIMILARITY_PERCENTAGE_TO_DO_RESTART,
    'q0': Q_0,
    'tare': max_capacity * TARE_PERCENTAGE,
    'work_with_candidate_nodes': True,
    # 'model_ls_it': GeneralVNS,
}

bwacs = BWACS(**parameters_ants)
global_best_solution, best_solutions_set = bwacs.run()
