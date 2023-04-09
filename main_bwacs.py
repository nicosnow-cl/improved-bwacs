import numpy as np

from src.new.aco import FreeAnt
from src.new.acs import BWACS
from src.new.helpers import get_coords_matrix, get_distances_matrix
from src.new.models import VRPModel
from src.readers import ReaderCVRPLIB

ALPHA = 1
BETA = 2
DELTA = 2
INSTANCE = 'instances/CVRPLIB/CMT/CMT1'
MAX_ITERATIONS = 250
P = 0.2
P_M = 0.3
Q_0 = 0.8
SIMILARITY_PERCENTAGE_TO_DO_RESTART = 50
TARE_PERCENTAGE = 0.15

reader = ReaderCVRPLIB(INSTANCE)
depot, clients, loc_x, loc_y, demands_array, total_demand, max_capacity, k, \
    tightness_ratio = reader.read()

nodes = [depot] + clients
loc_x_lst = [loc_x[node] for node in nodes]
loc_y_lst = [loc_y[node] for node in nodes]
coords_matrix = get_coords_matrix(nodes, loc_x_lst, loc_y_lst)
matrix_costs = get_distances_matrix(nodes, coords_matrix)

parameters = {
    'alpha': ALPHA,
    'ants_num': len(clients),
    'beta': BETA,
    'delta': DELTA,
    'demands_array': np.array([demands_array[node] for node in demands_array]),
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
    'q0': Q_0,
    'percentage_of_similarity': SIMILARITY_PERCENTAGE_TO_DO_RESTART,
    'tare': max_capacity * TARE_PERCENTAGE,
    'ipynb': True
}

acs = BWACS(**parameters)
acs.run()
