from math import ceil
from threading import Thread

from src.new.aco import FreeAnt
from src.new.acs import ACS
from src.new.helpers import get_distances_matrix
from src.new.heuristics import HeuristicModel
from src.new.metaheuristics import GeneralVNS
from src.new.models import VRPModel
from src.readers import ReaderCVRPLIB

# EXECUTE RUNNING THIS COMMAND: python examples_py/main_acs.py

ALPHA = 1  # 0.75, 1, 1.05, 1.1, 1.25, 1.5, 1.75, 2
ANTS_NUM_RELATION = 2  # 1, 2
BETA = 2  # 2, 2.5, 3,  3.5
GAMMA = 1  # 1, 1.5 2
INSTANCE = 'instances/CVRPLIB/CMT/CMT1'
ITERATION_LOCAL_SEARCH_MODEL = GeneralVNS  # None, GeneralVNS
MAX_ITERATIONS = 500
MIN_ITERATIONS = 200
P = 0.1  # 0.05, 0.1, 0.15, 0.2, 0.25, 0.3
PHEROMONES_LOCAL_UPDATE = True
PROBABILITIES_MATRIX_TYPE = 'classic'  # 'classic', 'normalized'
Q_0 = 0.8
TARE_PERCENTAGE = 0.15
THREAD = False
CANDIDATE_NODES_TYPE = 'best'  # None, 'best', 'random'

reader = ReaderCVRPLIB(INSTANCE)
depot, clients, loc_x, loc_y, demands, _, max_capacity, k_optimal, _ = \
    reader.read()

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

parameters_heuristics = {
    'demands': demands,
    'importance_distances': BETA,
    'importance_savings': GAMMA,
    'matrix_coords': matrix_coords,
    'nodes': nodes,
}

heuristics = HeuristicModel(**parameters_heuristics)
matrix_heuristics = heuristics.get_heuristic_matrix(['distance', 'saving'])

parameters_ants = {
    'alpha': ALPHA,
    'ants_num': ceil(len(clients) / ANTS_NUM_RELATION),
    'beta': BETA,
    'demands': demands,
    'ipynb': True,
    'k_optimal': k_optimal,
    'matrix_costs': matrix_distances,
    'matrix_heuristics': matrix_heuristics,
    'max_capacity': max_capacity,
    'max_iterations': min(iterations, MAX_ITERATIONS),
    'model_ant': FreeAnt,
    # 'model_ls_it': ITERATION_LOCAL_SEARCH_MODEL,
    'model_problem': VRPModel,
    'nodes': nodes,
    'p': P,
    # 'pheromones_local_update': PHEROMONES_LOCAL_UPDATE,
    'q0': Q_0,
    'tare': max_capacity * TARE_PERCENTAGE,
    # 'type_candidate_nodes': CANDIDATE_NODES_TYPE,
}

acs = ACS(**parameters_ants)

if THREAD:
    thread = Thread(target=acs.run)
    thread.start()
else:
    acs.run()
