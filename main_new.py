import numpy as np
import random
import time

from src.new.helpers import get_repeated_elements_from_list
from src.new.aco import FreeAnt
from src.new.metaheuristics import GeneralVNS
from src.new.models import VRPModel, VehicleModel
from src.readers import ReaderCVRPLIB


def create_coords_matrix(nodes, loc_x, loc_y):
    return np.array([(loc_x[i], loc_y[i]) for i in nodes])


def create_distances_matrix(nodes, coords_matrix, metric='euclidean'):
    distances_matrix = np.zeros((len(nodes), len(nodes)))
    l_norm = 1 if metric == 'manhattan' else 2

    for i in nodes:
        for j in nodes:
            if i != j:
                distances_matrix[i][j] = np.linalg.norm(
                    coords_matrix[i] - coords_matrix[j], ord=l_norm)

    return distances_matrix


def create_energies_matrix(nodes,
                           depot,
                           tare,
                           distances_matrix,
                           demands_array):
    energies_matrix = np.zeros((len(nodes), len(nodes)))

    for i in nodes:
        if i == depot:
            energies_matrix[i] = np.multiply(distances_matrix[i], tare)
        else:
            energies_matrix[i] = np.multiply(
                distances_matrix[i], (demands_array[i] + tare))

    return energies_matrix


def create_simple_pheromones_matrix(nodes):
    return np.full((len(nodes), len(nodes)), 1)


def create_initial_pheromones_matrix(nodes, greedy_quality):
    total_clients = len(nodes)
    t_delta = total_clients / greedy_quality
    t_min = t_delta ** (total_clients / 2)
    t_max = t_delta * (total_clients / 2)

    pheromones_matrix = np.full((len(nodes), len(nodes)), t_delta)

    return pheromones_matrix, t_delta, t_min, t_max


def create_pheromones_matrix(node, t_delta):
    return np.full((len(node), len(node)), t_delta)


def calculate_t_values_legacy(distance_matrix, num_ant,
                              alpha=1, beta=2, p=0.2):
    """
    Calculates the values for t_0, t_min, and t_max using the given parameters
    and distance matrix.

    Parameters:
    distance_matrix (numpy.ndarray): A square matrix representing the
    distances between all nodes.
    num_vehicles (int): The number of vehicles in the fleet.
    alpha (float): The importance of the pheromone trail in the
    decision-making process.
    beta (float): The importance of the distance between nodes in the
    decision-making process.
    rho (float): The rate at which pheromone evaporation occurs.

    Returns:
    tuple: A tuple containing the values for t_0, t_min, and t_max.
    """

    n = distance_matrix.shape[0]  # Number of nodes
    m = num_ant  # Number of ants

    # Calculate average distance and maximum distance between nodes
    avg_distance = distance_matrix.mean()
    max_distance = distance_matrix.max()

    # Calculate t_0
    t_0 = 1 / (p * (m * avg_distance))

    # Calculate t_min
    t_min = 1 / ((1 - p) * n * m *
                 (alpha * avg_distance + beta * max_distance))

    # Calculate t_max
    t_max = 1 / (p * n * m * (alpha * avg_distance + beta * max_distance))

    return t_0, t_min, t_max


def calculate_t_values(best_solution_quality, ants_num, pheromones_matrix,
                       probabilities_matrix, p=0.2):
    t_max = 1 / (1 - p) * (1 / best_solution_quality)

    max_pheromone_value = np.max(pheromones_matrix)
    best_probability_value = np.max(probabilities_matrix)
    n_root_probabilitiy = best_probability_value ** (1 / ants_num)

    t_min = ((2 * max_pheromone_value)*(1 - n_root_probabilitiy)) / \
        ((ants_num - 2) * n_root_probabilitiy)

    return t_min, t_max


def get_solution_quality(solution_costs):
    # print(format(1 / sum(solution_costs), '.50f'))
    return 1 / sum(solution_costs)


def generate_solution_arcs(solution):
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


def generate_plain_solution_arcs(solution):
    solution_arcs = generate_solution_arcs(solution)

    return [arc for route_arcs in solution_arcs for arc in route_arcs]


def get_evaporated_pheromones_matrix(pheromones_matrix, t_min,  p=0.2):
    evaporation_rate = (1 - p)

    evaporated_pheromones_matrix = pheromones_matrix * evaporation_rate

    evaporated_pheromones_matrix[evaporated_pheromones_matrix < t_min] = t_min

    return evaporated_pheromones_matrix


def get_increased_pheromones_matrix(pheromones_matrix,
                                    global_best_solution_arcs,
                                    global_best_solution_quality,
                                    t_max):
    increased_pheromones_matrix = pheromones_matrix.copy()
    pheromones_amout = 0.2 * (1 / global_best_solution_quality)
    # print(f'pheromones_amout: {pheromones_amout}')
    # pheromones_amout = 1 / global_best_solution_quality

    # print(pheromones_amout)
    # plain_global_best_solution_arcs = generate_plain_solution_arcs(
    #     global_best_solution)

    # for i, j in plain_global_best_solution_arcs:
    #     increased_pheromones_matrix[i][j] += global_best_solution_quality

    for arcs_idxs in global_best_solution_arcs:
        increased_pheromones_matrix[arcs_idxs[:, 0],
                                    arcs_idxs[:, 1]] += \
            pheromones_amout

    increased_pheromones_matrix[increased_pheromones_matrix > t_max] = t_max

    return increased_pheromones_matrix


def get_decreased_pheromones_matrix(pheromones_matrix, global_best_solution,
                                    current_worst_solution, t_min, p=0.2):
    evaporation_rate = (1 - p)
    decreased_pheromones_matrix = pheromones_matrix.copy()

    plain_global_best_solution_arcs = generate_plain_solution_arcs(
        global_best_solution)
    plain_current_worst_solution_arcs = generate_plain_solution_arcs(
        current_worst_solution)

    for i, j in plain_current_worst_solution_arcs:
        if (i, j) not in plain_global_best_solution_arcs:
            decreased_pheromones_matrix[i][j] *= evaporation_rate

    decreased_pheromones_matrix[decreased_pheromones_matrix < t_min] = t_min

    return decreased_pheromones_matrix


def get_mutated_pheromones_matrix(pheromones_matrix,
                                  global_best_solution,
                                  current_iteration,
                                  iteration_when_do_restart,
                                  max_iterations,
                                  t_min,
                                  t_max,
                                  t_threshold=None,
                                  delta=2,
                                  p_m=0.3):
    mutation_intensity = get_mutation_intensity(current_iteration,
                                                iteration_when_do_restart,
                                                max_iterations,
                                                delta)
    # Original threshold function
    default_t_threshold = get_t_threshold(
        pheromones_matrix, global_best_solution)
    mutation_value = mutation_intensity * \
        (t_threshold if t_threshold is not None else default_t_threshold) \
        * 0.00005

    mutated_pheromones_matrix = pheromones_matrix.copy()

    # Use triu_indices to get upper triangle indices
    iu = np.triu_indices(pheromones_matrix.shape[0], k=1)

    # Update elements in upper triangle with random mutations
    mask = np.random.rand(len(iu[0])) < p_m
    mut = np.random.choice([-1, 1], size=len(iu[0])) * mutation_value
    mutated_pheromones_matrix[iu] += mask * mut

    # Use np.clip to clamp values between t_min and t_max
    np.clip(mutated_pheromones_matrix, t_min,
            t_max, out=mutated_pheromones_matrix)

    return mutated_pheromones_matrix


def get_mutation_intensity(current_iteration,
                           iteration_when_do_restart,
                           max_iterations,
                           delta):
    return ((current_iteration - iteration_when_do_restart) /
            (max_iterations - iteration_when_do_restart)) * delta


def get_t_threshold(pheromones_matrix, global_best_solution):
    plain_global_best_solution_arcs = generate_plain_solution_arcs(
        global_best_solution)
    pheromones = []

    for i, j in plain_global_best_solution_arcs:
        pheromones.append(pheromones_matrix[i][j])

    return np.average(pheromones)


def check_stagnation(iteration_best_solution,
                     iteration_worst_solution,
                     percentage_of_similarity):

    plain_iteration_best_solution_arcs = generate_plain_solution_arcs(
        iteration_best_solution)
    plain_iteration_worst_solution_arcs = generate_plain_solution_arcs(
        iteration_worst_solution)

    iteration_best_solution_set_arcs = set(plain_iteration_best_solution_arcs)
    iteration_worst_solution_set_arcs = set(
        plain_iteration_worst_solution_arcs)

    different_tuples = iteration_best_solution_set_arcs & \
        iteration_worst_solution_set_arcs
    actual_percentage = (len(different_tuples) /
                         len(iteration_best_solution_set_arcs.union(
                             plain_iteration_worst_solution_arcs))) * 100

    return actual_percentage >= percentage_of_similarity


def get_candidate_starting_nodes(best_solutions, clients):
    best_starting_nodes = {route[1]
                           for solution in best_solutions
                           for route in solution[0]}

    weights = [2 if node in best_starting_nodes else 1 for node in clients]

    candidate_starting_nodes = random.choices(
        clients, weights=weights, k=len(clients))

    return candidate_starting_nodes


INSTANCE = 'instances/CVRPLIB/CMT/CMT1'
# INSTANCE = 'instances/CVRPLIB/Golden/Golden_1'
# INSTANCE = 'instances/TSPLIB/Eil51/eil51.tsp'

reader = ReaderCVRPLIB(INSTANCE)
depot, clients, loc_x, loc_y, demands_array, total_demand, max_capacity, k, \
    tightness_ratio = reader.read()

TARE_PERCENTAGE = 0.15
ALPHA = 1
BETA = 2
MAX_ITERATIONS = 250
BEST_SOLUTIONS = []
BEST_SOLUTION = None
P = 0.2
q0 = 0.8
SIMILARITY_PERCENTAGE_TO_DO_RESTART = 50

tare = max_capacity * TARE_PERCENTAGE
nodes = [depot] + clients
demands_array = np.array([demands_array[node] for node in demands_array])
coords_matrix = create_coords_matrix(nodes, loc_x, loc_y)
matrix_costs = create_distances_matrix(nodes, coords_matrix)
energies_matrix = create_energies_matrix(nodes, depot, tare, matrix_costs,
                                         demands_array)
distances_mask = np.logical_and(
    matrix_costs != 0, np.isfinite(matrix_costs))
normalized_distances_matrix = np.zeros_like(matrix_costs)
normalized_distances_matrix[distances_mask] = np.divide(
    1, matrix_costs[distances_mask])

energies_mask = np.logical_and(
    energies_matrix != 0, np.isfinite(energies_matrix))
normalized_energies_matrix = np.zeros_like(energies_matrix)
normalized_energies_matrix[energies_mask] = np.divide(
    1, energies_matrix[energies_mask])

simple_pheromones_matrix = create_simple_pheromones_matrix(nodes)
simple_probabilities_matrix = np.multiply(np.power(simple_pheromones_matrix,
                                                   ALPHA),
                                          np.power(normalized_distances_matrix,
                                                   BETA))

greedy_ant = FreeAnt(nodes, demands_array, simple_probabilities_matrix,
                     matrix_costs, max_capacity, tare, q0, VRPModel)

ANT_COUNT = len(nodes)
# t_delta, t_min, t_max = calculate_t_values(
#     distances_matrix, ANT_COUNT, ALPHA, BETA, P)
# print(t_delta, t_min, t_max)

BEST_GREEDY_FITNESS = np.inf
for i in range(ANT_COUNT):
    (_, greedy_fitness, _, _, _) = greedy_ant.generate_solution()

    if greedy_fitness < BEST_GREEDY_FITNESS:
        BEST_GREEDY_FITNESS = greedy_fitness
# print(create_initial_pheromones_matrix(nodes, greedy_fitness))


# BASE_PHEROMONES_MATRIX, t_delta, t_min, t_max = \
#     create_initial_pheromones_matrix(nodes, greedy_fitness)
BASE_PHEROMONES_MATRIX = create_pheromones_matrix(nodes, 0.5)
pheromones_matrix = BASE_PHEROMONES_MATRIX.copy()
probabilities_matrix = np.multiply(np.power(pheromones_matrix, ALPHA),
                                   np.power(normalized_distances_matrix, BETA))
t_min, t_max = calculate_t_values(
    BEST_GREEDY_FITNESS, ANT_COUNT, BASE_PHEROMONES_MATRIX,
    probabilities_matrix, P)

ant = FreeAnt(nodes, demands_array, probabilities_matrix, matrix_costs,
              max_capacity, tare, q0, VRPModel)
local_search = GeneralVNS(matrix_costs, demands_array,
                          tare, max_capacity, k, MAX_ITERATIONS, VRPModel)

last_iteration_when_do_restart = 0
candidate_starting_nodes = []

start_time = time.time()
for i in range(MAX_ITERATIONS):
    print(f'Iteration {i + 1}')

    iterations_solutions = []
    reach_stagnation = False

    for j in range(ANT_COUNT):
        solution, fitness, routes_arcs, costs, loads = ant.generate_solution(
            candidate_starting_nodes.copy())
        iterations_solutions.append((solution, fitness, routes_arcs,
                                     costs, loads))

    # iterations_solutions_ls = []
    # for iteration_solution in iterations_solutions:
    #     if len(iteration_solution[0]) == k:
    #         ls_solution = local_search.improve(iteration_solution[0], i)
    #         if ls_solution[1] < iteration_solution[1]:
    #             iterations_solutions_ls.append(ls_solution)

    iterations_solutions_sorted = sorted(iterations_solutions,
                                         key=lambda d: d[1])
    iterations_solutions_sorted_and_restricted = [
        solution for solution in iterations_solutions_sorted
        if len(solution[0]) == k]

    iteration_best_solution = iterations_solutions_sorted_and_restricted[0] \
        if len(iterations_solutions_sorted_and_restricted) > 0 else \
        iterations_solutions_sorted[0]
    iteration_worst_solution = iterations_solutions_sorted[-1]
    average_iteration_costs = np.average([solution[1] for solution in
                                          iterations_solutions_sorted])

    print('    > Iteration resoluts: BEST({}), WORST({}), AVG({})'
          .format(iteration_best_solution[1],
                  iteration_worst_solution[1],
                  average_iteration_costs))

    # if (last_iteration_when_do_restart != 0):
    #     # LS by VNS
    #     ls_solution = local_search.improve(iteration_best_solution[0], i)
    #     if ls_solution[1] < iteration_best_solution[1]:
    #         iteration_best_solution = ls_solution

    global_best_solution = BEST_SOLUTION if BEST_SOLUTION \
        else iteration_best_solution

    pheromones_matrix = get_evaporated_pheromones_matrix(
        pheromones_matrix, t_min, P)
    pheromones_matrix = get_increased_pheromones_matrix(
        pheromones_matrix,
        global_best_solution[2],
        global_best_solution[1],
        t_max)
    pheromones_matrix = get_decreased_pheromones_matrix(
        pheromones_matrix,
        global_best_solution[0],
        iteration_worst_solution[0],
        t_min,
        P)

    if (last_iteration_when_do_restart != 0):
        pheromones_matrix = get_mutated_pheromones_matrix(
            pheromones_matrix,
            global_best_solution[0],
            i,
            last_iteration_when_do_restart,
            MAX_ITERATIONS,
            t_min,
            t_max)

    reach_stagnation = check_stagnation(
        iteration_best_solution[0],
        iteration_worst_solution[0],
        SIMILARITY_PERCENTAGE_TO_DO_RESTART)

    if reach_stagnation:
        print('    > Stagnation detected!')
        last_iteration_when_do_restart = i
        pheromones_matrix = BASE_PHEROMONES_MATRIX.copy()

    probabilities_matrix = (pheromones_matrix ** ALPHA) * \
        (normalized_distances_matrix ** BETA)
    ant.set_probabilities_matrix(probabilities_matrix)

    if BEST_SOLUTION is None \
            or iteration_best_solution[1] < global_best_solution[1]:
        BEST_SOLUTION = iteration_best_solution
    BEST_SOLUTIONS.append(iteration_best_solution)

    t_min, t_max = calculate_t_values(
        BEST_SOLUTION[1], ANT_COUNT, pheromones_matrix, probabilities_matrix, P
    )

    # if last_iteration_when_do_restart != 0:
    #     candidate_starting_nodes = \
    #         get_candidate_starting_nodes(BEST_SOLUTIONS, clients)


# unique_pheromones = np.unique(pheromones_matrix)
# unique_pheromones_sorted = sorted(unique_pheromones)
# print('\nUnique pheromones: {}'.format(unique_pheromones_sorted))

final_time = time.time()
time_elapsed = final_time - start_time
print(f'\nTime elapsed: {time_elapsed}')

BEST_SOLUTIONS_SORTED = sorted(BEST_SOLUTIONS,
                               key=lambda d: d[1])
BEST_SOLUTIONS_SET = []
best_solutions_fitness = []
for solution in BEST_SOLUTIONS_SORTED:
    if solution[1] not in best_solutions_fitness:
        BEST_SOLUTIONS_SET.append(solution)
        best_solutions_fitness.append(solution[1])

print('Best solution: {}'.format(
    (BEST_SOLUTION[1], len(BEST_SOLUTION[0]), BEST_SOLUTION[4])))
print('Best 5 solutions: {}'
      .format([(ant_solution[1], len(ant_solution[0]), ant_solution[4])
               for ant_solution in BEST_SOLUTIONS_SET][:5]))

if last_iteration_when_do_restart > 0:
    print(f'Last iteration when do restart: {last_iteration_when_do_restart}')
