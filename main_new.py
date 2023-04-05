import random
import numpy as np
import time

from src.new.aco import FreeAnt
from src.new.metaheuristics import GeneralVNS
from src.new.models import VRPModel
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

    evaporated_pheromones_matrix = np.multiply(pheromones_matrix,
                                               evaporation_rate)

    evaporated_pheromones_matrix[evaporated_pheromones_matrix < t_min] = t_min

    return evaporated_pheromones_matrix


def get_increased_pheromones_matrix(pheromones_matrix, global_best_solution,
                                    global_best_costs, t_max):
    global_best_solution_quality = get_solution_quality(global_best_costs)
    plain_global_best_solution_arcs = generate_plain_solution_arcs(
        global_best_solution)

    increased_pheromones_matrix = pheromones_matrix.copy()

    for i, j in plain_global_best_solution_arcs:
        increased_pheromones_matrix[i][j] += global_best_solution_quality

    increased_pheromones_matrix[increased_pheromones_matrix > t_max] = t_max

    return increased_pheromones_matrix


def get_decreased_pheromones_matrix(pheromones_matrix, global_best_solution,
                                    current_worst_solution, t_min, p=0.2):
    evaporation_rate = (1 - p)
    plain_global_best_solution_arcs = generate_plain_solution_arcs(
        global_best_solution)
    plain_current_worst_solution_arcs = generate_plain_solution_arcs(
        current_worst_solution)

    decreased_pheromones_matrix = pheromones_matrix.copy()

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
        * 0.005

    mutated_pheromones_matrix = pheromones_matrix.copy()
    nodes = range(pheromones_matrix.shape[0])

    for i in nodes:
        for j in nodes:
            if i != j and random.random() < p_m:
                a = random.randint(0, 1)

                if a == 0:
                    mutated_pheromones_matrix[i][j] += mutation_value
                else:
                    mutated_pheromones_matrix[i][j] -= mutation_value

    mutated_pheromones_matrix[mutated_pheromones_matrix < t_min] = t_min
    mutated_pheromones_matrix[mutated_pheromones_matrix > t_max] = t_max

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


INSTANCE = 'instances/CVRPLIB/CMT/CMT1'
# INSTANCE = 'instances/CVRPLIB/Golden/Golden_1'
# INSTANCE = 'instances/TSPLIB/Eil51/eil51.tsp'

reader = ReaderCVRPLIB(INSTANCE)
depot, clients, loc_x, loc_y, demands, total_demand, max_capacity, k, \
    tightness_ratio = reader.read()

TARE_PERCENTAGE = 0.15
ALPHA = 1
BETA = 2.5
MAX_ITERATIONS = 250
BEST_SOLUTIONS = []
P = 0.2
q0 = 0.8
SIMILARITY_PERCENTAGE_TO_DO_RESTART = 50

tare = max_capacity * TARE_PERCENTAGE
nodes = [depot] + clients
demands_array = np.array([demands[node] for node in demands])
coords_matrix = create_coords_matrix(nodes, loc_x, loc_y)
distances_matrix = create_distances_matrix(nodes, coords_matrix)
energies_matrix = create_energies_matrix(nodes, depot, tare, distances_matrix,
                                         demands_array)
normalized_distances_matrix = np.divide(1, distances_matrix)
normalized_energies_matrix = np.divide(1, energies_matrix)

simple_pheromones_matrix = create_simple_pheromones_matrix(nodes)
simple_probabilities_matrix = np.multiply(np.power(simple_pheromones_matrix,
                                                   ALPHA),
                                          np.power(normalized_distances_matrix,
                                                   BETA))
greedy_ant = FreeAnt(nodes, demands_array, max_capacity, tare,
                     distances_matrix, simple_probabilities_matrix,
                     q0, VRPModel)

ANT_COUNT = int(len(nodes))

GREEDY_SOLUTION = None
for i in range(ANT_COUNT):
    greedy_solution, greedy_rout_arcs, greedy_costs, greedy_load = \
        greedy_ant.generate_solution()
    if GREEDY_SOLUTION is None or sum(greedy_costs) < sum(
            GREEDY_SOLUTION[1]):
        GREEDY_SOLUTION = (greedy_solution, greedy_costs, greedy_load)


BASE_PHEROMONES_MATRIX, t_delta, t_min, t_max = \
    create_initial_pheromones_matrix(nodes, sum(GREEDY_SOLUTION[1]))
pheromones_matrix = BASE_PHEROMONES_MATRIX.copy()
probabilities_matrix = np.multiply(np.power(pheromones_matrix, ALPHA),
                                   np.power(normalized_distances_matrix, BETA))


ant = FreeAnt(nodes, demands_array, max_capacity, tare,
              distances_matrix, probabilities_matrix, q0, VRPModel)
local_search = GeneralVNS(distances_matrix, demands_array,
                          tare, max_capacity, k, MAX_ITERATIONS, VRPModel)

last_iteration_when_do_restart = 0
candidate_starting_nodes = None

start_time = time.time()
for i in range(MAX_ITERATIONS):
    print(f'Iteration {i + 1}')

    iterations_solutions = []
    reach_stagnation = False

    for j in range(ANT_COUNT):
        if candidate_starting_nodes is not None:
            ant.set_best_start_nodes(candidate_starting_nodes.copy())

        solution, routes_arcs, costs, load = ant.generate_solution()
        iterations_solutions.append(list((solution, costs, load)))

    iterations_solutions_sorted = sorted(iterations_solutions,
                                         key=lambda d: sum(d[1]))
    iterations_solutions_sorted_and_restricted = [
        solution for solution in iterations_solutions_sorted
        if len(solution[0]) == k]

    iteration_best_solution = iterations_solutions_sorted_and_restricted[0] \
        if len(iterations_solutions_sorted_and_restricted) > 0 else \
        iterations_solutions_sorted[0]
    iteration_worst_solution = iterations_solutions_sorted[-1]
    average_iteration_costs = np.average([sum(solution[1]) for solution in
                                          iterations_solutions_sorted])

    print('    > Iteration resoluts: BEST({}), WORST({}), AVG({})'
          .format(sum(iteration_best_solution[1]),
                  sum(iteration_worst_solution[1]),
                  average_iteration_costs))

    # # LS by VNS
    # ls_solution = local_search.improve(iteration_best_solution[0], i)
    # if sum(ls_solution[1]) < sum(iteration_best_solution[1]):
    #     iteration_best_solution = ls_solution

    global_best_solution = iteration_best_solution if len(
        BEST_SOLUTIONS) == 0 else BEST_SOLUTIONS[0]

    pheromones_matrix = get_evaporated_pheromones_matrix(
        pheromones_matrix, t_min, P)
    pheromones_matrix = get_increased_pheromones_matrix(
        pheromones_matrix,
        global_best_solution[0],
        global_best_solution[1],
        t_max)
    pheromones_matrix = get_decreased_pheromones_matrix(
        pheromones_matrix,
        global_best_solution[0],
        iteration_worst_solution[0],
        t_min,
        P)

    # if (last_iteration_when_do_restart != 0):
    #     pheromones_matrix = get_mutated_pheromones_matrix(
    #         pheromones_matrix,
    #         global_best_solution[0],
    #         i,
    #         last_iteration_when_do_restart,
    #         MAX_ITERATIONS,
    #         t_min,
    #         t_max,
    #         # t_threshold=get_solution_quality(global_best_solution[1])
    #     )

    reach_stagnation = check_stagnation(
        iteration_best_solution[0],
        iteration_worst_solution[0],
        SIMILARITY_PERCENTAGE_TO_DO_RESTART)

    if reach_stagnation:
        print('    > Stagnation detected!')
        last_iteration_when_do_restart = i
        pheromones_matrix = BASE_PHEROMONES_MATRIX.copy()

    ant.set_probabilities_matrix(np.multiply(
        np.power(pheromones_matrix, ALPHA),
        np.power(normalized_distances_matrix, BETA)))

    if i == 0:
        BEST_SOLUTIONS.append(iteration_best_solution)
    else:
        tuple_iteratin_best_solution = [
            tuple(route) for route in iteration_best_solution[0]]
        is_solution_already_stored = False

        for solution in BEST_SOLUTIONS:
            tuple_solution = [
                tuple(route) for route in solution[0]]
            different_routes = set(tuple_solution) - \
                set(tuple_iteratin_best_solution)

            if len(different_routes) == 0:
                is_solution_already_stored = True
                break

        if not is_solution_already_stored:
            BEST_SOLUTIONS.append(iteration_best_solution)
            BEST_SOLUTIONS = sorted(BEST_SOLUTIONS, key=lambda d: sum(d[1]))

    # best_starting_nodes = []
    # for solution in BEST_SOLUTIONS:
    #     for route in solution[0]:
    #         best_starting_nodes.append(route[1])
    # weights = (1.5 if node in best_starting_nodes else 1 for node in clients)
    # candidate_starting_nodes = random.choices(clients,
    #                                           weights=weights,
    #                                           k=len(clients))


final_time = time.time()
time_elapsed = final_time - start_time

print(f'\nTime elapsed: {time_elapsed}')
print('Best 5 solutions: {}'
      .format([(sum(ant_solution[1]), len(ant_solution[0]), ant_solution[2])
               for ant_solution in sorted(BEST_SOLUTIONS,
                                          key=lambda d: sum(d[1]))][:5]))
if last_iteration_when_do_restart > 0:
    print(f'Last iteration when do restart: {last_iteration_when_do_restart}')
