import random
import numpy as np
import time

from src.new.aco import FreeAnt
from src.readers import ReaderCVRPLIB

INSTANCE = 'instances/CVRPLIB/CMT/CMT1'
TARE_PERCENTAGE = 0.15
ALPHA = 1
BETA = 2


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
    total_clients = len(nodes) - 1
    t_delta = total_clients / greedy_quality
    t_min = t_delta / total_clients
    # t_max = t_delta * total_clients
    t_max = 1

    pheromones_matrix = np.full((len(nodes), len(nodes)), t_delta)

    return pheromones_matrix, t_delta, t_min, t_max


def get_solution_quality(solution_costs):
    # print(format(1 / sum(solution_energies), '.50f'))
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


def get_evaporated_pheromones_matrix(pheromones_matrix, t_min,  p=0.2):
    evaporation_rate = (1 - p)
    evaporated_pheromones_matrix = np.multiply(pheromones_matrix,
                                               evaporation_rate)
    # evaporated_pheromones_matrix[evaporated_pheromones_matrix < t_min] = t_min

    return evaporated_pheromones_matrix


def get_increased_pheromones_matrix(pheromones_matrix, global_best_solution,
                                    global_best_costs, t_max):
    global_best_solution_arcs = generate_solution_arcs(global_best_solution)
    global_best_solution_quality = get_solution_quality(global_best_costs)

    increased_pheromones_matrix = pheromones_matrix.copy()

    for route_arcs in global_best_solution_arcs:
        for i, j in route_arcs:
            increased_pheromones_matrix[i][j] += global_best_solution_quality

    increased_pheromones_matrix[increased_pheromones_matrix > t_max] = t_max

    return increased_pheromones_matrix


def get_decreased_phermones_matrix(pheromones_matrix, global_best_solution,
                                   current_worst_solution, t_min, p=0.2):
    evaporation_rate = (1 - p)
    global_best_solution_arcs = generate_solution_arcs(global_best_solution)
    current_worst_solution_arcs = generate_solution_arcs(
        current_worst_solution)

    decreased_pheromones_matrix = pheromones_matrix.copy()

    for route_arcs in global_best_solution_arcs:
        for i, j in route_arcs:
            if (i, j) not in current_worst_solution_arcs:
                decreased_pheromones_matrix[i][j] *= evaporation_rate

    # decreased_pheromones_matrix[decreased_pheromones_matrix < t_min] = t_min

    return decreased_pheromones_matrix


def get_mutated_pheromones_matrix(pheromones_matrix,
                                  global_best_solution,
                                  current_iteration,
                                  iteration_when_do_restart,
                                  max_iterations,
                                  t_min,
                                  delta=4,
                                  p_m=0.3):
    mutation_intensity = get_mutation_intensity(current_iteration,
                                                iteration_when_do_restart,
                                                max_iterations,
                                                delta)
    t_threshold = get_t_threshold(pheromones_matrix, global_best_solution)
    mutation_value = mutation_intensity * t_threshold

    mutated_pheromones_matrix = pheromones_matrix.copy()

    for i in range(pheromones_matrix.shape[0]):
        _ = random.random()
        if _ <= p_m:
            a = random.randint(0, 1)

            if a == 1:
                mutated_pheromones_matrix[i] += mutation_value
            else:
                mutated_pheromones_matrix[i] -= mutation_value

    mutated_pheromones_matrix[mutated_pheromones_matrix < t_min] = t_min

    return mutated_pheromones_matrix


def get_mutation_intensity(current_iteration,
                           iteration_when_do_restart,
                           max_iterations,
                           delta):
    return ((current_iteration - iteration_when_do_restart) /
            (max_iterations - iteration_when_do_restart)) * delta


def get_t_threshold(pheromones_matrix, global_best_solution):
    global_best_solution_arcs = generate_solution_arcs(global_best_solution)
    pheromones = []

    for route_arcs in global_best_solution_arcs:
        for i, j in route_arcs:
            pheromones.append(pheromones_matrix[i][j])

    return sum(pheromones) / len(pheromones)


reader = ReaderCVRPLIB(INSTANCE)
depot, clients, loc_x, loc_y, demands, total_demand, max_capacity, k, \
    tightness_ratio = reader.read()

tare = max_capacity * TARE_PERCENTAGE
nodes = np.array([depot] + clients)
demands_array = np.array([demands[node] for node in demands])
coords_matrix = create_coords_matrix(nodes, loc_x, loc_y)
distances_matrix = create_distances_matrix(nodes, coords_matrix)
energies_matrix = create_energies_matrix(nodes, depot, tare, distances_matrix,
                                         demands_array)
normalized_distances_matrix = np.divide(1, distances_matrix)
normalized_energies_matrix = np.divide(1, energies_matrix)


MAX_ITERATIONS = 100
ANT_COUNT = 50
BEST_SOLUTIONS = []
P = 0.2
q0 = 0.8

simple_pheromones_matrix = create_simple_pheromones_matrix(nodes)
simple_probabilities_matrix = np.multiply(np.power(simple_pheromones_matrix,
                                                   ALPHA),
                                          np.power(normalized_distances_matrix,
                                                   BETA))
greedy_ant = FreeAnt(nodes, demands_array, max_capacity, tare,
                     distances_matrix, simple_probabilities_matrix, q0)

GREEDY_SOLUTION = None
for i in range(ANT_COUNT):
    greedy_solution, greedy_costs, greedy_load = greedy_ant.generate_solution()
    if GREEDY_SOLUTION is None or sum(greedy_costs) < sum(
            GREEDY_SOLUTION[1]):
        GREEDY_SOLUTION = (greedy_solution, greedy_costs, greedy_load)


pheromones_matrix, t_delta, t_min, t_max = create_initial_pheromones_matrix(
    nodes, sum(GREEDY_SOLUTION[1]))
probabilities_matrix = np.multiply(np.power(pheromones_matrix, ALPHA),
                                   np.power(normalized_distances_matrix, BETA))


ant = FreeAnt(nodes, demands_array, max_capacity, tare,
              distances_matrix, probabilities_matrix, q0)

start_time = time.time()
for i in range(MAX_ITERATIONS):
    print(f'Iteration {i + 1}')
    iterations_solutions = []

    for j in range(ANT_COUNT):
        solution, costs, load = ant.generate_solution()
        iterations_solutions.append(list((solution, costs, load)))

    print('    > Mean solutions quality: ' +
          str(np.average([sum(solution[1]) for solution in
                          iterations_solutions])))

    iteration_solutions_sorted = sorted(
        iterations_solutions,
        key=lambda d: sum(d[1]) and (len(d[0]) == k))
    best_iteration_solution = iteration_solutions_sorted[0]
    global_best_solution = best_iteration_solution if len(
        BEST_SOLUTIONS) == 0 else BEST_SOLUTIONS[0]
    iteration_worst_solution = iteration_solutions_sorted[-1]

    pheromones_matrix = get_evaporated_pheromones_matrix(
        pheromones_matrix, t_min, P)
    pheromones_matrix = get_increased_pheromones_matrix(
        pheromones_matrix,
        global_best_solution[0],
        global_best_solution[1],
        t_max)
    pheromones_matrix = get_decreased_phermones_matrix(
        pheromones_matrix,
        global_best_solution[0],
        iteration_worst_solution[0],
        t_min,
        P)
    # pheromones_matrix = get_mutated_pheromones_matrix(pheromones_matrix,
    #                                                   global_best_solution[0],
    #                                                   i,
    #                                                   0,
    #                                                   MAX_ITERATIONS,
    #                                                   t_min)

    ant.set_probabilities_matrix(np.multiply(
        np.power(pheromones_matrix, ALPHA),
        np.power(normalized_distances_matrix, BETA)))

    BEST_SOLUTIONS.append(best_iteration_solution)
    BEST_SOLUTIONS = sorted(BEST_SOLUTIONS, key=lambda d: sum(d[1]))


final_time = time.time()
time_elapsed = final_time - start_time

print(f'Time elapsed: {time_elapsed}')
print(
    f'Best 5 solutions: {[(sum(ant_solution[1]), len(ant_solution[0]), len(ant_solution[2])) for ant_solution in sorted(BEST_SOLUTIONS, key=lambda d: sum(d[1]))][:5]}')
