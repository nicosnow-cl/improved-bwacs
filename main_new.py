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


def create_energies_matrix(nodes, depot, tare, distances_matrix, demands_array):
    energies_matrix = np.zeros((len(nodes), len(nodes)))

    for i in nodes:
        if i == depot:
            energies_matrix[i] = np.multiply(distances_matrix[i], tare)
        else:
            energies_matrix[i] = np.multiply(
                distances_matrix[i], (demands_array[i] + tare))

    return energies_matrix


def create_pheromones_matrix(nodes, base_matrix=None):
    t_min, t_max = 0, 1

    if base_matrix is not None:
        t_min, t_max = np.min(base_matrix[base_matrix != 0]), np.max(
            base_matrix[base_matrix != np.inf])

    # pheromones_matrix = np.random.uniform(low=t_min, high=t_max,
    #                                       size=(len(nodes), len(nodes)))

    pheromones_matrix = np.random.uniform(low=t_min, high=t_min,
                                          size=(len(nodes), len(nodes)))

    # pheromones_matrix = np.zeros((len(nodes), len(nodes)))

    # for cluster in self.clusters:
    #     cluster_arcs = list(self.permutations(cluster, 2))
    #     print(cluster_arcs)
    #     for i, j in cluster_arcs:
    #         # self.pheromones_matrix[i][j] = (self.t_max / 2) * 1.25
    #         self.pheromones_matrix[i][j] = self.t_max

    return pheromones_matrix, t_min, t_max


def get_solution_quality(solution_energies):
    # print(format(1 / sum(solution_energies), '.50f'))
    return 1 / sum(solution_energies)


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


def get_evaporated_pheromones_matrix(pheromones_matrix, p=0.02):
    evaporation_rate = (1 - p)
    evaporated_pheromones_matrix = np.multiply(
        pheromones_matrix, evaporation_rate)
    # new_pheromones_matrix[new_pheromones_matrix < t_min] = t_min

    return evaporated_pheromones_matrix


def get_increased_pheromones_matrix(pheromones_matrix, global_best_solution,
                                    global_best_costs):
    global_best_solution_arcs = generate_solution_arcs(global_best_solution)
    global_best_solution_quality = get_solution_quality(global_best_costs)

    increased_pheromones_matrix = pheromones_matrix.copy()

    for route_arcs in global_best_solution_arcs:
        for i, j in route_arcs:
            increased_pheromones_matrix[i][j] += global_best_solution_quality

    return increased_pheromones_matrix


def get_decreased_phermones_matrix(pheromones_matrix, global_best_solution,
                                   current_worst_solution, p=0.02):
    evaporation_rate = (1 - p)
    global_best_solution_arcs = generate_solution_arcs(global_best_solution)
    current_worst_solution_arcs = generate_solution_arcs(
        current_worst_solution)

    decreased_pheromones_matrix = pheromones_matrix.copy()

    for route_arcs in global_best_solution_arcs:
        for i, j in route_arcs:
            if (i, j) not in current_worst_solution_arcs:
                decreased_pheromones_matrix[i][j] *= evaporation_rate

    return decreased_pheromones_matrix


reader = ReaderCVRPLIB(INSTANCE)
depot, clients, loc_x, loc_y, demands, total_demand, vehicle_load, k, \
    tightness_ratio = reader.read()

tare = vehicle_load * TARE_PERCENTAGE
nodes = np.array([depot] + clients)
demands_array = np.array([demands[node] for node in demands])
coords_matrix = create_coords_matrix(nodes, loc_x, loc_y)
distances_matrix = create_distances_matrix(nodes, coords_matrix)
energies_matrix = create_energies_matrix(nodes, depot, tare, distances_matrix,
                                         demands_array)
normalized_distances_matrix = np.divide(1, distances_matrix)
normalized_energies_matrix = np.divide(1, energies_matrix)
pheromones_matrix, t_min, t_max = create_pheromones_matrix(
    nodes, normalized_distances_matrix)
probabilities_matrix = np.multiply(np.power(pheromones_matrix, ALPHA),
                                   np.power(normalized_distances_matrix, BETA))


MAX_ITERATIONS = 200
ANT_COUNT = 50
BEST_SOLUTIONS = []

ant = FreeAnt(nodes, demands_array, vehicle_load, tare,
              distances_matrix, probabilities_matrix, 0.2)

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
        pheromones_matrix, 0.2)
    pheromones_matrix = get_increased_pheromones_matrix(
        pheromones_matrix,
        global_best_solution[0],
        global_best_solution[1])
    pheromones_matrix = get_decreased_phermones_matrix(
        pheromones_matrix,
        global_best_solution[0],
        iteration_worst_solution[0],
        0.2)

    ant.set_probabilities_matrix(np.multiply(
        np.power(pheromones_matrix, ALPHA),
        np.power(normalized_distances_matrix, BETA)))

    BEST_SOLUTIONS.append(best_iteration_solution)
    print(BEST_SOLUTIONS)
    BEST_SOLUTIONS = sorted(set(BEST_SOLUTIONS), key=lambda d: sum(d[1]))


final_time = time.time()
time_elapsed = final_time - start_time

print(f'Time elapsed: {time_elapsed}')
print(
    f'Best 5 solutions: {[(sum(ant_solution[1]), len(ant_solution[0]), len(ant_solution[2])) for ant_solution in sorted(BEST_SOLUTIONS, key=lambda d: sum(d[1]))][:5]}')
