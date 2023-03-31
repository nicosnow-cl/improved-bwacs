import numpy as np
import time

from src.new.aco import FreeAnt
from src.readers import ReaderCVRPLIB

INSTANCE = 'instances/CVRPLIB/CMT/CMT1'
TARE_PERCENTAGE = 0.15
ALPHA = 1
BETA = 1


def createCoordsMatrix(nodes, loc_x, loc_y):
    return np.array([(loc_x[i], loc_y[i]) for i in nodes])


def createDistancesMatrix(nodes, coords_matrix, metric='euclidean'):
    distances_matrix = np.zeros((len(nodes), len(nodes)))
    _ord = 1 if metric == 'manhattan' else 2

    for i in nodes:
        for j in nodes:
            if i != j:
                distances_matrix[i][j] = np.linalg.norm(
                    coords_matrix[i] - coords_matrix[j], ord=_ord)

    return distances_matrix


def createEnergiesMatrix(nodes, depot, tare, distances_matrix, demands_array):
    energies_matrix = np.zeros((len(nodes), len(nodes)))

    for i in nodes:
        if i == depot:
            energies_matrix[i] = np.multiply(distances_matrix[i], tare)
        else:
            energies_matrix[i] = np.multiply(
                distances_matrix[i], (demands_array[i] + tare))

    return energies_matrix


def createPheronomesMatrix(nodes, base_matrix=None):
    # pheromones_matrix = np.full((len(nodes), len(nodes)), random.random())
    t_min, t_max = 0, 1
    if base_matrix is not None:
        t_min, t_max = np.min(base_matrix[base_matrix != 0]), np.max(
            base_matrix[base_matrix != np.inf])

    pheromones_matrix = np.random.uniform(
        low=t_min, high=t_max, size=(len(nodes), len(nodes)))

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


def update_pheromones_matrix(solution, pheromones_matrix, t_min, t_max,
                             quality=None,
                             p=0.02):
    solution_arcs = generate_solution_arcs(solution)

    evaporation_rate = (1 - p)
    new_pheromones_matrix = np.multiply(pheromones_matrix, evaporation_rate)
    new_pheromones_matrix[new_pheromones_matrix < 0] = t_min

    for route_arcs in solution_arcs:
        for i, j in route_arcs:
            new_pheromones_matrix[i][j] += quality if quality \
                is not None else t_max * evaporation_rate

    new_pheromones_matrix[new_pheromones_matrix > t_max] = t_max
    return new_pheromones_matrix


reader = ReaderCVRPLIB(INSTANCE)
depot, clients, loc_x, loc_y, demands, total_demand, vehicle_load, k, \
    tightness_ratio = reader.read()

tare = vehicle_load * TARE_PERCENTAGE
nodes = np.array([depot] + clients)
demands_array = np.array([demands[node] for node in demands])
coords_matrix = createCoordsMatrix(nodes, loc_x, loc_y)
distances_matrix = createDistancesMatrix(nodes, coords_matrix)
energies_matrix = createEnergiesMatrix(
    nodes, depot, tare, distances_matrix, demands_array)
normalized_distances_matrix = np.divide(1, distances_matrix)
normalized_energies_matrix = np.divide(1, energies_matrix)
pheromones_matrix, t_min, t_max = createPheronomesMatrix(
    nodes, normalized_energies_matrix)
probabilities_matrix = np.multiply(np.power(pheromones_matrix, ALPHA),
                                   np.power(normalized_energies_matrix, BETA))

MAX_ITERATIONS = 200
BEST_SOLUTIONS = []

ant = FreeAnt(nodes, demands_array, vehicle_load, tare,
              distances_matrix, probabilities_matrix, 0.2)

start_time = time.time()
for i in range(MAX_ITERATIONS):
    print(f'Iteration {i + 1}')
    iterations_solutions = []

    for j in range(50):
        solution, energies, loads = ant.generate_solution()
        iterations_solutions.append({'solution': solution,
                                     'energies': energies, 'loads': loads})

        # new_pheromones_matrix = update_pheromones_matrix(
        #     solution,
        #     pheromones_matrix,
        #     t_min,
        #     t_max,
        #     get_solution_quality(energies),
        #     0.1)

        # ant.set_probabilities_matrix(np.multiply(
        #     np.power(new_pheromones_matrix, ALPHA),
        #     np.power(normalized_energies_matrix, BETA)))

    print(
        '    > Mean solutions quality: ' +
        str(np.average([sum(solution["energies"]) for solution in
                        iterations_solutions])))

    iteration_best_solution = sorted(iterations_solutions, key=lambda d: sum(
        d["energies"]) and (len(d["solution"]) <= k))[0]

    new_pheromones_matrix = update_pheromones_matrix(
        iteration_best_solution["solution"],
        pheromones_matrix,
        t_min,
        t_max,
        get_solution_quality(iteration_best_solution["energies"]))

    ant.set_probabilities_matrix(np.multiply(
        np.power(new_pheromones_matrix, ALPHA),
        np.power(normalized_energies_matrix, BETA)))

    BEST_SOLUTIONS.append(iteration_best_solution)

final_time = time.time()
time_elapsed = final_time - start_time

print(f'Time elapsed: {time_elapsed}')
print(
    f'Best 5 solutions: {[(sum(ant_solution["energies"]), len(ant_solution["solution"])) for ant_solution in sorted(BEST_SOLUTIONS, key=lambda d: sum(d["energies"]))][:5]}')
