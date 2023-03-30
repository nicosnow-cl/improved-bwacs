import numpy as np
import time

from src.new.aco import FreeAnt
from src.readers import ReaderCVRPLIB

INSTANCE = 'instances/CVRPLIB/CMT/CMT1'
TARE_PERCENTAGE = 0.15
ALPHA = 0
BETA = 2


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


def createPheronomesMatrix(nodes):
    # pheromones_matrix = np.full((len(nodes), len(nodes)), random.random())
    pheromones_matrix = np.random.uniform(
        low=0, high=1, size=(len(nodes), len(nodes)))

    # for cluster in self.clusters:
    #     cluster_arcs = list(self.permutations(cluster, 2))
    #     print(cluster_arcs)
    #     for i, j in cluster_arcs:
    #         # self.pheromones_matrix[i][j] = (self.t_max / 2) * 1.25
    #         self.pheromones_matrix[i][j] = self.t_max

    return pheromones_matrix


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
pheromones_matrix = createPheronomesMatrix(nodes)
normalized_distances_matrix = np.divide(1, distances_matrix)
normalized_energies_matrix = np.divide(1, energies_matrix)
probabilities_matrix = np.multiply(np.power(pheromones_matrix, ALPHA),
                                   np.power(normalized_energies_matrix, BETA))
# probabilities_matrix = pheromones_matrix ** ALPHA * energies_matrix ** BETA

MAX_ITERATIONS = 200
BEST_SOLUTIONS = []

ant = FreeAnt(nodes, demands_array, vehicle_load, tare,
              distances_matrix, probabilities_matrix, 0.2)

start_time = time.time()
for i in range(MAX_ITERATIONS):
    print(f'Iteration {i + 1}')

    for j in range(50):
        solution, energies, loads = ant.generate_solution()
        # BEST_SOLUTIONS.append(
        #     {'solution': solution, 'energies': energies, 'loads': loads})

        if len(solution) <= k:
            # print(
            #     f'Ant {j + 1} found a solution with {len(solution)} routes and {sum(energies)} energy')
            BEST_SOLUTIONS.append(
                {'solution': solution, 'energies': energies, 'loads': loads})

final_time = time.time()
time_elapsed = final_time - start_time

print(f'Time elapsed: {time_elapsed}')
print(
    f'Best 5 solutions: {[sum(ant_solution["energies"]) for ant_solution in sorted(BEST_SOLUTIONS, key=lambda d: sum(d["energies"]))][:5]}')
