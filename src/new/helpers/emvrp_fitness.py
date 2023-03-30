# Only for EMVRP

class EMVRPFitness:
    def __init__(self, distances_matrix, demands, tare):
        self.distances_matrix = distances_matrix
        self.demands = demands
        self.tare = tare

    def fitness_function(self, route):
        route_energy = 0
        vehicle_weight = None
        prev_node = None

        for pos, i in enumerate(route):
            if pos == 0:
                vehicle_weight = self.tare
            else:
                route_energy += self.distances_matrix[prev_node][i] * \
                    vehicle_weight
                vehicle_weight += self.demands[i]

            prev_node = i

        return route_energy

    @staticmethod
    def fitness_by_route(self, solution):
        return [self.fitness_function(route) for route in solution]

    @staticmethod
    def fitness(self, solution):
        return sum(self.fitness_by_route(solution))
