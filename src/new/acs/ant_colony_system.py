from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
from typing import Any, List, Tuple
import numpy as np
import time

from ..ants import AntSolution
from ..helpers import same_line_print
from .ant_system import AS
from .aco_solution import ACOSolution

MAX_FLOAT = 1.0
MIN_FLOAT = np.finfo(float).tiny


class ACS(AS):
    epsilon: float
    pheromones_local_update: bool
    q0: float

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.epsilon = self.p / self.ants_num
        self.pheromones_local_update = False
        self.q0 = 0.8
        self.t_zero = (self.t_max + self.t_min) / 2

        self.__dict__.update(kwargs)

    def print_intance_parameters(self):
        super().print_intance_parameters()

        print('\tepsilon:', self.epsilon)
        print('\tpheromones_local_update:', self.pheromones_local_update)
        print('\tq0:', self.q0)

    # def create_pheromones_matrix(self,
    #                              t_delta: float = 0.5,
    #                              t_min: float = 0.1,
    #                              t_max: float = 1.0,
    #                              solutions=None) -> np.ndarray:
    #     """
    #     Creates the initial matrix of pheromone trail levels.

    #     Parameters:
    #         t_delta (float, optional): The initial value of the pheromone
    #         trail levels.

    #     Returns:
    #         A matrix of pheromone trail levels with all values initialized to
    #         t_delta.
    #         The matrix has shape (num_nodes, num_nodes), where num_nodes
    #         is the number of nodes in the problem.
    #     """

    #     shape = len(self.nodes)
    #     matrix_pheromones = np.full((shape, shape), t_delta)
    #     # matrix_pheromones = np.full((shape, shape), t_delta)

    #     if self.arcs_clusters_lst:
    #         num_clusters = len(self.arcs_clusters_lst)
    #         clusters_factor = 1 + \
    #             (self.arcs_clusters_importance / num_clusters)

    #         clusters_arcs_flattened = []
    #         for clusters_arcs in self.arcs_clusters_lst:
    #             clusters_arcs_flattened += get_flattened_list(clusters_arcs)

    #         for i, j in clusters_arcs_flattened:
    #             matrix_pheromones[i][j] = t_delta * (1 + clusters_factor)

    #     # if self.arcs_clusters_lst:
    #     #     clusters_arcs_flattened = []

    #     #     for clusters_arcs in self.arcs_clusters_lst:
    #     #         clusters_arcs_flattened += get_flattened_list(clusters_arcs)

    #     #     for i in range(shape):
    #     #         for j in range(shape):
    #     #             if (i, j) not in clusters_arcs_flattened:
    #     #                 matrix_pheromones[i][j] = t_delta

    #     if solutions:
    #         solutions_sorted = sorted(solutions, key=lambda d: d[1])
    #         best_solutions = solutions_sorted[:5]

    #         for solution in best_solutions:
    #             solution_flattened_arcs = get_flattened_list(solution[2])

    #             for i, j in solution_flattened_arcs:
    #                 matrix_pheromones[i][j] += self.get_acs_fitness(
    #                     solution[1]) * self.p

    #     return matrix_pheromones

    # def get_initial_t_delta(self, matrix_costs: np.ndarray) -> float:
    #     """
    #     Calculates the initial value of the pheromone trail levels.

    #     Parameters:
    #         matrix_costs (np.ndarray): A matrix of the costs between each pair
    #         of nodes.

    #     Returns:
    #         The initial value of the pheromone trail levels.
    #     """

    #     return 1 / (len(self.nodes) * matrix_costs.max())

    # def get_t_delta(self, matrix: np.ndarray) -> float:
    #     """
    #     Calculates the initial value of the pheromone trail levels.

    #     Parameters:
    #         matrix (np.ndarray): A matrix.

    #     Returns:
    #         The initial value of the pheromone trail levels.
    #     """

    #     return 1 / matrix.max()

    # def calculate_t_min_t_max(self,
    #                           best_solution_quality: float) \
    #         -> Tuple[float, float]:
    #     """
    #     Calculates the minimum and maximum values of the pheromone trail
    #     levels.

    #     Parameters:
    #         best_quality (float): The quality of the best solution found so
    #         far.

    #     Returns:
    #         A tuple containing the minimum and maximum values of the pheromone
    #         trail levels.
    #     """

    #     n = len(self.nodes)

    #     t_max = 1 / (self.p * best_solution_quality)
    #     t_min = t_max * (1 - (0.05) ** (1 / n)) / \
    #         ((n / 2 - 1) * (0.05) ** (1 / n))

    #     return t_min, t_max

    # def calculate_t_min_t_max_mmas(self, best_solution_quality: float):
    #     n = len(self.nodes)
    #     avg = n / 2
    #     p_best = 0.05
    #     p_best_n_root = exp(log(p_best) / n)

    #     t_max = (1 / (1 - self.p)) * \
    #         self.get_acs_fitness(best_solution_quality)

    #     upper = t_max * (1 - p_best_n_root)
    #     lower = (avg - 1) * p_best_n_root

    #     t_min = upper / lower

    #     return t_min, t_max

    # def get_acs_fitness(self, solutin_quality: float) -> float:
    #     """
    #     Calculates the quality of a solution as the inverse of the sum of its
    #     costs.

    #     Parameters:
    #         solution_costs (List[float]): A list of the costs of each tour in
    #         the solution.

    #     Returns:
    #         The quality of the solution as a float. The higher the value, the
    #         better the solution.
    #     """

    #     return 1 / solutin_quality

    # def evaporate_pheromones_matrix(self,
    #                                 evaporation_rate: float = None) -> None:
    #     """
    #     Evaporates the pheromone trail levels in the pheromone matrix.

    #     Parameters:
    #         None.

    #     Returns:
    #         None.
    #     """

    #     self.matrix_pheromones *= self.evaporation_rate if evaporation_rate \
    #         is None else evaporation_rate

    # def update_pheromones_matrix(self,
    #                              solution_arcs,
    #                              solution_quality,
    #                              factor=1):
    #     """
    #     Updates the pheromones matrix based on the given solution arcs and
    #     quality.

    #     Parameters:
    #         solution_arcs (List[np.ndarray]): List of 2D numpy arrays
    #         containing the arcs used in the solution.

    #         solution_quality (float): The quality of the solution.

    #     Returns:
    #         None.
    #     """

    #     pheromones_amount = self.get_acs_fitness(solution_quality) * factor

    #     # for arcs_idxs in solution_arcs:
    #     #     self.matrix_pheromones[arcs_idxs[:, 0],
    #     #                            arcs_idxs[:, 1]] += pheromones_amount

    #     for arcs_lst in solution_arcs:
    #         for i, j in arcs_lst:
    #             self.matrix_pheromones[i][j] += pheromones_amount

    # def set_bounds_to_pheromones_matrix(self, max=1) -> None:
    #     """
    #     Sets the minimum and maximum values for the pheromone trail levels,
    #     based on the values of t_min and t_max.

    #     Parameters:
    #         None.

    #     Returns:
    #         None.
    #     """

    #     np.clip(self.matrix_pheromones, self.t_min,
    #             max, out=self.matrix_pheromones)

    # def get_normalized_matrix(self, matrix: np.ndarray) -> np.ndarray:
    #     mask = (matrix != 0) & np.isfinite(matrix)

    #     with np.errstate(divide='ignore'):  # ignore division by zero warnings
    #         return np.divide(1, matrix, out=np.zeros_like(matrix), where=mask)

    # def get_probabilities_matrix(self, pheromones_matrix: np.ndarray) \
    #         -> np.ndarray:
    #     """
    #     Get the updated matrix of probabilities of choosing an arc.

    #     Parameters:
    #         None.

    #     Returns:
    #         A matrix(ndarray) of probabilities of choosing an arc.
    #     """

    #     if self.type_probabilities_matrix == 'normalized':
    #         inv_distances_matrix = get_inversed_matrix(
    #             self.matrix_costs)
    #         min_not_zero_value = inv_distances_matrix[
    #             inv_distances_matrix != 0].min()
    #         max_value = \
    #             inv_distances_matrix[inv_distances_matrix != np.inf].max()

    #         # Here we normalice the values between min distance
    #         # and max distance.
    #         scaler = MinMaxScaler(feature_range=(
    #             min_not_zero_value, max_value))
    #         norm_matrix_pheromones = scaler.fit_transform(pheromones_matrix)

    #         return np.multiply(np.power(norm_matrix_pheromones, self.alpha),
    #                            self.matrix_heuristics)
    #     else:
    #         return np.multiply(np.power(pheromones_matrix, self.alpha),
    #                            self.matrix_heuristics)

    # def get_candidate_nodes_weight(self, solutions, type: str = 'best'):
    #     """
    #     Returns a list of candidate starting nodes for the ants, biased
    #     towards the best starting nodes from the given solutions.

    #     Parameters:
    #         solutions(list): A list of solutions to the TSP problem, each
    #         represented as a tuple of a list of arcs and their
    #         corresponding cost.

    #     Returns:
    #         list: A list of candidate starting nodes for the ants.
    #     """

    #     if type == 'random':
    #         return [random.random() for _ in range(0, len(self.nodes))]
    #     else:
    #         all_clients = self.nodes[1:][:]
    #         half_clients_len = ceil(len(all_clients) / 2)
    #         max_candidates_set = ceil(half_clients_len / 2)

    #         clientes_sorted_by_distance = sorted(
    #             all_clients, key=lambda x: self.matrix_costs[x][0])
    #         closest_nodes = set(
    #             clientes_sorted_by_distance[:max_candidates_set])

    #         step = ceil(self.k_optimal / 2)
    #         distributed_solutions = []
    #         if len(solutions) <= self.k_optimal * step:
    #             distributed_solutions = sorted(solutions, key=lambda d: d[1])[
    #                 :self.k_optimal]
    #         else:
    #             distributed_solutions = sorted(solutions, key=lambda d: d[1])[
    #                 ::step][:self.k_optimal]

    #         best_starting_nodes = set()
    #         for solution in distributed_solutions:
    #             # if len(best_starting_nodes) >= self.k_optimal:
    #             #     break

    #             for route in solution[0]:
    #                 start_node = route[1]
    #                 # end_node = route[-2]

    #                 best_starting_nodes.add(start_node)
    #                 # best_starting_nodes.add(end_node)

    #         random_nodes = set(random.sample(all_clients, max_candidates_set))

    #         weights = [get_element_ranking(
    #             node,
    #             1,
    #             [best_starting_nodes, closest_nodes, random_nodes],
    #             True)
    #             for node in self.nodes]
    #         return weights

    def solve(self) -> ACOSolution:
        """
        Solve the problem using the Ant Colony Optimization algorithm.

        Args:
            None.

        Returns:
            ACOSolution: A dictionary with the best-global solution,
            best-iterations solutions and statistics data to the problem.
        """

        self.print_intance_parameters()
        print('\n')

        errors = self.model_problem.validate_instance(
            self.nodes, self.demands, self.max_capacity)
        if errors:
            raise Exception(errors)

        # Starting initial matrixes
        self.matrix_pheromones = self.create_pheromones_matrix(self.t_max)
        self.matrix_probabilities = self.create_probabilities_matrix(
            self.matrix_pheromones.copy(),
            self.matrix_heuristics.copy(),
            self.alpha,
            self.beta)

        # Greedy ants to find the best initial solution
        greedy_ant = self.model_ant(self.nodes,
                                    self.demands,
                                    self.matrix_probabilities,
                                    self.matrix_costs,
                                    self.max_capacity,
                                    self.tare,
                                    self.model_problem,
                                    self.q0)

        greedy_ant_best_solution: AntSolution = {
            'cost': np.inf,
            'routes_arcs': [],
            'routes_costs': [],
            'routes_loads': [],
            'routes': []
        }
        for _ in range(self.ants_num):
            greedy_ant_solution = greedy_ant.generate_solution()
            if greedy_ant_solution['cost'] < greedy_ant_best_solution['cost']:
                greedy_ant_best_solution = greedy_ant_solution
        self.t_zero = self.get_as_fitness(
            (len(self.nodes) - 1) * greedy_ant_best_solution['cost'])

        # Create real pheromones matrix
        self.matrix_pheromones = self.create_pheromones_matrix(
            self.t_max)
        self.matrix_probabilities = self.create_probabilities_matrix(
            self.matrix_pheromones.copy(),
            self.matrix_heuristics.copy(),
            self.alpha,
            self.beta)

        # Create ants
        ant = self.model_ant(self.nodes,
                             self.demands,
                             self.matrix_probabilities.copy(),
                             self.matrix_costs,
                             self.max_capacity,
                             self.tare,
                             self.model_problem,
                             self.q0)

        # Set iteration local search method
        ls_it = None
        if self.model_ls_it:
            ls_it = self.model_ls_it(self.matrix_costs,
                                     self.demands,
                                     self.tare,
                                     self.max_capacity,
                                     self.k_optimal,
                                     self.max_iterations,
                                     self.model_problem)

        # Solve parameters
        best_solutions = []
        candidate_nodes_weights = None
        global_best_solution = {'cost': np.inf, 'routes_arcs': [
        ], 'routes_costs': [], 'routes_loads': [], 'routes': []}
        iterations_mean_costs = []
        iterations_median_costs = []
        iterations_std_costs = []
        iterations_times = []
        max_outputs_to_print = 10
        outputs_to_print = []
        start_time = time.time()

        # Loop over max_iterations
        with tqdm(total=self.max_iterations) as pbar:
            for it in range(self.max_iterations):
                pbar.set_description('Global Best: {}'
                                     .format('{:.5f}'.format(
                                         global_best_solution['cost'])
                                     ))
                pbar.update(1)

                iterations_solutions = []

                # Generate solutions for each ant and update pheromones matrix
                for _ in range(self.ants_num):
                    ant_solution = ant.generate_solution(
                        candidate_nodes_weights)
                    iterations_solutions.append(ant_solution)

                    # Update pheromones matrix with local update
                    if self.pheromones_local_update and \
                            len(ant_solution['routes']) == self.k_optimal:
                        self.matrix_pheromones = self.add_pheromones_to_matrix(
                            self.matrix_pheromones,
                            ant_solution['routes_arcs'],
                            1 / self.t_zero,
                            self.epsilon)
                        self.matrix_pheromones = \
                            self.evaporate_pheromones_matrix(
                                self.matrix_pheromones,
                                1 - self.epsilon)

                        self.apply_bounds_to_pheromones_matrix(
                            self.t_min, self.t_max)

                        # Update probabilities matrix
                        self.matrix_probabilities = \
                            self.create_probabilities_matrix(
                                self.matrix_pheromones.copy(),
                                self.matrix_heuristics.copy(),
                                self.alpha,
                                self.beta)
                        ant.set_probabilities_matrix(
                            self.matrix_probabilities.copy())

                # Sort solutions by fitness and filter by k_optimal
                iterations_solutions_sorted = sorted(iterations_solutions,
                                                     key=lambda d: d['cost'])
                iterations_solutions_sorted_and_restricted = [
                    solution for solution in iterations_solutions_sorted
                    if len(solution['routes']) == self.k_optimal]

                # Select best and worst solutions and compute relative costs
                iteration_best_solution = {'cost': np.inf, 'routes_arcs': [
                ], 'routes_costs': [], 'routes_loads': [], 'routes': []}
                iteration_worst_solution = iterations_solutions_sorted[-1]
                if iterations_solutions_sorted_and_restricted:
                    iteration_best_solution = \
                        iterations_solutions_sorted_and_restricted[0]
                else:
                    iteration_best_solution = iterations_solutions_sorted[0]

                # Calculate relative costs
                costs_median = np.median(
                    [solution['cost'] for solution
                     in iterations_solutions_sorted])
                costs_mean = np.mean(
                    [solution['cost'] for solution
                     in iterations_solutions_sorted])
                costs_std = np.std(
                    [solution['cost'] for solution
                     in iterations_solutions_sorted])

                # Update iteration output
                iteration_output = [
                    '\n\t> Iteration results: BEST({}), WORST({})'
                    .format(iteration_best_solution['cost'],
                            iteration_worst_solution['cost']),
                    '\t                     MED({}), AVG({}), STD({})\n'
                    .format(costs_median,
                            costs_mean,
                            costs_std)
                ]

                # LS on best iteration solution
                ls_it_solution = {'cost': np.inf, 'routes_arcs': [
                ], 'routes_costs': [], 'routes_loads': [], 'routes': []}
                if ls_it:
                    ls_it_solution = ls_it.improve(
                        iteration_best_solution['routes'], it)
                    iteration_output[0] += ', LS({})'.format(
                        ls_it_solution['cost'])

                # Update global best solution if LS best solution is better
                # or iteration best solution is better
                if ls_it_solution['cost'] < global_best_solution['cost']:
                    global_best_solution = ls_it_solution
                elif iteration_best_solution['cost'] < \
                        global_best_solution['cost']:
                    global_best_solution = iteration_best_solution

                # Update pheromone matrix by global best
                self.matrix_pheromones = self.add_pheromones_to_matrix(
                    self.matrix_pheromones,
                    global_best_solution['routes_arcs'],
                    global_best_solution['cost'],
                    self.p)

                # Evaporate pheromones
                self.matrix_pheromones = self.evaporate_pheromones_matrix(
                    self.matrix_pheromones,
                    self.evaporation_rate)

                # Apply bounds to pheromones matrix
                self.matrix_pheromones = \
                    self.apply_bounds_to_pheromones_matrix(self.t_min,
                                                           self.t_max)

                # Update probabilities matrix
                self.matrix_probabilities = self.create_probabilities_matrix(
                    self.matrix_pheromones.copy(),
                    self.matrix_heuristics.copy(),
                    self.alpha,
                    self.beta)
                ant.set_probabilities_matrix(self.matrix_probabilities.copy())

                # Append iteration best solution to list of best solutions
                best_solutions.append(iteration_best_solution)
                iterations_mean_costs.append(costs_mean)
                iterations_median_costs.append(costs_median)
                iterations_std_costs.append(costs_std)
                iterations_times.append(time.time() - start_time)

                # Update candidate nodes weights
                if self.type_candidate_nodes is not None \
                        and len(best_solutions):
                    candidate_nodes_weights = self.get_candidate_nodes_weight(
                        best_solutions, self.type_candidate_nodes)

                # # Print iteration output
                # if self.ipynb:
                #     for line in iteration_output:
                #         print(line)
                # else:
                #     if len(outputs_to_print) == max_outputs_to_print:
                #         outputs_to_print.pop(0)

                #     iteration_output = ['Iteration {}/{}:'.format(
                #         i + 1, self.max_iterations
                #     )] + iteration_output
                #     outputs_to_print.append(iteration_output)
                #     same_line_print(outputs_to_print)

        # Ending the algorithm run
        final_time = time.time()
        time_elapsed = final_time - start_time

        # Sort best solutions by fitness and filter by k_optimal
        best_solutions_set = []
        best_solutions_fitness = set()
        for ant_solution in sorted(best_solutions, key=lambda d: d['cost']):
            if ant_solution['cost'] not in best_solutions_fitness:
                best_solutions_set.append(ant_solution)
                best_solutions_fitness.add(ant_solution['cost'])

        print(f'\n-- Time elapsed: {time_elapsed} --')

        print('\nBEST SOLUTION FOUND: {}'.format(
            (global_best_solution['cost'],
             global_best_solution['routes'],
             len(global_best_solution['routes']),
             global_best_solution['routes_loads'])))
        print('Best 5 solutions: {}'
              .format([(ant_solution['cost'],
                        len(ant_solution['routes']),
                        ant_solution['routes_loads'])
                       for ant_solution in best_solutions_set][:5]))

        return {
            'best_solutions': best_solutions,
            'global_best_solution': global_best_solution,
            'iterations_mean_costs': iterations_mean_costs,
            'iterations_median_costs': iterations_median_costs,
            'iterations_std_costs': iterations_std_costs,
            'iterations_times': iterations_times,
            'total_time': time_elapsed
        }
