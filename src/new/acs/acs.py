from math import exp, log
from sklearn.preprocessing import MinMaxScaler
from typing import Any, List, Tuple
import numpy as np
import random
import time

from ..helpers import same_line_print, get_flattened_list, get_inversed_matrix
from ..models import ProblemModel


class ACS:
    alpha: float
    ants_num: int
    arcs_clusters_importance: float
    arcs_clusters_lst: List[List[Tuple]]
    beta: float
    demands_array: np.ndarray
    evaporation_rate: float
    ipynb: bool
    k_optimal: int
    local_pheromone_update: bool
    matrix_costs: np.ndarray
    matrix_heuristics: np.ndarray
    matrix_pheromones: np.ndarray
    matrix_probabilities: np.ndarray
    max_capacity: float
    max_iterations: int
    model_ant: Any
    model_local_search: Any
    model_problem: ProblemModel
    nodes: List[int]
    p: float
    q0: float
    t_delta: float
    t_max: float
    t_min: float
    tare: float
    work_with_candidate_nodes: bool

    def __init__(self, **kwargs):
        self.arcs_clusters_importance = 1.5
        self.arcs_clusters_lst = None
        self.ipynb = False
        self.local_pheromone_update = False
        self.work_with_candidate_nodes = False
        self.t_min = 0
        self.t_max = 1

        self.__dict__.update(kwargs)

        self.evaporation_rate = (1 - self.p)

    def print_intance_parameters(self):
        print('\nPARAMETERS:')
        print('\tk_optimal:', self.k_optimal)
        print('\tants_num:', self.ants_num)
        print('\tmax_iterations:', self.max_iterations)
        print('\tmax_capacity:', self.max_capacity)
        print('\ttare:', self.tare)
        print('\talpha:', self.alpha)
        print('\tbeta:', self.beta)
        print('\tp:', self.p)
        print('\tq0:', self.q0)

    def create_pheromones_matrix(self, t_delta: float = 0.001) -> np.ndarray:
        """
        Creates the initial matrix of pheromone trail levels.

        Parameters:
            t_delta (float, optional): The initial value of the pheromone
            trail levels.

        Returns:
            A matrix of pheromone trail levels with all values initialized to
            t_delta.
            The matrix has shape (num_nodes, num_nodes), where num_nodes
            is the number of nodes in the problem.
        """

        shape = len(self.nodes)
        matrix_pheromones = np.full((shape, shape), self.t_delta)

        if self.arcs_clusters_lst:
            num_clusters = len(self.arcs_clusters_lst)
            clusters_factor = 1 + \
                (self.arcs_clusters_importance / num_clusters)

            clusters_arcs_flattened = []
            for clusters_arcs in self.arcs_clusters_lst:
                clusters_arcs_flattened += get_flattened_list(clusters_arcs)

            for i, j in clusters_arcs_flattened:
                matrix_pheromones[i][j] *= clusters_factor

        # if self.arcs_clusters_lst:
        #     matrix_pheromones = np.full((shape, shape), self.t_max)
        #     clusters_arcs_flattened = []

        #     for clusters_arcs in self.arcs_clusters_lst:
        #         clusters_arcs_flattened += get_flattened_list(clusters_arcs)

        #     for i in range(shape):
        #         for j in range(shape):
        #             if (i, j) not in clusters_arcs_flattened:
        #                 matrix_pheromones[i][j] *= self.evaporation_rate

        return matrix_pheromones

    def get_initial_t_delta(self, matrix_costs: np.ndarray) -> float:
        """
        Calculates the initial value of the pheromone trail levels.

        Parameters:
            matrix_costs (np.ndarray): A matrix of the costs between each pair
            of nodes.

        Returns:
            The initial value of the pheromone trail levels.
        """

        return 1 / (len(self.nodes) * matrix_costs.max())

    def calculate_t_min_t_max(self,
                              best_solution_quality: float) \
            -> Tuple[float, float]:
        """
        Calculates the minimum and maximum values of the pheromone trail
        levels.

        Parameters:
            best_quality (float): The quality of the best solution found so
            far.

        Returns:
            A tuple containing the minimum and maximum values of the pheromone
            trail levels.
        """

        n = len(self.nodes)

        t_max = 1 / (self.p * best_solution_quality)
        t_min = t_max * (1 - (0.05) ** (1 / n)) / \
            ((n / 2 - 1) * (0.05) ** (1 / n))

        return t_min, t_max

    def calculate_t_min_t_max_mmas(self, best_solution_quality: float):
        n = len(self.nodes)
        avg = n / 2
        p_best = 0.05
        p_best_n_root = exp(log(p_best) / n)

        t_max = (1 / (1 - self.p)) * \
            self.get_acs_fitness(best_solution_quality)

        upper = t_max * (1 - p_best_n_root)
        lower = (avg - 1) * p_best_n_root

        t_min = upper / lower

        return t_min, t_max

    def get_acs_fitness(self, solutin_quality: float) -> float:
        """
        Calculates the quality of a solution as the inverse of the sum of its
        costs.

        Parameters:
            solution_costs (List[float]): A list of the costs of each tour in
            the solution.

        Returns:
            The quality of the solution as a float. The higher the value, the
            better the solution.
        """

        return 1 / solutin_quality

    def evaporate_pheromones_matrix(self,
                                    evaporation_rate: float = None) -> None:
        """
        Evaporates the pheromone trail levels in the pheromone matrix.

        Parameters:
            None.

        Returns:
            None.
        """

        self.matrix_pheromones *= self.evaporation_rate if evaporation_rate \
            is None else evaporation_rate

    def update_pheromones_matrix(self,
                                 solution_arcs,
                                 solution_quality,
                                 factor=1):
        """
        Updates the pheromones matrix based on the given solution arcs and
        quality.

        Parameters:
            solution_arcs (List[np.ndarray]): List of 2D numpy arrays
            containing the arcs used in the solution.

            solution_quality (float): The quality of the solution.

        Returns:
            None.
        """

        pheromones_amount = factor * self.get_acs_fitness(solution_quality)

        for arcs_idxs in solution_arcs:
            self.matrix_pheromones[arcs_idxs[:, 0],
                                   arcs_idxs[:, 1]] += pheromones_amount

    def set_bounds_to_pheromones_matrix(self) -> None:
        """
        Sets the minimum and maximum values for the pheromone trail levels,
        based on the values of t_min and t_max.

        Parameters:
            None.

        Returns:
            None.
        """

        np.clip(self.matrix_pheromones, self.t_min,
                self.t_max, out=self.matrix_pheromones)

    def get_normalized_matrix(self, matrix: np.ndarray) -> np.ndarray:
        mask = (matrix != 0) & np.isfinite(matrix)

        with np.errstate(divide='ignore'):  # ignore division by zero warnings
            return np.divide(1, matrix, out=np.zeros_like(matrix), where=mask)

    def get_probabilities_matrix(self, pheromones_matrix: np.ndarray) -> np.ndarray:
        """
        Get the updated matrix of probabilities of choosing an arc.

        Parameters:
            None.

        Returns:
            A matrix(ndarray) of probabilities of choosing an arc.
        """

        # inv_distances_matrix = get_inversed_matrix(
        #     self.matrix_costs)
        # min_not_zero_value = inv_distances_matrix[
        #     inv_distances_matrix != 0].min()
        # max_value = inv_distances_matrix[inv_distances_matrix != np.inf].max()

        # # Here we normalice the values between min distance and max distance.
        # scaler = MinMaxScaler(feature_range=(min_not_zero_value, max_value))
        # norm_matrix_pheromones = scaler.fit_transform(self.matrix_pheromones)

        return np.multiply(np.power(pheromones_matrix, self.alpha),
                           #    np.power(self.matrix_heuristics, self.beta))
                           self.matrix_heuristics)

    def get_candidate_starting_nodes(self, solutions):
        """
        Returns a list of candidate starting nodes for the ants, biased
        towards the best starting nodes from the given solutions.

        Parameters:
            solutions(list): A list of solutions to the TSP problem, each
            represented as a tuple of a list of arcs and their
            corresponding cost.

        Returns:
            list: A list of candidate starting nodes for the ants.
        """

        solutions_sorted = sorted(
            filter(lambda sol: sol[1], solutions), key=lambda x: x[1])
        best_starting_nodes = {route[1]
                               for solution in solutions_sorted[:10]
                               for route in solution[0]}
        weights = {node: 1.5 if node in best_starting_nodes else 1
                   for node in self.nodes}

        return random.choices(self.nodes, weights=weights.values(),
                              k=self.ants_num)

    def run(self):
        # Starting initial matrixes
        self.t_delta = self.get_initial_t_delta(self.matrix_costs)
        self.matrix_pheromones = self.create_pheromones_matrix(self.t_delta)
        self.matrix_probabilities = self.get_probabilities_matrix()

        # Greedy ants to find the best initial solution
        greedy_ant = self.model_ant(self.nodes,
                                    self.demands_array,
                                    self.matrix_probabilities,
                                    self.matrix_costs,
                                    self.max_capacity,
                                    self.tare,
                                    self.q0,
                                    self.model_problem)

        greedy_ant_best_solution = (None, np.inf, None, None, None)
        for _ in range(self.ants_num):
            solution = greedy_ant.generate_solution()
            if solution[1] < greedy_ant_best_solution[1]:
                greedy_ant_best_solution = solution

        # Initial t_min and t_max and new t_delta
        self.t_min, self.t_max = self.calculate_t_min_t_max_mmas(
            greedy_ant_best_solution[1])
        self.t_delta = (self.t_min + self.t_max) / 2
        self.matrix_pheromones = self.create_pheromones_matrix(self.t_delta)
        self.matrix_probabilities = self.get_probabilities_matrix()

        # Create ants
        ant = self.model_ant(self.nodes,
                             self.demands_array,
                             self.matrix_probabilities,
                             self.matrix_costs,
                             self.max_capacity,
                             self.tare,
                             self.q0,
                             self.model_problem)

        best_solutions = []
        candidate_starting_nodes = []
        global_best_solution = (None, np.inf, None, None, None)
        max_outputs_to_print = 10
        outputs_to_print = []
        start_time = time.time()

        # Loop over max_iterations
        for i in range(self.max_iterations):
            iterations_solutions = []

            # Generate solutions for each ant and update pheromones matrix
            for _ in range(self.ants_num):
                solution = ant.generate_solution(candidate_starting_nodes)
                iterations_solutions.append(solution)

                # Local pheromone update
                if self.local_pheromone_update and \
                        len(solution[0]) == self.k_optimal:
                    ant_factor = (1 - self.p) / self.ants_num

                    # Update pheromones matrix with local update
                    self.update_pheromones_matrix(
                        solution[2], solution[1], ant_factor)
                    self.set_bounds_to_pheromones_matrix()

                    # Update probabilities matrix
                    self.matrix_probabilities = self.get_probabilities_matrix()
                    ant.set_probabilities_matrix(self.matrix_probabilities)

            # Sort solutions by fitness and filter by k_optimal
            iterations_solutions_sorted = sorted(iterations_solutions,
                                                 key=lambda d: d[1])
            iterations_solutions_sorted_and_restricted = [
                solution for solution in iterations_solutions_sorted
                if len(solution[0]) == self.k_optimal]

            # Select best and worst solutions and compute average cost
            if iterations_solutions_sorted_and_restricted:
                iteration_best_solution = \
                    iterations_solutions_sorted_and_restricted[0]
            else:
                iteration_best_solution = iterations_solutions_sorted[0]
            iteration_worst_solution = iterations_solutions_sorted[-1]
            average_iteration_costs = np.mean(
                [solution[1] for solution in iterations_solutions_sorted])

            # Update iteration output
            iteration_output = [
                f'Iteration {i + 1}',
                '    > Iteration resoluts: BEST({}), WORST({}), AVG({})'
                .format(iteration_best_solution[1],
                        iteration_worst_solution[1],
                        average_iteration_costs)
            ]

            # Update global best solution if iteration best solution is better
            if iteration_best_solution[1] < global_best_solution[1]:
                global_best_solution = iteration_best_solution

            # Evaporate pheromones and update pheromone matrix by global best
            self.evaporate_pheromones_matrix()
            self.update_pheromones_matrix(global_best_solution[2],
                                          global_best_solution[1])

            # Update t_min and t_max and set bounds to pheromones matrix
            self.t_min, self.t_max = self.calculate_t_min_t_max_mmas(
                global_best_solution[1])
            self.set_bounds_to_pheromones_matrix()

            # Update probabilities matrix
            self.matrix_probabilities = self.get_probabilities_matrix()
            ant.set_probabilities_matrix(self.matrix_probabilities)

            # Append iteration best solution to list of best solutions
            best_solutions.append(iteration_best_solution)

            if self.work_with_candidate_nodes:
                candidate_starting_nodes = self.get_candidate_starting_nodes(
                    best_solutions)

            # Print iteration output
            if self.ipynb:
                for line in iteration_output:
                    print(line)
            else:
                if len(outputs_to_print) == max_outputs_to_print:
                    outputs_to_print.pop(0)

                outputs_to_print.append(iteration_output)
                same_line_print(outputs_to_print)

        final_time = time.time()
        time_elapsed = final_time - start_time
        print(f'\nTime elapsed: {time_elapsed}')

        best_solutions_set = []
        best_solutions_fitness = set()
        for solution in sorted(best_solutions, key=lambda d: d[1]):
            if solution[1] not in best_solutions_fitness:
                best_solutions_set.append(solution)
                best_solutions_fitness.add(solution[1])

        print('Best solution: {}'.format(
            (global_best_solution[1],
             global_best_solution[0],
             len(global_best_solution[0]),
             global_best_solution[4])))
        print('Best 5 solutions: {}'
              .format([(ant_solution[1], len(ant_solution[0]), ant_solution[4])
                       for ant_solution in best_solutions_set][:5]))

        # print(f't_min: {self.t_min} | t_max: {self.t_max}')
        # print(f'Pheromones min: {self.matrix_pheromones.min()}')
        # print(f'Pheromones max: {self.matrix_pheromones.max()}')
        # print(sorted(np.unique(self.matrix_pheromones)))
