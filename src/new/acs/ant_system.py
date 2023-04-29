from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
from typing import Any, List, Tuple
import numpy as np
import time

from ..helpers import get_inversed_matrix, same_line_print

MAX_FLOAT = 1.0
MIN_FLOAT = np.finfo(np.float32).min


class AS:
    alpha: float
    ants_num: int
    arcs_clusters_importance: float
    arcs_clusters_lst: List[List[Tuple]]
    beta: float
    demands: List[float]
    evaporation_rate: float
    ipynb: bool
    k_optimal: int
    matrix_costs: np.ndarray
    matrix_heuristics: np.ndarray
    matrix_pheromones: np.ndarray
    matrix_probabilities: np.ndarray
    max_capacity: float
    max_iterations: int
    model_ant: Any
    model_ls_it: Any
    nodes: List[int]
    p: float
    t_max: float
    t_min: float
    tare: float
    type_probabilities_matrix: str

    def __init__(self, **kwargs):
        self.arcs_clusters_importance = 0
        self.arcs_clusters_lst = None
        self.ipynb = False
        self.model_ls_it = None
        self.p = 0.2
        self.evaporation_rate = (1 - self.p)
        self.t_max = MAX_FLOAT
        self.t_min = MIN_FLOAT
        self.type_probabilities_matrix = 'normal'

        self.__dict__.update(kwargs)

    def print_intance_parameters(self):
        print('\nPARAMETERS:')
        print('\talpha:', self.alpha)
        print('\tants_num:', self.ants_num)
        print('\tarcs_clusters_importance:', self.arcs_clusters_importance)
        print('\tbeta:', self.beta)
        print('\tdemands:', self.demands)
        print('\tevaporation_rate:', self.evaporation_rate)
        print('\tk_optimal:', self.k_optimal)
        print('\tmax_capacity:', self.max_capacity)
        print('\tmax_iterations:', self.max_iterations)
        print('\tp:', self.p)
        print('\ttare:', self.tare)
        print('\ttype_probabilities_matrix:', self.type_probabilities_matrix)
        print('\n')

    def create_pheromones_matrix(self, t_max: float = MAX_FLOAT) -> np.ndarray:
        """
        Creates the initial matrix of pheromone trail levels.

        Args:
            t_max (float, optional): The maximum value of the pheromone
            trail levels.

        Returns:
            A matrix of pheromone trail levels with all values initialized to
            t_delta. The matrix has shape (num_nodes, num_nodes), where
            num_nodes is the number of nodes in the problem.
        """

        shape = len(self.nodes)
        matrix_pheromones = np.full((shape, shape), t_max)

        return matrix_pheromones

    def get_as_fitness(self, solution_cost: float) -> float:
        """
        Calculates the quality of a solution as the inverse of the sum of its
        costs.

        Args:
            solution_cost (float): The cost of the solution.

        Returns:
            The quality of the solution.
        """

        return 1 / solution_cost

    def evaporate_pheromones_matrix(self,
                                    pheromones_matrix: np.ndarray,
                                    evaporation_rate: float = None) \
            -> np.ndarray:
        """
        Evaporates the pheromone trail levels in the pheromone matrix.

        Args:
            pheromones_matrix (np.ndarray): The matrix of pheromone trail
            levels.
            evaporation_rate (float, optional): The evaporation rate.

        Returns:
            The pheromone matrix after the evaporation.
        """

        pheromones_matrix *= self.evaporation_rate if evaporation_rate \
            is None else evaporation_rate

        return pheromones_matrix

    def add_pheromones_to_matrix(self,
                                 pheromones_matrix: np.ndarray,
                                 solution_arcs: List[Tuple],
                                 solution_quality: float,
                                 factor=1) -> np.ndarray:
        """
        Adds pheromone trail levels to the pheromone matrix.

        Args:
            pheromones_matrix (np.ndarray): The matrix of pheromone trail
            solution_arcs (List[Tuple]): The arcs of the solution.
            solution_quality (float): The quality of the solution.
            factor (float, optional): The factor to multiply the pheromone
            trail levels of the solution.

        Returns:
            The pheromone matrix after the addition.
        """

        pheromones_amount = self.get_as_fitness(solution_quality) * factor
        for arcs_lst in solution_arcs:
            for arc in arcs_lst:
                i, j = arc
                pheromones_matrix[i][j] += pheromones_amount

        return pheromones_matrix

    def apply_bounds_to_pheromones_matrix(self,
                                          t_min: float = MIN_FLOAT,
                                          t_max: float = MAX_FLOAT) \
            -> np.ndarray:
        """
        Applies the bounds to the pheromone matrix.

        Args:
            t_min (float, optional): The minimum value of the pheromone trail
            levels.
            t_max (float, optional): The maximum value of the pheromone trail
            levels.

        Returns:
            The pheromone matrix after the bounds application.
        """

        return np.clip(self.matrix_pheromones, t_min, t_max)

    def create_probabilities_matrix(self,
                                    pheromones_matrix: np.ndarray,
                                    heuristics_matrix: np.ndarray,
                                    alpha: float = 1.0,
                                    beta: float = 2.0) -> np.ndarray:
        """
        Creates the probabilities matrix.

        Args:
            pheromones_matrix (np.ndarray): The matrix of pheromone trail
            levels.
            heuristics_matrix (np.ndarray): The matrix of heuristic values.
            alpha (float, optional): The alpha parameter.
            beta (float, optional): The beta parameter.

        Returns:
            The probabilities matrix.
        """

        if self.type_probabilities_matrix == 'normalized':
            inv_distances_matrix = get_inversed_matrix(
                self.matrix_costs)
            min_not_zero_value = inv_distances_matrix[
                inv_distances_matrix != 0].min()
            max_value = \
                inv_distances_matrix[inv_distances_matrix != np.inf].max()

            # Here we normalice the values between min distance
            # and max distance.
            scaler = MinMaxScaler(feature_range=(
                min_not_zero_value, max_value))
            norm_matrix_pheromones = scaler.fit_transform(pheromones_matrix)

            return np.multiply(np.power(norm_matrix_pheromones, self.alpha),
                               self.matrix_heuristics)
        else:
            return np.multiply(np.power(pheromones_matrix, self.alpha),
                               self.matrix_heuristics)

    def solve(self):
        """
        Solves the problem.
        """

        self.print_intance_parameters()

        errors = self.model_problem.validate_instance(
            self.nodes, self.demands, self.max_capacity)
        if errors:
            raise Exception(errors)

        # Starting initial matrixes
        self.matrix_pheromones = self.create_pheromones_matrix(self.t_max)
        self.matrix_probabilities = self.create_probabilities_matrix(
            self.matrix_pheromones.copy(), self.matrix_heuristics.copy())

        # Create ants
        ant = self.model_ant(self.nodes,
                             self.demands,
                             self.matrix_probabilities.copy(),
                             self.matrix_costs,
                             self.max_capacity,
                             self.tare,
                             self.model_problem)

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
        global_best_solution = (None, np.inf, None, None, None)
        max_outputs_to_print = 10
        outputs_to_print = []
        start_time = time.time()

        # Loop over max_iterations
        with tqdm(total=self.max_iterations) as pbar:
            for it in range(self.max_iterations):
                pbar.set_description('Global Best: {}'
                                     .format('{:.5f}'.format(
                                         global_best_solution[1])
                                     ))
                pbar.update(1)

                iterations_solutions = []

                # Generate solutions for each ant and update pheromones matrix
                for _ in range(self.ants_num):
                    solution = ant.generate_solution()
                    iterations_solutions.append(solution)

                # Sort solutions by fitness and filter by k_optimal
                iterations_solutions_sorted = sorted(iterations_solutions,
                                                     key=lambda d: d[1])
                iterations_solutions_sorted_and_restricted = [
                    solution for solution in iterations_solutions_sorted
                    if len(solution[0]) == self.k_optimal]

                # Select best and worst solutions and compute relative costs
                iteration_best_solution = (None, np.inf, None, None, None)
                iteration_worst_solution = iterations_solutions_sorted[-1]
                if iterations_solutions_sorted_and_restricted:
                    iteration_best_solution = \
                        iterations_solutions_sorted_and_restricted[0]
                else:
                    iteration_best_solution = iterations_solutions_sorted[0]

                # Calculate relative costs
                costs_median = np.median(
                    [solution[1] for solution in iterations_solutions_sorted])
                costs_mean = np.mean(
                    [solution[1] for solution in iterations_solutions_sorted])
                costs_std = np.std(
                    [solution[1] for solution in iterations_solutions_sorted])

                # Update iteration output
                iteration_output = [
                    '\n\t> Iteration results: BEST({}), WORST({})'
                    .format(iteration_best_solution[1],
                            iteration_worst_solution[1]),
                    '\t                     MED({}), AVG({}), STD({})\n'
                    .format(costs_median,
                            costs_mean,
                            costs_std)
                ]

                # LS on best iteration solution
                ls_it_solution = (None, np.inf, None, None, None)
                if ls_it:
                    ls_it_solution = ls_it.improve(
                        iteration_best_solution[0], it)
                    iteration_output[0] += ', LS({})'.format(ls_it_solution[1])

                # Update global best solution if LS best solution is better
                # or iteration best solution is better
                if ls_it_solution[1] < global_best_solution[1]:
                    global_best_solution = ls_it_solution
                elif iteration_best_solution[1] < global_best_solution[1]:
                    global_best_solution = iteration_best_solution

                # Update pheromones matrix by individual ant
                for solution in iterations_solutions:
                    self.matrix_pheromones = self.add_pheromones_to_matrix(
                        self.matrix_pheromones,
                        solution[2],
                        solution[1])

                # Evaporate pheromones matrix
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

                # # Print iteration output
                # if self.ipynb:
                #     for line in iteration_output:
                #         print(line)
                # else:
                #     if len(outputs_to_print) == max_outputs_to_print:
                #         outputs_to_print.pop(0)

                #     iteration_output = ['Iteration {}/{}:'.format(
                #         it + 1, self.max_iterations
                #     )] + iteration_output
                #     outputs_to_print.append(iteration_output)
                #     same_line_print(outputs_to_print)

        # Ending the algorithm run
        final_time = time.time()
        time_elapsed = final_time - start_time

        # Sort best solutions by fitness and filter by k_optimal
        best_solutions_set = []
        best_solutions_fitness = set()
        for solution in sorted(best_solutions, key=lambda d: d[1]):
            if solution[1] not in best_solutions_fitness:
                best_solutions_set.append(solution)
                best_solutions_fitness.add(solution[1])

        print(f'\n-- Time elapsed: {time_elapsed} --')

        print('\nBEST SOLUTION FOUND: {}'.format(
            (global_best_solution[1],
             global_best_solution[0],
             len(global_best_solution[0]),
             global_best_solution[4])))
        print('Best 5 solutions: {}'
              .format([(ant_solution[1], len(ant_solution[0]), ant_solution[4])
                       for ant_solution in best_solutions_set][:5]))

        return (global_best_solution,
                best_solutions,
                costs_mean,
                costs_median,
                costs_std,
                time_elapsed)
