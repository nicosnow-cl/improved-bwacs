from typing import Any, List, Tuple
import numpy as np
import time

from ..helpers import get_flattened_list, same_line_print
from ..models import ProblemModel


class ACS:
    alpha: float
    ants_num: int
    beta: float
    delta: float
    demands_array: np.ndarray
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
    model_local_search: Any
    model_problem: ProblemModel
    nodes: List[int]
    normalized_matrix_heuristics: np.ndarray
    p: float
    q0: float
    t_delta: float
    t_max: float
    t_min: float
    tare: float

    def __init__(self, **kwargs):
        self.ipynb = False
        self.__dict__.update(kwargs)

        self.evaporation_rate = (1 - self.p)

    def create_pheromones_matrix(self, t_delta: float = 0.1) -> np.ndarray:
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

        return np.full((len(self.nodes), len(self.nodes)), t_delta)

    def get_t_delta(self, matrix_costs: np.ndarray) -> float:
        """
        Calculates the initial value of the pheromone trail levels.

        Parameters:
            matrix_costs (np.ndarray): A matrix of the costs between each pair
            of nodes.

        Returns:
            The initial value of the pheromone trail levels.
        """

        return 1 / (self.ants_num * matrix_costs.sum())

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

        t_max = (1 / self.evaporation_rate) * (1 / best_solution_quality)

        max_pheromone = self.matrix_pheromones.max()
        max_probability = self.matrix_probabilities.max()
        n_root_probabilitiy = max_probability ** (1 / self.ants_num)

        # a = (2 * max_pheromone) / ((self.ants_num - 2) * n_root_probabilitiy)
        # b = (2 * max_pheromone) * (1 - n_root_probabilitiy)

        a = (2 * max_pheromone) * (1 - n_root_probabilitiy)
        b = n_root_probabilitiy * (self.ants_num - 2)

        t_min = a / b

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

    def evaporate_pheromones_matrix(self) -> None:
        """
        Evaporates the pheromone trail levels in the pheromone matrix.

        Parameters:
            None.

        Returns:
            None.
        """

        self.matrix_pheromones *= self.evaporation_rate

    def update_pheromones_matrix(self, solution_arcs, solution_quality):
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

        pheromones_amout = self.p * self.get_acs_fitness(solution_quality)

        for arcs_idxs in solution_arcs:
            self.matrix_pheromones[arcs_idxs[:, 0],
                                   arcs_idxs[:, 1]] += pheromones_amout

    def get_mutation_intensity(self,
                               iteration: int,
                               restart_iteration: int) -> float:
        """
        Calculates the mutation intensity for a given iteration.

        Parameters:
            iteration: The current iteration number.

            restart_iteration: The iteration number when a restart is
            performed.

            max_iteration: The maximum iteration number.

        Returns:
            The mutation intensity for the given iteration as a float.
        """

        a = (iteration - restart_iteration)
        b = (self.max_iterations - restart_iteration)

        return (a / b) * self.delta

    def get_t_threshold(self,
                        global_best_solution_arcs: List[np.ndarray]) -> float:
        """
        Calculates the average pheromone level for the edges in the global
        best solution.

        Parameters:
            global_best_solution_arcs: A list of NumPy arrays containing the
            arcs that make up the edges of the global best solution.

        Returns:
            The average pheromone level for the edges in the global best
            solution.
        """

        plain_arcs = get_flattened_list(global_best_solution_arcs)
        pheromones = []

        for i, j in plain_arcs:
            pheromones.append(self.matrix_pheromones[i][j])

        return np.mean(pheromones)

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

    def get_normalized_matrix(self, matrix):
        mask = (matrix != 0) & np.isfinite(matrix)

        with np.errstate(divide='ignore'):  # ignore division by zero warnings
            return np.divide(1, matrix, out=np.zeros_like(matrix), where=mask)

    def get_probabilities_matrix(self):
        """
        Get the updated matrix of probabilities of choosing an arc.

        Parameters:
            None.

        Returns:
            None.
        """

        return np.multiply(np.power(self.matrix_pheromones, self.alpha),
                           np.power(self.normalized_matrix_heuristics,
                                    self.beta))

    def run(self):
        self.normalized_matrix_heuristics = self.get_normalized_matrix(
            self.matrix_heuristics)  # candidate to go on PROBLEM MODEL
        self.t_delta = self.get_t_delta(self.matrix_costs)
        self.matrix_pheromones = self.create_pheromones_matrix(self.t_delta)
        self.matrix_probabilities = self.get_probabilities_matrix()

        ant = self.model_ant(self.nodes,
                             self.demands_array,
                             self.matrix_probabilities,
                             self.matrix_costs,
                             self.max_capacity,
                             self.tare,
                             self.q0,
                             self.model_problem)

        global_best_solution = (None, np.inf, None, None, None)
        best_solutions = []

        max_outputs_to_print = 10
        outputs_to_print = []
        start_time = time.time()

        # Loop over max_iterations
        for i in range(self.max_iterations):
            iterations_solutions = []

            # Generate solutions for each ant
            for _ in range(self.ants_num):
                iterations_solutions.append(ant.generate_solution())

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

            # Evaporate pheromones and update pheromone matrix
            self.evaporate_pheromones_matrix()
            self.update_pheromones_matrix(global_best_solution[2],
                                          global_best_solution[1])

            # Update pheromone matrix bounds and probability matrix
            if i > 0:
                self.t_min, self.t_max = self.calculate_t_min_t_max(
                    global_best_solution[1])
                self.set_bounds_to_pheromones_matrix()
            self.matrix_probabilities = self.get_probabilities_matrix()
            ant.set_probabilities_matrix(self.matrix_probabilities)

            # Append iteration best solution to list of best solutions
            best_solutions.append(iteration_best_solution)

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
             len(global_best_solution[0]),
             global_best_solution[4])))
        print('Best 5 solutions: {}'
              .format([(ant_solution[1], len(ant_solution[0]), ant_solution[4])
                       for ant_solution in best_solutions_set][:5]))

        # print(f't_min: {self.t_min} | t_max: {self.t_max}')
        # print(f'Pheromones min: {self.matrix_pheromones.min()}')
        # print(f'Pheromones max: {self.matrix_pheromones.max()}')
        # print(sorted(np.unique(self.matrix_pheromones)))
