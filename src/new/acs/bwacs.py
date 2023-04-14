from typing import Any, List
import numpy as np
import time

from .acs import ACS
from ..helpers import get_flattened_list, same_line_print


class BWACS(ACS):
    delta: int
    model_ls_it: Any
    p_m: float
    percentage_of_similarity: float

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.delta = 2
        self.model_ls_it = None
        self.p_m = 0.3
        self.percentage_of_similarity = 50

        self.__dict__.update(kwargs)

    def penalize_pheromones_matrix(self,
                                   global_best_solution_arcs,
                                   current_worst_solution_arcs):
        """
        Decreases the pheromone level on arcs not included in the global best
        solution.

        Parameters:
            global_best_solution_arcs: A list of numpy arrays, each
            representing an arc in the global best solution.

            current_worst_solution_arcs: A list of numpy arrays, each
            representing an arc in the current worst solution.

        Returns:
            None
        """

        global_best_solution_arcs_set = set(
            get_flattened_list(global_best_solution_arcs, tuple))

        for route in current_worst_solution_arcs:
            for i, j in route:
                if (i, j) not in global_best_solution_arcs_set:
                    self.matrix_pheromones[i][j] *= self.evaporation_rate

    def mutate_pheromones_matrix(self,
                                 solution_arcs,
                                 current_iteration,
                                 restart_iteration):
        """
        Mutate the pheromone matrix of the ant colony optimization algorithm
        based on a solution.

        Parameters:
            solution_arcs (list): A list of arcs that represent a solution.

            current_iteration (int): An integer representing the current
            iteration of the algorithm.

            restart_iteration (int): An integer representing the iteration at
            which the algorithm was last restarted.

        Returns:
            None
        """

        mutation_intensity = self.get_mutation_intensity(current_iteration,
                                                         restart_iteration)
        t_threshold = self.get_t_threshold(solution_arcs)

        mutation_value = (self.p * mutation_intensity * t_threshold) * 0.00005

        # Use triu_indices to get upper triangle indices
        iu = np.triu_indices(self.matrix_pheromones.shape[0], k=1)

        # Update elements in upper triangle with random mutations
        mask = np.random.rand(len(iu[0])) < self.p_m
        mut = np.random.choice([-1, 1], size=len(iu[0])) * mutation_value

        self.matrix_pheromones[iu] += mask * mut

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
                        solution_arcs: List[np.ndarray]) -> float:
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

        plain_arcs = get_flattened_list(solution_arcs, tuple)
        pheromones = []

        for i, j in plain_arcs:
            pheromones.append(self.matrix_pheromones[i][j])

        return np.mean(pheromones)

    def reach_stagnation(self,
                         it_best_solution_arcs,
                         it_worst_solution_arcs):
        """
        Determine if the algorithm has reached stagnation based on the current
        and previous best and worst solutions.

        Args:
            it_best_solution_arcs (list): A list of arcs that represent the
            current best solution.

            it_worst_solution_arcs (list): A list of arcs that represent the
            current worst solution.

        Returns:
            bool: True if the percentage of similarity between the current
            best and worst solutions is greater than or equal to the
            predetermined threshold for stagnation, False otherwise.
        """

        it_best_solution_arcs_set = set(
            get_flattened_list(it_best_solution_arcs, tuple))
        it_worst_solution_arcs_set = set(
            get_flattened_list(it_worst_solution_arcs, tuple))

        different_tuples = it_best_solution_arcs_set & \
            it_worst_solution_arcs_set

        a = len(different_tuples)
        b = len(it_best_solution_arcs_set.union(it_worst_solution_arcs_set))
        actual_percentage = (a / b) * 100

        return actual_percentage >= self.percentage_of_similarity

    def run(self):
        """
        Runs the Ant Colony Optimization algorithm.

        It initializes the pheromone matrix, generates `max_iterations`
        solutions using `ants_num` ants, applies the local search method
        (if specified), updates the pheromone matrix and prints relevant
        information at each iteration.

        Parameters
        ----------
        None.

        Returns
        -------
        global_best_solution (tuple): The best solution found by the algorithm.

        global_best_solutions (list[tuple]): A list of the best solutions
        found by the algorithm at each iteration.
        """

        self.normalized_matrix_heuristics = self.get_normalized_matrix(
            self.matrix_heuristics)  # candidate to go on PROBLEM MODEL
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
        self.t_min, self.t_max = self.calculate_t_min_t_max(
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

        ls_it = None
        if self.model_ls_it:
            ls_it = self.model_ls_it(self.matrix_costs,
                                     self.demands_array,
                                     self.tare,
                                     self.max_capacity,
                                     self.k_optimal,
                                     self.max_iterations,
                                     self.model_problem)

        best_solutions = []
        candidate_starting_nodes = []
        global_best_solution = (None, np.inf, None, None, None)
        max_outputs_to_print = 10
        outputs_to_print = []
        restart_iteration = 0
        start_time = time.time()

        # Loop over max_iterations
        for i in range(self.max_iterations):
            iterations_solutions = []

            # Generate solutions for each ant
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

            # LS by VNS on best iteration solution
            if ls_it:
                ls_it_solution = ls_it.improve(iteration_best_solution[0], i)

                if ls_it_solution[1] < iteration_best_solution[1]:
                    iteration_best_solution = ls_it_solution

            # Update global best solution if iteration best solution is better
            if iteration_best_solution[1] < global_best_solution[1]:
                global_best_solution = iteration_best_solution

            # Evaporate pheromones and update pheromone matrix by BWACS
            self.evaporate_pheromones_matrix()
            self.update_pheromones_matrix(global_best_solution[2],
                                          global_best_solution[1])
            self.penalize_pheromones_matrix(global_best_solution[2],
                                            iteration_worst_solution[2])

            # Update t_min and t_max and set bounds to pheromones matrix
            self.t_min, self.t_max = self.calculate_t_min_t_max(
                global_best_solution[1])

            # Mutate pheromone matrix and check stagnation
            if self.reach_stagnation(iteration_best_solution[2],
                                     iteration_worst_solution[2]):
                iteration_output.append('    > Stagnation detected!')
                self.t_delta = (self.t_min + self.t_max) / 2
                self.matrix_pheromones = self.create_pheromones_matrix(
                    self.t_delta)
                restart_iteration = i

            if restart_iteration > 0:
                self.mutate_pheromones_matrix(global_best_solution[2],
                                              i, restart_iteration)

            self.set_bounds_to_pheromones_matrix()

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

        # Ending the algorithm run
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

        if restart_iteration:
            print(f'Last iteration when do restart: {restart_iteration}')

        # print(f't_min: {self.t_min} | t_max: {self.t_max}')
        # print(f'Pheromones min: {self.matrix_pheromones.min()}')
        # print(f'Pheromones max: {self.matrix_pheromones.max()}')
        # print(sorted(np.unique(self.matrix_pheromones)))

        return global_best_solution, best_solutions_set
