from tqdm import tqdm
from typing import List
import numpy as np
import time

from ..helpers import get_flattened_list, same_line_print
from .ant_colony_system import ACS


class BWACS(ACS):
    delta: int
    p_m: float
    percent_arcs_limit: float
    percent_quality_limit: float
    type_mutation: str

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.delta = 2
        self.p_m = 0.3
        self.type_mutation = None

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

        global_best_solution_arcs_set = get_flattened_list(
            global_best_solution_arcs, tuple)

        for route in current_worst_solution_arcs:
            for i, j in route:
                if (i, j) not in global_best_solution_arcs_set:
                    self.matrix_pheromones[i][j] *= self.evaporation_rate

    def mutate_pheromones_matrix_by_row(self,
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
        mutation_value = (mutation_intensity * t_threshold) * 0.0001

        for i in range(self.matrix_pheromones.shape[0]):
            if np.random.rand() < self.p_m:
                mutation_value *= np.random.choice([-1, 1])

                self.matrix_pheromones[i] += mutation_value

    def mutaute_pheromones_matrix_by_arcs(self,
                                          solution_arcs,
                                          current_iteration,
                                          restart_iteration):
        mutation_intensity = self.get_mutation_intensity(current_iteration,
                                                         restart_iteration)
        t_threshold = self.get_t_threshold(solution_arcs)

        mutation_value = (self.p * mutation_intensity * t_threshold) \
            * 0.0001

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

    def is_stagnation_reached_by_arcs(self,
                                      it_best_solution_arcs,
                                      it_worst_solution_arcs,
                                      similarity_percentage: float):
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
        arcs_similarity = (a / b)

        return arcs_similarity >= similarity_percentage

    def is_stagnation_reached_by_solutions(self,
                                           best_actual_quality: float,
                                           best_prev_quality: float,
                                           worst_actual_quality: float,
                                           actual_median: float,
                                           prev_median: float,
                                           similarity_percentage: float):
        no_improvements = best_actual_quality >= best_prev_quality

        quality_similarity = (best_actual_quality / worst_actual_quality)
        quality_similarity_reached = quality_similarity > similarity_percentage

        median_similarity = actual_median / prev_median
        median_similarity_reached = median_similarity > similarity_percentage

        return no_improvements and quality_similarity_reached \
            and median_similarity_reached

    def get_avg_steps_between_restarts(self, restarts):
        """
        Calculates the average number of iterations between restarts.

        Args:
            restarts (list): A list of integers representing the iteration
            number at which a restart was performed.

        Returns:
            float: The average number of iterations between restarts.
        """

        if len(restarts) > 1:
            steps = [restarts[i + 1] - restarts[i]
                     for i in range(len(restarts) - 1)]
            return np.mean(steps)
        else:
            return 0

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

        self.print_intance_parameters()

        errors = self.model_problem.validate_instance(
            self.nodes, self.demands, self.max_capacity)
        if errors:
            raise Exception(errors)

        # Starting initial matrixes
        self.t_delta = self.get_initial_t_delta(self.matrix_costs)
        self.matrix_pheromones = self.create_pheromones_matrix(self.t_delta)
        self.matrix_probabilities = self.get_probabilities_matrix(
            self.matrix_pheromones)

        # Greedy ants to find the best initial solution
        greedy_ant = self.model_ant(self.nodes,
                                    self.demands,
                                    self.matrix_probabilities.copy(),
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
        self.matrix_pheromones = self.create_pheromones_matrix(
            self.t_delta, self.t_min, self.t_max)
        self.matrix_probabilities = self.get_probabilities_matrix(
            self.matrix_pheromones.copy())

        # Create ants
        ant = self.model_ant(self.nodes,
                             self.demands,
                             self.matrix_probabilities.copy(),
                             self.matrix_costs,
                             self.max_capacity,
                             self.tare,
                             self.q0,
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

        best_prev_quality = np.inf
        best_solutions = []
        avg_costs = []
        median_costs = []
        candidate_nodes_weights = []
        global_best_solution = (None, np.inf, None, None, None)
        max_outputs_to_print = 10
        outputs_to_print = []
        prev_median = np.inf
        restarts = []
        start_time = time.time()

        # Loop over max_iterations
        with tqdm(total=self.max_iterations) as pbar:
            for i in range(self.max_iterations):
                pbar.set_description('Global Best: {}'
                                     .format('{:.5f}'.format(
                                         global_best_solution[1])
                                     ))
                pbar.update(1)

                iterations_solutions = []

                # Generate solutions for each ant
                for _ in range(self.ants_num):
                    solution = ant.generate_solution(candidate_nodes_weights)
                    iterations_solutions.append(solution)

                    # Local pheromone update
                    if self.pheromones_local_update:
                        local_factor = self.p / len(self.nodes)

                        # Update pheromones matrix with local update
                        self.update_pheromones_matrix(
                            solution[2], solution[1], local_factor)
                        self.set_bounds_to_pheromones_matrix()

                        # Update probabilities matrix
                        self.matrix_probabilities = \
                            self.get_probabilities_matrix(
                                self.matrix_pheromones.copy())
                        ant.set_probabilities_matrix(
                            self.matrix_probabilities.copy())

                # Sort solutions by fitness and filter by k_optimal
                iterations_solutions_sorted = sorted(iterations_solutions,
                                                     key=lambda d: d[1])
                iterations_solutions_sorted_and_restricted = [
                    solution for solution in iterations_solutions_sorted
                    if len(solution[0]) == self.k_optimal]

                # Select best and worst solutions and compute relative costs
                if iterations_solutions_sorted_and_restricted:
                    iteration_best_solution = \
                        iterations_solutions_sorted_and_restricted[0]
                else:
                    iteration_best_solution = iterations_solutions_sorted[0]

                iteration_worst_solution = iterations_solutions_sorted[-1]
                median_iteration_costs = np.median(
                    [solution[1] for solution in iterations_solutions_sorted])
                avg_iteration_costs = np.mean(
                    [solution[1] for solution in iterations_solutions_sorted])
                std_iteration_costs = np.std(
                    [solution[1] for solution in iterations_solutions_sorted])

                # Update iteration output
                iteration_output = [
                    '\n\t> Iteration results: BEST({}), WORST({})'
                    .format(iteration_best_solution[1],
                            iteration_worst_solution[1]),
                    '\t                     MED({}), AVG({}), STD({})\n'
                    .format(median_iteration_costs,
                            avg_iteration_costs,
                            std_iteration_costs)
                ]

                # LS on best iteration solution
                ls_it_solution = (None, np.inf, None, None, None)
                if ls_it:
                    ls_it_solution = ls_it.improve(
                        iteration_best_solution[0][:], i)
                    iteration_output[0] += ', LS({})'.format(ls_it_solution[1])

                # Update global best solution if LS best solution is better
                # or iteration best solution is better
                if ls_it_solution[1] < global_best_solution[1]:
                    global_best_solution = ls_it_solution
                elif iteration_best_solution[1] < global_best_solution[1]:
                    global_best_solution = iteration_best_solution

                # Evaporate pheromones and update pheromone matrix by BWACS
                self.evaporate_pheromones_matrix()
                self.update_pheromones_matrix(global_best_solution[2],
                                              global_best_solution[1])
                self.penalize_pheromones_matrix(global_best_solution[2],
                                                iteration_worst_solution[2])

                # Update t_min and t_max and
                self.t_min, self.t_max = self.calculate_t_min_t_max_mmas(
                    global_best_solution[1])

                # Restart pheromones matrix if stagnation is reached
                if self.percent_arcs_limit:
                    restarts_avg_steps = self.get_avg_steps_between_restarts(
                        restarts)
                    remaining_iterations = self.max_iterations - i

                    if remaining_iterations >= restarts_avg_steps and \
                        self.is_stagnation_reached_by_arcs(
                            iteration_best_solution[2],
                            iteration_worst_solution[2],
                            self.percent_arcs_limit):
                        iteration_output.append('\t* Stagnation detected!!!')
                        self.t_delta = (self.t_min + self.t_max) / 2
                        self.matrix_pheromones = self.create_pheromones_matrix(
                            self.t_delta,
                            self.t_min,
                            self.t_max)
                        restarts.append(i)
                elif self.percent_quality_limit:
                    restarts_avg_steps = self.get_avg_steps_between_restarts(
                        restarts)
                    remaining_iterations = self.max_iterations - i

                    if remaining_iterations >= restarts_avg_steps and \
                        self.is_stagnation_reached_by_solutions(
                            iteration_best_solution[1],
                            best_prev_quality,
                            iterations_solutions_sorted[-1][1],
                            median_iteration_costs,
                            prev_median,
                            self.percent_quality_limit):
                        iteration_output.append('\t* Stagnation detected!!!')
                        self.t_delta = (self.t_min + self.t_max) / 2
                        self.matrix_pheromones = self.create_pheromones_matrix(
                            self.t_delta,
                            self.t_min,
                            self.t_max,
                            best_solutions)
                        restarts.append(i)

                # Mutate pheromones matrix
                if len(restarts):
                    if self.type_mutation == 'arcs':
                        self.mutaute_pheromones_matrix_by_arcs(
                            global_best_solution[2],
                            i, restarts[-1])
                    elif self.type_mutation == 'rows':
                        self.mutate_pheromones_matrix_by_row(
                            global_best_solution[2],
                            i, restarts[-1])

                self.set_bounds_to_pheromones_matrix(self.t_max)

                # Update probabilities matrix
                self.matrix_probabilities = self.get_probabilities_matrix(
                    self.matrix_pheromones.copy())
                ant.set_probabilities_matrix(self.matrix_probabilities.copy())

                # Append iteration best solution to list of best solutions
                best_solutions.append(iteration_best_solution)
                avg_costs.append(avg_iteration_costs)
                median_costs.append(median_iteration_costs)

                # Update best_prev_quality, best_median
                best_prev_quality = iteration_best_solution[1]
                prev_median = median_iteration_costs

                # Update candidate starting nodes
                if self.type_candidate_nodes:
                    candidate_nodes_weights = self.get_candidate_nodes_weight(
                        best_solutions, self.type_candidate_nodes)

                # Print iteration output
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

        best_solutions_set = []
        best_solutions_fitness = set()
        for solution in sorted(best_solutions, key=lambda d: d[1]):
            if solution[1] not in best_solutions_fitness:
                best_solutions_set.append(solution)
                best_solutions_fitness.add(solution[1])

        print(f'\n-- Time elapsed: {time_elapsed} --')
        if len(restarts):
            print('Iterations when do restart: {}'.format(
                [restart_iteration + 1 for restart_iteration in restarts]))

        print('\nBEST SOLUTION FOUND: {}'.format(
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

        return global_best_solution, best_solutions, avg_costs, median_costs
