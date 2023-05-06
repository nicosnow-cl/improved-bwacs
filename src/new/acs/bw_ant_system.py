from copy import deepcopy
from tqdm import tqdm
from typing import List
import numpy as np
import time

from ..ants import AntSolution
from ..helpers import get_flattened_list
from .aco_solution import ACOSolution
from .mm_ant_system import MMAS


class BWAS(MMAS):
    p_m: float
    sigma: int
    type_mutation: str

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.p_m = 0.2
        self.rho = 0.2
        self.sigma = 4
        self.type_mutation = None

        self.__dict__.update(kwargs)

        self.evaporation_rate = 1 - self.rho

    def print_intance_parameters(self):
        super().print_intance_parameters()

        print("----------------------------------------")
        print("BWAS:")
        print("\tp_m:", self.p_m)
        print("\tsigma:", self.sigma)
        print("\ttype_mutation:", self.type_mutation)

    def penalize_pheromones_matrix(
        self,
        pheromones_matrix: np.ndarray,
        gb_solution_arcs: List[np.ndarray],
        curr_worst_solution_arcs: List[np.ndarray],
        evaporation_rate: float = None,
    ) -> np.ndarray:
        """
        Decreases the pheromone level on arcs not included in the global best
        solution.

        Args:
            pheromones_matrix (np.ndarray): A NumPy array representing the
            pheromone matrix.
            gb_solution_arcs (list): A list of NumPy arrays containing
            the arcs that make up the edges of the global best solution.
            curr_worst_solution_arcs (list): A list of NumPy arrays
            containing the arcs that make up the edges of the current worst
            solution.
            p (float): A float representing the pheromone decay coefficient.

        Returns:
            A NumPy array representing the updated pheromone matrix.
        """

        pheromones_matrix_copy = pheromones_matrix.copy()
        ev_rate = (
            self.evaporation_rate
            if evaporation_rate is None
            else evaporation_rate
        )

        gb_solution_arcs_flattened = get_flattened_list(
            gb_solution_arcs, elem_type=tuple
        )
        curr_worst_solution_arcs_flattened = get_flattened_list(
            curr_worst_solution_arcs, elem_type=tuple
        )

        for i, j in curr_worst_solution_arcs_flattened:
            if (i, j) not in gb_solution_arcs_flattened:
                pheromones_matrix_copy[i][j] *= ev_rate

        return pheromones_matrix_copy

    def mutate_pheromones_matrix_by_row(
        self,
        pheromones_matrix: np.ndarray,
        solution_arcs: List[np.ndarray],
        current_iteration: int,
        restart_iteration: int,
        sigma: float,
        p_m: float,
    ) -> np.ndarray:
        """
        Mutate the pheromone matrix of the ant colony optimization algorithm
        based on a solution.

        Args:
            pheromones_matrix (np.ndarray): A NumPy array representing the
            pheromone matrix.
            solution_arcs (list): A list of arcs that represent a solution.
            current_iteration (int): An integer representing the current
            iteration of the algorithm.
            restart_iteration (int): An integer representing the iteration at
            which the algorithm was last restarted.
            sigma (float): A float representing the sigma value.
            p_m (float): A float representing the probability of mutating a
            pheromones row.

        Returns:
            A NumPy array representing the updated pheromone matrix.
        """

        pheromones_matrix_copy = pheromones_matrix.copy()

        mutation_intensity = self.get_mutation_intensity(
            current_iteration, restart_iteration, sigma
        )
        t_threshold = self.get_t_threshold(
            pheromones_matrix_copy, solution_arcs
        )
        mutation_value = (mutation_intensity * t_threshold) * 0.00001

        for i in range(pheromones_matrix_copy.shape[0]):
            if np.random.rand() < p_m:
                mutation_value *= np.random.choice([-1, 1])

                pheromones_matrix_copy[i] += mutation_value

        return pheromones_matrix_copy

    def mutate_pheromones_matrix_by_arcs(
        self,
        pheromones_matrix: np.ndarray,
        solution_arcs: List[np.ndarray],
        current_iteration: int,
        restart_iteration: int,
        sigma: float,
        p_m: float,
    ) -> np.ndarray:
        """
        Mutate the pheromone matrix of the ant colony optimization algorithm
        based on a solution.

        Args:
            pheromones_matrix (np.ndarray): A NumPy array representing the
            pheromone matrix.
            solution_arcs (list): A list of arcs that represent a solution.
            current_iteration (int): An integer representing the current
            iteration of the algorithm.
            restart_iteration (int): An integer representing the iteration at
            which the algorithm was last restarted.
            sigma (float): A float representing the sigma value.
            p_m (float): A float representing the probability of mutating a
            pheromones arc.

        Returns:
            A NumPy array representing the updated pheromone matrix.
        """

        pheromones_matrix_copy = pheromones_matrix.copy()

        mutation_intensity = self.get_mutation_intensity(
            current_iteration, restart_iteration, sigma
        )
        t_threshold = self.get_t_threshold(
            pheromones_matrix_copy, solution_arcs
        )
        mutation_value = (mutation_intensity * t_threshold) * 0.00001

        # Use triu_indices to get upper triangle indices
        iu = np.triu_indices(pheromones_matrix_copy.shape[0], k=1)

        # Update elements in upper triangle with random mutations
        mask = np.random.rand(len(iu[0])) < p_m
        mut = np.random.choice([-1, 1], size=len(iu[0])) * mutation_value

        pheromones_matrix_copy[iu] += mask * mut

        return pheromones_matrix_copy

    def get_mutation_intensity(
        self, iteration: int, restart_iteration: int, sigma: float
    ) -> float:
        """
        Calculates the mutation intensity for a given iteration.

        Args:
            iteration: The current iteration number.
            restart_iteration: The iteration number when a restart is
            performed.
            max_iteration: The maximum iteration number.
            sigma: The sigma value.

        Returns:
            The mutation intensity for the given iteration as a float.
        """

        a = iteration - restart_iteration
        b = self.max_iterations - restart_iteration

        return (a / b) * sigma

    def get_t_threshold(
        self, pheromones_matrix: np.ndarray, solution_arcs: List[np.ndarray]
    ) -> float:
        """
        Calculates the average pheromone level for the edges in the global
        best solution.

        Args:
            pheromones_matrix: A NumPy array representing the pheromone matrix.
            solution_arcs: A list of NumPy arrays containing the
            arcs that make up the edges of the solution.

        Returns:
            The mean pheromone level for the edges in the global best
            solution.
        """

        plain_arcs = get_flattened_list(solution_arcs, tuple)
        pheromones = []

        for i, j in plain_arcs:
            pheromones.append(pheromones_matrix[i][j])

        return np.mean(pheromones)

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
            steps = [
                restarts[i + 1] - restarts[i] for i in range(len(restarts) - 1)
            ]
            return np.mean(steps)
        else:
            return 0

    def solve(self) -> ACOSolution:
        """
        Solve the problem using the Best-Worst Ant System algorithm.

        Args:
            None.

        Returns:
            ACOSolution: A dictionary with the best-global solution,
            best-iterations solutions and statistics data to the problem.
        """

        errors = self.model_problem.validate_instance(
            self.nodes, self.demands, self.max_capacity
        )
        if errors:
            raise Exception(errors)

        # Starting initial matrixes
        self.matrix_pheromones = self.create_pheromones_matrix(self.t_max)
        self.matrix_probabilities = self.create_probabilities_matrix(
            self.matrix_pheromones.copy(),
            self.matrix_heuristics.copy(),
            self.alpha,
            self.beta,
        )

        # Greedy ants to find the best initial solution
        greedy_ant = self.model_ant(
            self.nodes,
            self.demands,
            self.matrix_probabilities,
            self.matrix_costs,
            self.max_capacity,
            self.tare,
            self.model_problem,
            self.q0,
        )

        greedy_ant_best_solution: AntSolution = {
            "cost": np.inf,
            "routes_arcs": [],
            "routes_costs": [],
            "routes_loads": [],
            "routes": [],
        }

        for _ in range(self.ants_num):
            greedy_ant_solution = greedy_ant.generate_solution()
            if greedy_ant_solution["cost"] < greedy_ant_best_solution["cost"]:
                greedy_ant_best_solution = greedy_ant_solution

        self.t_zero = self.get_as_fitness(
            (len(self.nodes) - 1) * greedy_ant_best_solution["cost"]
        )

        # Create real pheromones matrix
        self.matrix_pheromones = self.create_pheromones_matrix(
            initial_pheromones=self.t_max,
            lst_clusters=self.lst_clusters,
        )
        self.matrix_probabilities = self.create_probabilities_matrix(
            self.matrix_pheromones.copy(),
            self.matrix_heuristics.copy(),
            self.alpha,
            self.beta,
        )

        # Create ants
        ant = self.model_ant(
            self.nodes,
            self.demands,
            self.matrix_probabilities.copy(),
            self.matrix_costs,
            self.max_capacity,
            self.tare,
            self.model_problem,
            self.q0,
        )

        # Set iteration local search method
        ls_it = None
        if self.model_ls_it:
            ls_it = self.model_ls_it(
                self.matrix_costs,
                self.demands,
                self.tare,
                self.max_capacity,
                self.k_optimal,
                self.max_iterations,
                self.model_problem,
            )

        self.print_intance_parameters()
        print("\n")

        # Solve parameters
        best_prev_quality = np.inf
        best_solutions = []
        candidate_nodes_weights = None
        global_best_solution: AntSolution = {
            "cost": np.inf,
            "routes_arcs": [],
            "routes_costs": [],
            "routes_loads": [],
            "routes": [],
        }
        iterations_best_solutions = []
        iterations_mean_costs = []
        iterations_median_costs = []
        iterations_restarts = []
        iterations_std_costs = []
        iterations_times = []
        outputs_to_print = []
        prev_median = np.inf
        start_time = time.time()
        pheromones_matrices = []

        # Loop over max_iterations
        with tqdm(total=self.max_iterations) as pbar:
            for it in range(self.max_iterations):
                pbar.set_description(
                    "Global Best -> {}".format(
                        "{:.5f}".format(global_best_solution["cost"])
                    )
                )
                pbar.update(1)

                pheromones_matrices.append(deepcopy(self.matrix_pheromones))
                iteration_solutions = []

                # Generate solutions for each ant and update pheromones matrix
                for ant_idx in range(self.ants_num):
                    if candidate_nodes_weights:
                        ant_solution = ant.generate_solution(
                            candidate_nodes_weights[ant_idx]
                        )
                    else:
                        ant_solution = ant.generate_solution()
                    iteration_solutions.append(ant_solution)

                    # Update pheromones matrix with local update
                    if (
                        self.pheromones_online_update
                        and len(ant_solution["routes"]) == self.k_optimal
                    ):
                        self.matrix_pheromones = self.add_pheromones_to_matrix(
                            self.matrix_pheromones,
                            ant_solution["routes_arcs"],
                            1 / self.t_zero,
                            self.epsilon,
                        )
                        self.matrix_pheromones = (
                            self.evaporate_pheromones_matrix(
                                self.matrix_pheromones, 1 - self.epsilon
                            )
                        )

                        self.matrix_pheromones = (
                            self.apply_bounds_to_pheromones_matrix(
                                self.matrix_pheromones, self.t_min, self.t_max
                            )
                        )

                        # Update probabilities matrix
                        self.matrix_probabilities = (
                            self.create_probabilities_matrix(
                                self.matrix_pheromones.copy(),
                                self.matrix_heuristics.copy(),
                                self.alpha,
                                self.beta,
                            )
                        )
                        ant.set_probabilities_matrix(
                            self.matrix_probabilities.copy()
                        )

                # Sort solutions by fitness and filter by k_optimal
                iterations_solutions_sorted: List[AntSolution] = sorted(
                    iteration_solutions, key=lambda d: d["cost"]
                )
                iterations_solutions_sorted_and_restricted = [
                    solution
                    for solution in iterations_solutions_sorted
                    if len(solution["routes"]) == self.k_optimal
                ]

                # Select best and worst solutions and compute relative costs
                iteration_best_solution: AntSolution = {
                    "cost": np.inf,
                    "routes_arcs": [],
                    "routes_costs": [],
                    "routes_loads": [],
                    "routes": [],
                }
                iteration_worst_solution = iterations_solutions_sorted[-1]
                if iterations_solutions_sorted_and_restricted:
                    iteration_best_solution = (
                        iterations_solutions_sorted_and_restricted[0]
                    )
                else:
                    iteration_best_solution = iterations_solutions_sorted[0]

                iterations_best_solutions.append(
                    iteration_best_solution.copy()
                )

                # Calculate relative costs
                costs_median = np.median(
                    [
                        solution["cost"]
                        for solution in iterations_solutions_sorted
                    ]
                )
                costs_mean = np.mean(
                    [
                        solution["cost"]
                        for solution in iterations_solutions_sorted
                    ]
                )
                costs_std = np.std(
                    [
                        solution["cost"]
                        for solution in iterations_solutions_sorted
                    ]
                )

                # Update iteration output
                iteration_output = [
                    "\n\t> Iteration results: BEST({}), WORST({})".format(
                        iteration_best_solution["cost"],
                        iteration_worst_solution["cost"],
                    ),
                    "\t                    MED({}), AVG({}), STD({})\n".format(
                        costs_median, costs_mean, costs_std
                    ),
                ]

                # LS on best iteration solution
                ls_it_solution: AntSolution = {
                    "cost": np.inf,
                    "routes_arcs": [],
                    "routes_costs": [],
                    "routes_loads": [],
                    "routes": [],
                }
                if ls_it:
                    ls_it_solution = ls_it.improve(
                        iteration_best_solution["routes"], it
                    )
                    iteration_output[0] += ", LS({})".format(
                        ls_it_solution["cost"]
                    )

                # Update global best solution if LS best solution is better
                # or iteration best solution is better
                if ls_it_solution["cost"] < global_best_solution["cost"]:
                    global_best_solution = ls_it_solution

                    self.t_zero = self.get_as_fitness(
                        (len(self.nodes) - 1) * global_best_solution["cost"]
                    )
                elif (
                    iteration_best_solution["cost"]
                    < global_best_solution["cost"]
                ):
                    global_best_solution = iteration_best_solution

                    self.t_zero = self.get_as_fitness(
                        (len(self.nodes) - 1) * global_best_solution["cost"]
                    )

                if self.type_pheromones_update:
                    # Evaporate pheromones
                    self.matrix_pheromones = self.evaporate_pheromones_matrix(
                        self.matrix_pheromones, self.evaporation_rate
                    )

                    # Update pheromone matrix
                    if self.type_pheromones_update == "all_ants":
                        for ant_solution in iteration_solutions:
                            self.matrix_pheromones = (
                                self.add_pheromones_to_matrix(
                                    self.matrix_pheromones,
                                    ant_solution["routes_arcs"],
                                    ant_solution["cost"],
                                    self.rho,
                                )
                            )
                    elif self.type_pheromones_update == "it_best":
                        self.matrix_pheromones = self.add_pheromones_to_matrix(
                            self.matrix_pheromones,
                            iteration_best_solution["routes_arcs"],
                            iteration_best_solution["cost"],
                        )
                    elif self.type_pheromones_update == "g_best":
                        self.matrix_pheromones = self.add_pheromones_to_matrix(
                            self.matrix_pheromones,
                            global_best_solution["routes_arcs"],
                            global_best_solution["cost"],
                        )
                    elif self.type_pheromones_update == "pseudo_g_best":
                        if (it + 1) % 5 == 0:
                            self.matrix_pheromones = (
                                self.add_pheromones_to_matrix(
                                    self.matrix_pheromones,
                                    global_best_solution["routes_arcs"],
                                    global_best_solution["cost"],
                                )
                            )
                        else:
                            self.matrix_pheromones = (
                                self.add_pheromones_to_matrix(
                                    self.matrix_pheromones,
                                    iteration_best_solution["routes_arcs"],
                                    iteration_best_solution["cost"],
                                )
                            )
                    else:
                        raise Exception("Invalid pheromones update type")

                    # Penalize pheromones matrix by worst solution
                    self.matrix_pheromones = self.penalize_pheromones_matrix(
                        self.matrix_pheromones,
                        global_best_solution["routes_arcs"],
                        iteration_worst_solution["routes_arcs"],
                    )

                # Restart pheromones matrix if stagnation is reached
                if self.percent_arcs_limit:
                    remaining_iterations = self.max_iterations - it

                    if (
                        remaining_iterations >= 50
                        and self.is_stagnation_reached_by_arcs(
                            iteration_best_solution["routes_arcs"],
                            iteration_worst_solution["routes_arcs"],
                            self.percent_arcs_limit,
                        )
                    ):
                        self.matrix_pheromones = self.create_pheromones_matrix(
                            initial_pheromones=self.t_max,
                            lst_clusters=self.lst_clusters,
                        )

                        iteration_output.append("\t* Stagnation detected!!!")
                        iterations_restarts.append(it)
                elif self.percent_quality_limit:
                    remaining_iterations = self.max_iterations - it

                    if (
                        remaining_iterations >= 50
                        and self.is_stagnation_reached_by_solutions(
                            iteration_best_solution["cost"],
                            best_prev_quality,
                            iteration_worst_solution["cost"],
                            costs_median,
                            prev_median,
                            self.percent_quality_limit,
                        )
                    ):
                        self.matrix_pheromones = self.create_pheromones_matrix(
                            initial_pheromones=self.t_max,
                            lst_clusters=self.lst_clusters,
                        )

                        iteration_output.append("\t* Stagnation detected!!!")
                        iterations_restarts.append(it)

                # Apply mutation to pheromones matrix
                if len(iterations_restarts):
                    if self.type_mutation == "arcs":
                        self.matrix_pheromones = (
                            self.mutate_pheromones_matrix_by_arcs(
                                self.matrix_pheromones,
                                global_best_solution["routes_arcs"],
                                it,
                                iterations_restarts[-1],
                                self.sigma,
                                self.p_m,
                            )
                        )
                    elif self.type_mutation == "rows":
                        self.matrix_pheromones = (
                            self.mutate_pheromones_matrix_by_row(
                                self.matrix_pheromones,
                                global_best_solution["routes_arcs"],
                                it,
                                iterations_restarts[-1],
                                self.sigma,
                                self.p_m,
                            )
                        )

                # Apply bounds to pheromones matrix
                self.matrix_pheromones = (
                    self.apply_bounds_to_pheromones_matrix(
                        self.matrix_pheromones, self.t_min, self.t_max
                    )
                )

                # Update probabilities matrix
                self.matrix_probabilities = self.create_probabilities_matrix(
                    self.matrix_pheromones.copy(),
                    self.matrix_heuristics.copy(),
                    self.alpha,
                    self.beta,
                )
                ant.set_probabilities_matrix(self.matrix_probabilities.copy())

                # Append iteration best solution to list of best solutions
                best_solutions.append(iteration_best_solution)
                iterations_mean_costs.append(costs_mean)
                iterations_median_costs.append(costs_median)
                iterations_std_costs.append(costs_std)
                iterations_times.append(time.time() - start_time)

                # Update best_prev_quality, best_median
                best_prev_quality = iteration_best_solution["cost"]
                prev_median = costs_median

                # Update candidate nodes weights
                if self.type_candidate_nodes is not None:
                    candidate_nodes_weights = self.get_candidate_nodes_weight(
                        best_solutions, self.type_candidate_nodes
                    )

                # Print results
                if self.ipynb:
                    continue
                else:
                    outputs_to_print.append(iteration_output)
                    outputs_to_print = self.print_results(outputs_to_print)

        # Ending the algorithm run
        final_time = time.time()
        time_elapsed = final_time - start_time

        # Sort best solutions by fitness and filter by k_optimal
        best_solutions_set = []
        best_solutions_fitness = set()
        for ant_solution in sorted(best_solutions, key=lambda d: d["cost"]):
            if ant_solution["cost"] not in best_solutions_fitness:
                best_solutions_set.append(ant_solution)
                best_solutions_fitness.add(ant_solution["cost"])

        print(f"\n-- Time elapsed: {time_elapsed} --")
        if len(iterations_restarts):
            print(
                "Iterations when do restart: {}".format(
                    [
                        restart_iteration + 1
                        for restart_iteration in iterations_restarts
                    ]
                )
            )

        print(
            "\nBEST SOLUTION FOUND: {}".format(
                (
                    global_best_solution["cost"],
                    global_best_solution["routes"],
                    len(global_best_solution["routes"]),
                    global_best_solution["routes_loads"],
                )
            )
        )
        print(
            "Best 5 solutions: {}".format(
                [
                    (
                        ant_solution["cost"],
                        len(ant_solution["routes"]),
                        ant_solution["routes_loads"],
                    )
                    for ant_solution in best_solutions_set
                ][:5]
            )
        )

        return {
            "best_solutions": best_solutions,
            "global_best_solution": global_best_solution,
            "iterations_best_solutions": iterations_best_solutions,
            "iterations_mean_costs": iterations_mean_costs,
            "iterations_median_costs": iterations_median_costs,
            "iterations_std_costs": iterations_std_costs,
            "iterations_times": iterations_times,
            "pheromones_matrices": pheromones_matrices,
            "total_time": time_elapsed,
        }
