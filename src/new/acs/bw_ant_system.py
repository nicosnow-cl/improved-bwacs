from copy import deepcopy
from tqdm import tqdm
from typing import List, Tuple
import numpy as np
import time
from random import random

from ..ants import AntSolution
from ..helpers import get_flattened_list
from .aco_solution import ACOSolution
from .mm_ant_system import MMAS


class BWAS(MMAS):
    p_m: float
    sigma: int
    type_mutation: str
    type_pheromones_model: str

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.p_m = 0.2
        self.rho = 0.2
        self.sigma = 4
        self.type_mutation = None
        self.type_pheromones_model = "as"

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
        gb_solution_arcs_flatten: List[Tuple[int, int]],
        curr_worst_solution_arcs_flatten: List[Tuple[int, int]],
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

        for arc in curr_worst_solution_arcs_flatten:
            if arc not in gb_solution_arcs_flatten:
                pheromones_matrix_copy[arc] *= (
                    self.evaporation_rate
                    if evaporation_rate is None
                    else evaporation_rate
                )

        return pheromones_matrix_copy

    def mutate_pheromones_matrix_by_row(
        self,
        pheromones_matrix: np.ndarray,
        solution_arcs: List[np.ndarray],
        solution_cost: float,
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
            pheromones_matrix_copy, solution_arcs, solution_cost
        )
        mutation_value = (mutation_intensity * t_threshold) * 0.5

        random_values = np.random.rand(pheromones_matrix_copy.shape[0])
        mutation_mask = random_values < p_m

        mutation_values = (
            np.random.choice([-1, 1], size=mutation_mask.sum())
            * mutation_value
        )

        pheromones_matrix_copy[mutation_mask] += mutation_values[:, np.newaxis]

        return pheromones_matrix_copy

    def mutate_pheromones_matrix_by_arcs(
        self,
        pheromones_matrix: np.ndarray,
        solution_arcs_flatten: List[Tuple[int, int]],
        solution_cost: float,
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
            pheromones_matrix_copy, solution_arcs_flatten, solution_cost
        )
        mutation_value = (mutation_intensity * t_threshold) * 0.5

        mask = np.random.rand(*pheromones_matrix_copy.shape) <= p_m

        mutation_values = np.random.choice(
            [-1, 1], size=pheromones_matrix_copy.shape
        )
        mutation_values *= mutation_value

        pheromones_matrix_copy += mask * mutation_values

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
        self,
        pheromones_matrix: np.ndarray,
        solution_arcs_flatten: List[Tuple[int, int]],
        solution_cost: float,
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

        pheromones = [pheromones_matrix[arc] for arc in solution_arcs_flatten]

        return np.sum(pheromones) / solution_cost

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
            nodes=self.nodes,
            lst_demands=self.demands,
            matrix_probabilities=self.matrix_probabilities.copy(),
            matrix_pheromones=self.matrix_pheromones.copy(),
            matrix_heuristics=self.matrix_heuristics.copy(),
            matrix_costs=self.matrix_costs.copy(),
            max_capacity=self.max_capacity,
            tare=self.tare,
            problem_model=self.model_problem,
            q0=self.q0,
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

        self.t_zero = self.get_as_fitness(greedy_ant_best_solution["cost"])

        if self.type_pheromones_model == "mmas":
            self.t_max, self.t_min = self.get_mmas_t_max_and_t_min(
                self.p_best, greedy_ant_best_solution["cost"]
            )

        if self.type_initial_pheromone == "tau_zero":
            self.initial_pheromones_value = self.t_zero
        else:
            self.initial_pheromones_value = self.t_max

        # Create real pheromones matrix
        self.matrix_pheromones = self.create_pheromones_matrix(
            initial_pheromones=self.initial_pheromones_value,
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
            nodes=self.nodes,
            lst_demands=self.demands,
            matrix_probabilities=self.matrix_probabilities.copy(),
            matrix_pheromones=self.matrix_pheromones.copy(),
            matrix_heuristics=self.matrix_heuristics.copy(),
            matrix_costs=self.matrix_costs.copy(),
            max_capacity=self.max_capacity,
            tare=self.tare,
            problem_model=self.model_problem,
            q0=self.q0,
        )

        # Set all ants solutions local search method
        ls_solutions = None
        if self.model_ls_solutions:
            ls_solutions = self.model_ls_solutions(
                self.matrix_costs.copy(),
                self.demands,
                self.tare,
                self.max_capacity,
                self.k_optimal,
                self.max_iterations,
                self.model_problem,
            )

        # Set iteration local search method
        ls_it = None
        if self.model_ls_it:
            ls_it = self.model_ls_it(
                self.matrix_costs.copy(),
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

                prev_gb_solution = deepcopy(global_best_solution)
                pheromones_matrices.append(deepcopy(self.matrix_pheromones))
                iteration_solutions = []

                # Ants loop
                start_time_ants = time.time()
                # Generate solutions for each ant and update pheromones matrix
                for ant_idx in range(self.ants_num):
                    if candidate_nodes_weights:
                        ant_solution = ant.generate_solution(
                            candidate_nodes_weights[0]
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
                # print("Ants time: ", time.time() - start_time_ants)

                # LS on all ant solutions
                if ls_solutions:
                    improved_solutions = []

                    for solution in iteration_solutions:
                        improved_solution = ls_solutions.improve(
                            solution["routes"], it
                        )
                        improved_solutions.append(improved_solution)

                    iteration_solutions = improved_solutions[:]

                # Sort solutions by fitness and filter by k_optimal
                start_time_sort = time.time()
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
                # print("Sort time: ", time.time() - start_time_sort)

                # Calculate relative costs
                start_time_relative_costs = time.time()
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
                # print(
                #     "Relative costs time: ",
                #     time.time() - start_time_relative_costs,
                # )

                # LS on best iteration solution
                start_time_ls_it = time.time()
                ls_it_solution: AntSolution = {
                    "cost": np.inf,
                    "routes_arcs": [],
                    "routes_arcs_flatten": [],
                    "routes_costs": [],
                    "routes_loads": [],
                    "routes": [],
                }
                if ls_it and len(iterations_restarts):
                    if random() > 0.3:
                        ls_it_solution = ls_it.improve(
                            deepcopy(iteration_best_solution["routes"]),
                            deepcopy(iteration_best_solution["routes_costs"]),
                            curr_iteration=it,
                            max_iterations=self.max_iterations,
                        )
                    else:
                        ls_it_solution = ls_it.improve(
                            deepcopy(global_best_solution["routes"]),
                            deepcopy(global_best_solution["routes_costs"]),
                            curr_iteration=it,
                            max_iterations=self.max_iterations,
                        )

                    iteration_output[0] += ", LS({})".format(
                        ls_it_solution["cost"]
                    )
                # print("LS it time: ", time.time() - start_time_ls_it)

                # Update global best solution if LS best solution is better
                # or iteration best solution is better
                start_time_global_best = time.time()
                if ls_it_solution["cost"] < global_best_solution["cost"]:
                    global_best_solution = ls_it_solution

                    self.t_zero = self.get_as_fitness(
                        global_best_solution["cost"]
                    )

                    if self.type_pheromones_model == "mmas":
                        self.t_max, self.t_min = self.get_mmas_t_max_and_t_min(
                            self.p_best, global_best_solution["cost"]
                        )

                        if self.type_initial_pheromone == "tau_zero":
                            self.initial_pheromones_value = self.t_zero
                        else:
                            self.initial_pheromones_value = self.t_max
                elif (
                    iteration_best_solution["cost"]
                    < global_best_solution["cost"]
                ):
                    global_best_solution = iteration_best_solution

                    self.t_zero = self.get_as_fitness(
                        global_best_solution["cost"]
                    )

                    if self.type_pheromones_model == "mmas":
                        self.t_max, self.t_min = self.get_mmas_t_max_and_t_min(
                            self.p_best, global_best_solution["cost"]
                        )

                        if self.type_initial_pheromone == "tau_zero":
                            self.initial_pheromones_value = self.t_zero
                        else:
                            self.initial_pheromones_value = self.t_max
                # print(
                #     "Global best time: ", time.time() - start_time_global_best
                # )

                if self.type_pheromones_update:
                    # Evaporate pheromones
                    start_time_evaporation = time.time()
                    self.matrix_pheromones = self.evaporate_pheromones_matrix(
                        self.matrix_pheromones, self.evaporation_rate
                    )
                    # print(
                    #     "Evaporation time: ",
                    #     time.time() - start_time_evaporation,
                    # )

                    # Penalize pheromones matrix by worst solution
                    start_time_penalize = time.time()
                    self.matrix_pheromones = self.penalize_pheromones_matrix(
                        self.matrix_pheromones,
                        global_best_solution["routes_arcs_flatten"],
                        iteration_worst_solution["routes_arcs_flatten"],
                    )
                    # print("Penalize time: ", time.time() - start_time_penalize)

                    # Update pheromone matrix
                    start_time_update = time.time()
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
                        # if (it + 1) % 5 == 0:
                        #     self.matrix_pheromones = (
                        #         self.add_pheromones_to_matrix(
                        #             self.matrix_pheromones,
                        #             global_best_solution["routes_arcs"],
                        #             global_best_solution["cost"],
                        #         )
                        #     )
                        # else:
                        #     self.matrix_pheromones = (
                        #         self.add_pheromones_to_matrix(
                        #             self.matrix_pheromones,
                        #             iteration_best_solution["routes_arcs"],
                        #             iteration_best_solution["cost"],
                        #         )
                        #     )

                        if (it + 1) % 3 == 0:
                            self.matrix_pheromones = (
                                self.add_pheromones_to_matrix(
                                    self.matrix_pheromones,
                                    global_best_solution[
                                        "routes_arcs_flatten"
                                    ],
                                    global_best_solution["cost"],
                                )
                            )

                        self.matrix_pheromones = self.add_pheromones_to_matrix(
                            self.matrix_pheromones,
                            iteration_best_solution["routes_arcs_flatten"],
                            iteration_best_solution["cost"],
                        )
                    else:
                        raise Exception("Invalid pheromones update type")
                    # print("Update time: ", time.time() - start_time_update)

                # Apply mutation to pheromones matrix
                remaining_iterations = self.max_iterations - it

                # Restart pheromones matrix if stagnation is reached
                if self.restart_after_iterations:
                    if (
                        (it + 1) % self.restart_after_iterations == 0
                        and remaining_iterations >= 50
                    ):
                        start_time_restart = time.time()
                        self.matrix_pheromones = self.create_pheromones_matrix(
                            initial_pheromones=self.initial_pheromones_value,
                            lst_clusters=self.lst_clusters,
                            curr_iteration=it,
                        )

                        iteration_output.append("\t* Stagnation detected!!!")
                        iterations_restarts.append(it)
                        # print(
                        #     "Restart time: ", time.time() - start_time_restart
                        # )
                    else:
                        start_time_mutation = time.time()
                        if self.type_mutation == "arcs":
                            self.matrix_pheromones = (
                                self.mutate_pheromones_matrix_by_arcs(
                                    self.matrix_pheromones,
                                    global_best_solution[
                                        "routes_arcs_flatten"
                                    ],
                                    global_best_solution["cost"],
                                    it + 1,
                                    it,
                                    self.sigma,
                                    self.p_m,
                                )
                            )
                        elif self.type_mutation == "rows":
                            self.matrix_pheromones = (
                                self.mutate_pheromones_matrix_by_row(
                                    self.matrix_pheromones,
                                    global_best_solution[
                                        "routes_arcs_flatten"
                                    ],
                                    global_best_solution["cost"],
                                    it + 1,
                                    it,
                                    self.sigma,
                                    self.p_m,
                                )
                            )
                        # print(
                        #     "Mutation time: ",
                        #     time.time() - start_time_mutation,
                        # )
                elif self.percent_arcs_limit:
                    if (
                        remaining_iterations >= 50
                        and self.is_stagnation_reached_by_arcs(
                            iteration_best_solution["routes_arcs_flatten"],
                            iteration_worst_solution["routes_arcs_flatten"],
                            self.percent_arcs_limit,
                        )
                    ):
                        self.matrix_pheromones = self.create_pheromones_matrix(
                            initial_pheromones=self.initial_pheromones_value,
                            lst_clusters=self.lst_clusters,
                            curr_iteration=it,
                        )

                        iteration_output.append("\t* Stagnation detected!!!")
                        iterations_restarts.append(it)
                elif self.percent_quality_limit:
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
                            initial_pheromones=self.initial_pheromones_value,
                            lst_clusters=self.lst_clusters,
                            curr_iteration=it,
                        )

                        iteration_output.append("\t* Stagnation detected!!!")
                        iterations_restarts.append(it)

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
                ant.set_pheromones_matrix(self.matrix_pheromones.copy())
                # ant.set_heuristics_matrix(self.matrix_heuristics.copy())

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
                if self.type_candidate_nodes is not None and (
                    (it + 1) % 5 == 0
                    or global_best_solution["cost"] != prev_gb_solution["cost"]
                ):
                    candidate_nodes_weights = self.get_candidate_nodes_weight(
                        [global_best_solution], self.type_candidate_nodes
                    )

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
