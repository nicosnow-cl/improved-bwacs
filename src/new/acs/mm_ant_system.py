from math import ceil, exp, log
from tqdm import tqdm
from typing import List, Tuple
import numpy as np
import time

from ..ants import AntSolution
from .ant_colony_system import ACS
from .aco_solution import ACOSolution
from ..helpers import get_flattened_list


class MMAS(ACS):
    delta: float
    initial_pheromones_value: float
    p_best: float
    percent_arcs_limit: float
    percent_iterations_restart: float
    percent_quality_limit: float
    restart_after_iterations: int
    type_initial_pheromone: str

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.delta = 0.5
        self.p_best = 0.05
        self.percent_arcs_limit = None
        self.percent_iterations_restart = None
        self.percent_quality_limit = None
        self.rho = 0.8
        self.type_initial_pheromone = "tau_max"
        self.type_pheromones_update = "pseudo_g_best"

        self.__dict__.update(kwargs)

        self.evaporation_rate = self.rho
        self.initial_pheromones_value = self.t_max
        self.restart_after_iterations = (
            ceil(self.max_iterations * self.percent_iterations_restart)
            if self.percent_iterations_restart
            else None
        )

    def print_intance_parameters(self):
        super().print_intance_parameters()

        print("----------------------------------------")
        print("MMAS:")
        print("\tdelta:", self.delta)
        print("\tinitial_pheromones_value:", self.initial_pheromones_value)
        print("\tp_best:", self.p_best)
        print("\tpercent_arcs_limit:", self.percent_arcs_limit)
        print("\tpercent_quality_limit:", self.percent_quality_limit)
        print("\ttype_initial_pheromone:", self.type_initial_pheromone)

    def get_mmas_t_max_and_t_min(
        self, p_best: float, best_solution_quality: float
    ) -> Tuple[float, float]:
        n = len(self.nodes)
        avg = n / 2
        p_best_n_root = exp(log(p_best) / n)

        t_max = (1 / (1 - self.evaporation_rate)) * self.get_as_fitness(
            best_solution_quality
        )

        upper = t_max * (1 - p_best_n_root)
        lower = (avg - 1) * p_best_n_root

        t_min = upper / lower

        return t_max, t_min

    def mmas_add_pheromones_to_matrix(
        self,
        pheromones_matrix: np.ndarray,
        solution_arcs: List[Tuple],
        solution_quality: float,
        factor: float = 1.0,
    ) -> np.ndarray:
        pheromones_matrix_copy = pheromones_matrix.copy()

        a = 1 / (1 - self.rho)
        solution_fitness = self.get_as_fitness(solution_quality)
        pheromone_amount = (a * solution_fitness) * factor

        for arcs in solution_arcs:
            for arc in arcs:
                i, j = arc
                pheromones_matrix_copy[i][j] += pheromone_amount

        return pheromones_matrix_copy

    def is_stagnation_reached_by_arcs(
        self,
        it_best_solution_arcs: List[tuple],
        it_worst_solution_arcs: List[tuple],
        similarity_percentage: float,
    ) -> bool:
        """
        Determine if the algorithm has reached stagnation based of arcs on the
        current and previous best and worst solutions.

        Args:
            it_best_solution_arcs (List[tuple]): The arcs of the current best
            solution.
            it_worst_solution_arcs (List[tuple]): The arcs of the current worst
            solution.
            similarity_percentage (float): The percentage of similarity between
            the current best and worst solutions to be considered as
            stagnation.

        Returns:
            bool: True if the percentage of similarity between the current
            best and worst solutions is greater than or equal to the
            predetermined threshold for stagnation, False otherwise.
        """

        it_best_solution_arcs_set = set(
            get_flattened_list(it_best_solution_arcs, tuple)
        )
        it_worst_solution_arcs_set = set(
            get_flattened_list(it_worst_solution_arcs, tuple)
        )

        different_tuples = (
            it_best_solution_arcs_set & it_worst_solution_arcs_set
        )

        a = len(different_tuples)
        b = len(it_best_solution_arcs_set.union(it_worst_solution_arcs_set))
        arcs_similarity = a / b

        return arcs_similarity >= similarity_percentage

    def is_stagnation_reached_by_solutions(
        self,
        best_actual_quality: float,
        best_prev_quality: float,
        worst_actual_quality: float,
        actual_median: float,
        prev_median: float,
        similarity_percentage: float,
    ) -> bool:
        """
        Determine if the algorithm has reached stagnation based of solutions
        quality and median.

        Args:
            best_actual_quality (float): The quality of the current best
            solution.
            best_prev_quality (float): The quality of the previous best
            solution.
            worst_actual_quality (float): The quality of the current worst
            solution.
            actual_median (float): The median of the current solutions.
            prev_median (float): The median of the previous solutions.
            similarity_percentage (float): The percentage of similarity between
            the current best and worst solutions to be considered as
            stagnation.

        Returns:
            bool: True if the percentage of similarity between the current
            best and worst solutions is greater than or equal to the
            predetermined threshold for stagnation, False otherwise.
        """

        no_improvements = best_actual_quality >= best_prev_quality

        quality_similarity = best_actual_quality / worst_actual_quality
        quality_similarity_reached = quality_similarity > similarity_percentage

        median_similarity = actual_median / prev_median
        median_similarity_reached = median_similarity > similarity_percentage

        return (
            no_improvements
            and quality_similarity_reached
            and median_similarity_reached
        )

    def apply_pheromones_trail_smoothing(
        self, pheromones_matrix: np.ndarray, t_max: float, delta: float
    ) -> np.ndarray:
        """
        Apply MMAX pheromones trail smoothing (PTS) to the pheromones matrix.

        Args:
            pheromones_matrix (np.ndarray): The pheromones matrix.
            t_max (float): The t_max parameter.
            delta (float): The delta parameter.

        Returns:
            np.ndarray: The pheromones matrix after the PTS.
        """

        pheromones_matrix_copy = pheromones_matrix.copy()
        shape = pheromones_matrix.shape

        for i in range(shape[0]):
            for j in range(shape[1]):
                if pheromones_matrix_copy[i][j] < t_max and i != j:
                    smooth_value = delta * (t_max - pheromones_matrix[i][j])
                    pheromones_matrix_copy[i][j] += smooth_value

        return pheromones_matrix_copy

    def solve(self) -> ACOSolution:
        """
        Solve the problem using the MAX-MIN Ant System algorithm.

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

        self.t_zero = self.get_as_fitness(greedy_ant_best_solution["cost"])

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
        global_best_solution = {
            "cost": np.inf,
            "routes_arcs": [],
            "routes_costs": [],
            "routes_loads": [],
            "routes": [],
        }
        iterations_mean_costs = []
        iterations_median_costs = []
        iterations_stagnations = []
        iterations_std_costs = []
        iterations_times = []
        outputs_to_print = []
        prev_median = np.inf
        start_time = time.time()

        # Loop over max_iterations
        with tqdm(total=self.max_iterations) as pbar:
            for it in range(self.max_iterations):
                pbar.set_description(
                    "Global Best -> {}".format(
                        "{:.5f}".format(global_best_solution["cost"])
                    )
                )
                pbar.update(1)

                iterations_solutions = []

                # Generate solutions for each ant and update pheromones matrix
                for ant_idx in range(self.ants_num):
                    if candidate_nodes_weights:
                        ant_solution = ant.generate_solution(
                            candidate_nodes_weights[ant_idx]
                        )
                    else:
                        ant_solution = ant.generate_solution()
                    iterations_solutions.append(ant_solution)

                    # Update pheromones matrix with local update
                    if self.pheromones_online_update:
                        # Update pheromones matrix
                        self.matrix_pheromones = self.add_pheromones_to_matrix(
                            self.matrix_pheromones,
                            ant_solution["routes_arcs"],
                            1 / self.t_zero,
                            self.epsilon,
                        )

                        # Evaporate pheromones matrix
                        self.matrix_pheromones = (
                            self.evaporate_pheromones_matrix(
                                self.matrix_pheromones, 1 - self.epsilon
                            )
                        )

                        # Apply bounds to pheromones matrix
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
                iterations_solutions_sorted = sorted(
                    iterations_solutions, key=lambda d: d["cost"]
                )
                iterations_solutions_sorted_and_restricted = [
                    solution
                    for solution in iterations_solutions_sorted
                    if len(solution["routes"]) == self.k_optimal
                ]

                # Select best and worst solutions and compute relative costs
                iteration_best_solution = {
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
                ls_it_solution = {
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
                        global_best_solution["cost"]
                    )

                    # Update t_min and t_max and
                    self.t_max, self.t_min = self.get_mmas_t_max_and_t_min(
                        self.p_best, global_best_solution["cost"]
                    )

                    self.initial_pheromones_value = (
                        self.t_zero
                        if self.type_initial_pheromone == "tau_zero"
                        else self.t_max
                    )
                elif (
                    iteration_best_solution["cost"]
                    < global_best_solution["cost"]
                ):
                    global_best_solution = iteration_best_solution

                    self.t_zero = self.get_as_fitness(
                        global_best_solution["cost"]
                    )

                    # Update t_min and t_max and
                    self.t_max, self.t_min = self.get_mmas_t_max_and_t_min(
                        self.p_best, global_best_solution["cost"]
                    )

                    self.initial_pheromones_value = (
                        self.t_zero
                        if self.type_initial_pheromone == "tau_zero"
                        else self.t_max
                    )

                # Evaporate pheromones
                self.matrix_pheromones = self.evaporate_pheromones_matrix(
                    self.matrix_pheromones, self.evaporation_rate
                )

                # Update pheromone matrix
                if self.type_pheromones_update == "all_ants":
                    for ant_solution in iterations_solutions:
                        self.matrix_pheromones = (
                            self.mmas_add_pheromones_to_matrix(
                                self.matrix_pheromones,
                                ant_solution["routes_arcs"],
                                ant_solution["cost"],
                                self.rho,
                            )
                        )
                elif self.type_pheromones_update == "it_best":
                    self.matrix_pheromones = (
                        self.mmas_add_pheromones_to_matrix(
                            self.matrix_pheromones,
                            iteration_best_solution["routes_arcs"],
                            iteration_best_solution["cost"],
                        )
                    )

                elif self.type_pheromones_update == "g_best":
                    self.matrix_pheromones = (
                        self.mmas_add_pheromones_to_matrix(
                            self.matrix_pheromones,
                            global_best_solution["routes_arcs"],
                            global_best_solution["cost"],
                        )
                    )
                elif self.type_pheromones_update == "pseudo_g_best":
                    if (it + 1) % 3 == 0:
                        self.matrix_pheromones = self.mmas_add_pheromones_to_matrix(
                            self.matrix_pheromones,
                            global_best_solution["routes_arcs"],
                            global_best_solution["cost"],
                            # self.evaporation_rate,
                        )
                    else:
                        self.matrix_pheromones = (
                            self.mmas_add_pheromones_to_matrix(
                                self.matrix_pheromones,
                                iteration_best_solution["routes_arcs"],
                                iteration_best_solution["cost"],
                            )
                        )

                # Apply PTS if stagnation is reached
                remaining_iterations = self.max_iterations - it

                if self.restart_after_iterations:
                    if (
                        (it + 1) % self.restart_after_iterations == 0
                        and remaining_iterations >= 50
                    ):
                        self.matrix_pheromones = (
                            self.apply_pheromones_trail_smoothing(
                                self.matrix_pheromones, self.t_max, self.delta
                            )
                        )

                        iteration_output.append("\t* Stagnation detected!!!")
                        iterations_stagnations.append(it)
                elif self.percent_arcs_limit:
                    remaining_iterations = self.max_iterations - it

                    if (
                        remaining_iterations >= 50
                        and self.is_stagnation_reached_by_arcs(
                            iteration_best_solution["routes_arcs"],
                            iteration_worst_solution["routes_arcs"],
                            self.percent_arcs_limit,
                        )
                    ):
                        self.matrix_pheromones = (
                            self.apply_pheromones_trail_smoothing(
                                self.matrix_pheromones, self.t_max, self.delta
                            )
                        )

                        iteration_output.append("\t* Stagnation detected!!!")
                        iterations_stagnations.append(it)
                elif (
                    self.percent_quality_limit
                    and self.is_stagnation_reached_by_solutions(
                        iteration_best_solution["cost"],
                        best_prev_quality,
                        iteration_worst_solution["cost"],
                        costs_median,
                        prev_median,
                        self.percent_quality_limit,
                    )
                ):
                    self.matrix_pheromones = (
                        self.apply_pheromones_trail_smoothing(
                            self.matrix_pheromones, self.t_max, self.delta
                        )
                    )

                    iteration_output.append("\t* Stagnation detected!!!")
                    iterations_stagnations.append(it)

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
                if self.type_candidate_nodes is not None and len(
                    best_solutions
                ):
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
        if len(iterations_stagnations):
            print(
                "Iterations when do PTS: {}".format(
                    [
                        stagnation_iteration + 1
                        for stagnation_iteration in iterations_stagnations
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
            "iterations_mean_costs": iterations_mean_costs,
            "iterations_median_costs": iterations_median_costs,
            "iterations_std_costs": iterations_std_costs,
            "iterations_times": iterations_times,
            "total_time": time_elapsed,
        }
