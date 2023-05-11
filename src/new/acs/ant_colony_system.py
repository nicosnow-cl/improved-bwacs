from random import random
from tqdm import tqdm
import numpy as np
import time

from ..ants import AntSolution
from .aco_solution import ACOSolution
from .ant_system import AS

MAX_FLOAT = 1.0
MIN_FLOAT = np.finfo(float).tiny


class ACS(AS):
    epsilon: float
    pheromones_online_update: bool
    q0: float

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.epsilon = self.rho / self.ants_num
        self.pheromones_online_update = False
        self.type_pheromones_update = "g_best"
        self.q0 = 0.2

        self.__dict__.update(kwargs)

    def print_intance_parameters(self):
        super().print_intance_parameters()

        print("----------------------------------------")
        print("ACS:")
        print("\tepsilon:", self.epsilon)
        print(
            "\tpheromones_local_update:",
            "yes" if self.pheromones_online_update else "no",
        )
        print("\tq0:", self.q0)

    def solve(self) -> ACOSolution:
        """
        Solve the problem using the Ant Colony Optimization algorithm.

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
        self.matrix_pheromones = self.create_pheromones_matrix(
            self.t_max, self.lst_clusters
        )
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
            self.t_max, self.lst_clusters
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
        iterations_std_costs = []
        iterations_times = []
        outputs_to_print = []
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
                        self.apply_bounds_to_pheromones_matrix(
                            self.t_min, self.t_max
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
                elif (
                    iteration_best_solution["cost"]
                    < global_best_solution["cost"]
                ):
                    global_best_solution = iteration_best_solution

                # Update pheromone matrix
                if self.type_pheromones_update == "all_ants":
                    for ant_solution in iterations_solutions:
                        self.matrix_pheromones = self.add_pheromones_to_matrix(
                            self.matrix_pheromones,
                            ant_solution["routes_arcs"],
                            ant_solution["cost"],
                            self.rho,
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
                    if random.random() < 0.75:
                        self.matrix_pheromones = self.add_pheromones_to_matrix(
                            self.matrix_pheromones,
                            iteration_best_solution["routes_arcs"],
                            iteration_best_solution["cost"],
                            max(
                                self.rho,
                                (
                                    (self.max_iterations - it)
                                    / self.max_iterations
                                ),
                            ),
                        )
                    else:
                        self.matrix_pheromones = self.add_pheromones_to_matrix(
                            self.matrix_pheromones,
                            global_best_solution["routes_arcs"],
                            global_best_solution["cost"],
                        )
                else:
                    raise Exception("Invalid pheromones update type")

                # Evaporate pheromones
                self.matrix_pheromones = self.evaporate_pheromones_matrix(
                    self.matrix_pheromones, self.evaporation_rate
                )

                # Apply bounds to pheromones matrix
                self.matrix_pheromones = (
                    self.apply_bounds_to_pheromones_matrix(
                        self.t_min, self.t_max
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
