from math import ceil
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
from typing import Any, List, Tuple
import numpy as np
import random
import time

# import itertools
from scipy.spatial import ConvexHull

from ..ants import AntSolution, FreeAnt
from ..helpers import get_inversed_matrix, same_line_print, clear_lines

# get_flattened_list
from ..models import ProblemModel
from .aco_solution import ACOSolution

MAX_FLOAT = 1.0
MIN_FLOAT = np.finfo(float).tiny


class AS:
    alpha: float
    ants_num: int
    beta: float
    demands: List[float]
    evaporation_rate: float
    ipynb: bool
    k_optimal: int
    lst_clusters: List[List[List[int]]]
    matrix_coords = np.ndarray
    matrix_costs: np.ndarray
    matrix_heuristics: np.ndarray
    matrix_pheromones: np.ndarray
    matrix_probabilities: np.ndarray
    inv_matrix_costs: np.ndarray
    max_capacity: float
    max_iterations: int
    model_ant: Any
    model_ls_it: Any
    model_ls_solutions: Any
    model_problem: ProblemModel
    nodes: List[int]
    rho: float
    t_max: float
    t_min: float
    t_zero: float
    tare: float
    type_candidate_nodes: str
    type_pheromones_update: str
    type_probabilities_matrix: str

    def __init__(self, **kwargs):
        self.alpha = 1
        self.beta = 1
        self.ipynb = False
        self.lst_clusters = None
        self.matrix_coords = None
        self.max_iterations = 300
        self.model_ls_it = None
        self.model_ls_solutions = None
        self.rho = 0.2
        self.evaporation_rate = 1 - self.rho
        self.t_max = MAX_FLOAT
        self.t_min = MIN_FLOAT
        self.t_zero = (self.t_max + self.t_min) / 2
        self.tare = 0
        self.type_candidate_nodes = None
        self.type_pheromones_update = "all_ants"
        self.type_probabilities_matrix = "classic"

        self.__dict__.update(kwargs)

    def print_intance_parameters(self):
        print("\nPARAMETERS")
        print("----------------------------------------")
        print("AS:")
        print("\talpha:", self.alpha)
        print("\tants_num:", self.ants_num)
        print("\tbeta:", self.beta)
        print("\tdemands:", len(self.demands))
        print("\tevaporation_rate:", self.evaporation_rate)
        print(
            "\titerations_local_search:", "yes" if self.model_ls_it else "no"
        )
        print("\tk_optimal:", self.k_optimal)
        print("\tlst_clusters:", "yes" if self.lst_clusters else "no")
        print("\tmax_capacity:", self.max_capacity)
        print("\tmax_iterations:", self.max_iterations)
        print(
            "\tmin_demand: {}, max_demand: {}, mean: {}".format(
                min(self.demands[1:]), max(self.demands), np.mean(self.demands)
            )
        )
        print(
            "\tmodel_ant:",
            "Free Ant"
            if type(self.model_ant) == type(FreeAnt)
            else "Restricted Ant",
        )
        print("\tnodes:", len(self.nodes[1:]))
        print("\trho:", self.rho)
        print("\tt_max: {:.50f}".format(self.t_max))
        print("\tt_min: {:.50f}".format(self.t_min))
        print("\tt_zero: {:.50f}".format(self.t_zero))
        print("\ttare:", self.tare)
        print("\ttype_candidate_nodes:", self.type_candidate_nodes)
        print("\ttype_pheromones_update:", self.type_pheromones_update)
        print("\ttype_probabilities_matrix:", self.type_probabilities_matrix)

    def create_pheromones_matrix(
        self,
        initial_pheromones: float = MAX_FLOAT,
        lst_clusters: List[List[List[int]]] = None,
        curr_iteration: int = 0,
    ) -> np.ndarray:
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
        matrix_pheromones = np.full((shape, shape), initial_pheromones)

        if lst_clusters is not None:
            total_arcs = []

            diff = self.t_max - self.t_zero
            actual_t_zero = self.t_zero + (
                diff * (curr_iteration / self.max_iterations)
            )

            for clusters in lst_clusters:
                for cluster in clusters:
                    nodes_points = [
                        [
                            self.matrix_coords[node][0],
                            self.matrix_coords[node][1],
                        ]
                        for node in cluster
                    ]
                    hull = ConvexHull(nodes_points)
                    vertexs = [cluster[vertex] for vertex in hull.vertices]

                    depot_to_nodes_arcs = list(
                        zip([0] * len(vertexs), vertexs)
                    )
                    vertexs_arcs = list(
                        zip(vertexs, vertexs[1:] + vertexs[:1])
                    )
                    vertex_arcs_inversed = [
                        (arc[1], arc[0]) for arc in vertexs_arcs[::-1]
                    ]

                    total_arcs += (
                        depot_to_nodes_arcs
                        + vertexs_arcs
                        + vertex_arcs_inversed
                    )

            for i in range(shape):
                for j in range(shape):
                    if (i, j) not in total_arcs:
                        # matrix_pheromones[i][j] = self.t_min  # bad
                        # matrix_pheromones[i][j] = self.t_zero  # bad
                        # matrix_pheromones[i][j] *= self.rho  # not too bad
                        # matrix_pheromones[i][j] *= 0.5
                        # matrix_pheromones[i][
                        #     j
                        # ] *= self.evaporation_rate  # good
                        matrix_pheromones[i][j] = actual_t_zero  # good
                        # matrix_pheromones[i][j] = initial_pheromones

            # clusters_arcs = [list(itertools.combinations(
            #     cluster, 2)) for cluster in clusters]
            # clusters_arcs_flattened = get_flattened_list(clusters_arcs)

            # for i in range(shape):
            #     for j in range(shape):
            #         if (i, j) in clusters_arcs_flattened:
            #             matrix_pheromones[i][j] *= self.evaporation_rate

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

    def evaporate_pheromones_matrix(
        self, pheromones_matrix: np.ndarray, evaporation_rate: float
    ) -> np.ndarray:
        """
        Evaporates the pheromone trail levels in the pheromone matrix.

        Args:
            pheromones_matrix (np.ndarray): The matrix of pheromone trail
            levels.
            evaporation_rate (float, optional): The evaporation rate.

        Returns:
            The pheromone matrix after the evaporation.
        """

        pheromones_matrix_copy = pheromones_matrix.copy()
        pheromones_matrix_copy *= evaporation_rate

        return pheromones_matrix_copy

    def add_pheromones_to_matrix(
        self,
        pheromones_matrix: np.ndarray,
        solution_arcs_flatten: List[Tuple[int, int]],
        solution_quality: float,
        factor: float = 1.0,
    ) -> np.ndarray:
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

        pheromones_matrix_copy = pheromones_matrix.copy()
        pheromones_amount = self.get_as_fitness(solution_quality) * factor

        for arc in solution_arcs_flatten:
            pheromones_matrix_copy[arc] += pheromones_amount

        return pheromones_matrix_copy

    def apply_bounds_to_pheromones_matrix(
        self,
        pheromones_matrix: np.ndarray,
        t_min: float = MIN_FLOAT,
        t_max: float = MAX_FLOAT,
    ) -> np.ndarray:
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

        return np.clip(pheromones_matrix.copy(), t_min, t_max)

    def create_probabilities_matrix(
        self,
        pheromones_matrix: np.ndarray,
        heuristics_matrix: np.ndarray,
        inv_costs_matrix: np.ndarray,
        alpha: float = 1.0,
        beta: float = 1.0,
    ) -> np.ndarray:
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

        pheromones_matrix_copy = pheromones_matrix.copy()

        if self.type_probabilities_matrix == "classic":
            probabilities_matrix = (
                pheromones_matrix_copy**alpha * heuristics_matrix
            )
            # return probabilities_matrix
            return probabilities_matrix / probabilities_matrix.sum()
        else:
            min_not_zero_value = inv_costs_matrix[inv_costs_matrix != 0].min()
            max_value = inv_costs_matrix[inv_costs_matrix != np.inf].max()

            scaler = MinMaxScaler(
                feature_range=(min_not_zero_value, max_value)
            )
            norm_matrix_pheromones = scaler.fit_transform(
                pheromones_matrix_copy
            )
            probabilities_matrix = (
                norm_matrix_pheromones**alpha * heuristics_matrix
            )

            return probabilities_matrix / probabilities_matrix.sum()

    def get_candidate_nodes_weight(
        self, solutions: List[AntSolution], type: str = "best"
    ) -> List[float]:
        """
        Returns a list of candidate starting nodes for the ants, biased
        towards the best starting nodes from the given solutions.

        Args:
            solutions (List[AntSolution]): The list of solutions.
            type (str, optional): The type of candidate nodes to return.

        Returns:
            A list of candidate starting nodes for the ants.
        """

        all_clients = self.nodes[1:]
        half_clients_length = ceil(len(all_clients) / 2)

        if type == "random":
            ants_weights = []
            selected_nodes = random.choices(
                all_clients,
                k=half_clients_length,
            )

            for _ in range(self.ants_num):
                ants_weights.append(
                    [
                        random.uniform(0.95, 1.0)
                        if node in selected_nodes
                        else 0.0
                        for node in self.nodes
                    ]
                )

            return ants_weights
        else:
            inv_costs = self.inv_matrix_costs[0][all_clients]
            prob_matrix = inv_costs / inv_costs.sum()

            closest_nodes = set(
                random.choices(
                    all_clients,
                    weights=prob_matrix,
                    k=max(ceil(len(all_clients) * 0.3), 10),
                )
            )

            best_nodes = []
            for solution in solutions:
                for route in solution["routes"]:
                    best_nodes.append(route[1])

            best_nodes = set(best_nodes)

            best_clusters_nodes = set()
            if self.lst_clusters:
                for clusters in self.lst_clusters:
                    for cluster in clusters:
                        nodes_num = ceil(len(cluster) * 0.3)
                        cluster_nodes_sorted = sorted(
                            cluster, key=lambda x: self.matrix_costs[x - 1]
                        )
                        best_clusters_nodes.update(
                            cluster_nodes_sorted[:nodes_num]
                        )
            min_value = min(inv_costs)
            max_value = max(inv_costs)
            middle_value = (min_value + max_value) / 2

            def get_ranking(node):
                if node == self.nodes[0]:
                    return 0.0
                elif node in best_nodes:
                    return inv_costs[node - 1] + random.uniform(
                        middle_value, max_value
                    )
                elif node in closest_nodes:
                    return inv_costs[node - 1] + random.uniform(
                        min_value, max_value
                    )
                elif node in best_clusters_nodes:
                    return random.uniform(middle_value, max_value)
                else:
                    return random.uniform(min_value, middle_value)

            return [
                [get_ranking(node) for node in self.nodes]
                for _ in range(self.ants_num)
            ]

    def print_results(
        self, outputs_to_print: List[List[str]], max_saved_outputs: int = 5
    ) -> List[List[str]]:
        """
        Print the results of the algorithm (in a iteration).

        Args:
            outputs_to_print (List[str]): The outputs to print.
            max_saved_outputs (int, optional): The maximum of outputs to save.

        Returns:
            List[str]: The outputs after printed.
        """

        if len(outputs_to_print) >= max_saved_outputs:
            outputs_to_print.pop(0)

        for it_outputs in outputs_to_print:
            same_line_print(it_outputs, False)

        clear_lines(max_saved_outputs)

        return outputs_to_print

    def solve(self) -> ACOSolution:
        """
        Solve the problem using the Ant System algorithm.

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
        self.inv_matrix_costs = (
            get_inversed_matrix(self.matrix_costs)
            if self.matrix_costs is not None
            else None
        )
        self.matrix_pheromones = self.create_pheromones_matrix(
            initial_pheromones=self.t_max, lst_clusters=self.lst_clusters
        )
        self.matrix_probabilities = self.create_probabilities_matrix(
            self.matrix_pheromones,
            self.matrix_heuristics,
            self.inv_matrix_costs,
            self.alpha,
            self.beta,
        )

        # Create ants
        ant = self.model_ant(
            nodes=self.nodes,
            lst_demands=self.demands,
            matrix_probabilities=self.matrix_probabilities,
            matrix_pheromones=self.matrix_pheromones,
            matrix_heuristics=self.matrix_heuristics,
            matrix_costs=self.matrix_costs,
            max_capacity=self.max_capacity,
            tare=self.tare,
            problem_model=self.model_problem,
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
        best_solutions: List[AntSolution] = []
        candidate_nodes_weights = None
        global_best_solution: AntSolution = {
            "cost": np.inf,
            "routes_arcs": [],
            "routes_arcs_flatten": [],
            "routes_costs": [],
            "routes_loads": [],
            "routes": [],
        }
        iterations_best_solutions = []
        iterations_mean_costs = []
        iterations_median_costs = []
        iterations_std_costs = []
        iterations_times = []
        pheromones_matrices = []
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

                pheromones_matrices.append(self.matrix_pheromones)
                iterations_solutions = []

                # Generate solutions for each ant and update pheromones matrix
                for ant_idx in range(self.ants_num):
                    if candidate_nodes_weights:
                        ant_solution = ant.generate_solution(
                            candidate_nodes_weights[ant_idx]
                        )
                    else:
                        ant_solution = ant.generate_solution()
                    iterations_solutions.append(ant_solution.copy())

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
                    "routes_arcs_flatten": [],
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
                    "It. {}/{} (GB: {}):".format(
                        it + 1,
                        self.max_iterations,
                        "{:.5f}".format(global_best_solution["cost"]),
                    ),
                    "\t> Results: BEST({}), WORST({})".format(
                        iteration_best_solution["cost"],
                        iteration_worst_solution["cost"],
                    ),
                    "\t           MED({}), AVG({}), STD({})".format(
                        costs_median, costs_mean, costs_std
                    ),
                ]

                # LS on best iteration solution
                ls_it_solution = {
                    "cost": np.inf,
                    "routes_arcs": [],
                    "routes_arcs_flatten": [],
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
                    global_best_solution = ls_it_solution.copy()
                elif (
                    iteration_best_solution["cost"]
                    < global_best_solution["cost"]
                ):
                    global_best_solution = iteration_best_solution.copy()

                # Update pheromones matrix
                if self.type_pheromones_update == "all_ants":
                    for ant_solution in iterations_solutions:
                        self.matrix_pheromones = self.add_pheromones_to_matrix(
                            self.matrix_pheromones,
                            ant_solution["routes_arcs_flatten"],
                            ant_solution["cost"],
                        )
                elif self.type_pheromones_update == "it_best":
                    self.matrix_pheromones = self.add_pheromones_to_matrix(
                        self.matrix_pheromones,
                        iteration_best_solution["routes_arcs_flatten"],
                        iteration_best_solution["cost"],
                    )
                elif self.type_pheromones_update == "g_best":
                    self.matrix_pheromones = self.add_pheromones_to_matrix(
                        self.matrix_pheromones,
                        global_best_solution["routes_arcs_flatten"],
                        global_best_solution["cost"],
                    )
                elif self.type_pheromones_update == "pseudo_g_best":
                    if random.random() < 0.75:
                        self.matrix_pheromones = self.add_pheromones_to_matrix(
                            self.matrix_pheromones,
                            iteration_best_solution["routes_arcs_flatten"],
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
                            global_best_solution["routes_arcs_flatten"],
                            global_best_solution["cost"],
                        )
                else:
                    raise Exception("Invalid pheromones update type")

                # Evaporate pheromones matrix
                self.matrix_pheromones = self.evaporate_pheromones_matrix(
                    self.matrix_pheromones, self.evaporation_rate
                )

                # Apply bounds to pheromones matrix
                self.matrix_pheromones = (
                    self.apply_bounds_to_pheromones_matrix(
                        self.matrix_pheromones, self.t_min, self.t_max
                    )
                )

                # Update probabilities matrix
                self.matrix_probabilities = self.create_probabilities_matrix(
                    self.matrix_pheromones,
                    self.matrix_heuristics,
                    self.inv_matrix_costs,
                    self.alpha,
                    self.beta,
                )
                ant.set_probabilities_matrix(self.matrix_probabilities)

                # Append iteration best solution to list of best solutions
                best_solutions.append(iteration_best_solution)
                iterations_mean_costs.append(costs_mean)
                iterations_median_costs.append(costs_median)
                iterations_std_costs.append(costs_std)
                iterations_times.append(time.time() - start_time)

                # Update candidate nodes weights
                if self.type_candidate_nodes is not None:
                    candidate_nodes_weights = self.get_candidate_nodes_weight(
                        [global_best_solution], self.type_candidate_nodes
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
            "iterations_best_solutions": iterations_best_solutions,
            "iterations_mean_costs": iterations_mean_costs,
            "iterations_median_costs": iterations_median_costs,
            "iterations_std_costs": iterations_std_costs,
            "iterations_times": iterations_times,
            "pheromones_matrices": pheromones_matrices,
            "total_time": time_elapsed,
        }
