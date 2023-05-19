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

        pheromones_matrix_copy = np.multiply(
            pheromones_matrix_copy, evaporation_rate
        )

        return pheromones_matrix_copy

    def add_pheromones_to_matrix(
        self,
        pheromones_matrix: np.ndarray,
        solution_arcs: List[Tuple],
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

        for arcs in solution_arcs:
            for arc in arcs:
                pheromones_matrix_copy[arc[0]][arc[1]] += pheromones_amount

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

        pheromones_matrix_copy = pheromones_matrix.copy()

        return np.clip(pheromones_matrix_copy, t_min, t_max)

    def create_probabilities_matrix(
        self,
        pheromones_matrix: np.ndarray,
        heuristics_matrix: np.ndarray,
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
        heuristics_matrix_copy = heuristics_matrix.copy()

        if self.type_probabilities_matrix == "classic":
            matrix = pheromones_matrix_copy**alpha * heuristics_matrix_copy
            return matrix
            # return matrix / matrix.sum()
        else:
            inv_distances_matrix = get_inversed_matrix(self.matrix_costs)
            min_not_zero_value = inv_distances_matrix[
                inv_distances_matrix != 0
            ].min()
            max_value = inv_distances_matrix[
                inv_distances_matrix != np.inf
            ].max()

            # Here we normalice the values between min distance
            # and max distance.
            scaler = MinMaxScaler(
                feature_range=(min_not_zero_value, max_value)
            )
            norm_matrix_pheromones = scaler.fit_transform(
                pheromones_matrix_copy
            )
            matrix = norm_matrix_pheromones**alpha * heuristics_matrix_copy

            return matrix / matrix.sum()

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

        all_clients = self.nodes[1:][:]

        if type == "random":
            ants_weights = []
            selected_nodes = np.random.choice(
                all_clients,
                size=ceil(len(all_clients) / 2),
                replace=False,
            )

            for _ in range(self.ants_num):
                ants_weights.append(
                    [
                        np.random.uniform(0.95, 1.0)
                        if node in selected_nodes
                        else 0.0
                        for node in self.nodes
                    ]
                )

            return ants_weights
        else:
            costs = self.matrix_costs[0][all_clients]
            inv_costs = np.divide(
                1,
                costs,
                out=np.zeros_like(costs),
            )
            prob_matrix = (
                inv_costs[: int(len(all_clients) / 2)]
                / inv_costs[: int(len(all_clients) / 2)].sum()
            )

            closest_nodes = set(
                np.random.choice(
                    all_clients[: int(len(all_clients) / 2)],
                    size=max(ceil(len(all_clients) * 0.3), 10),
                    p=prob_matrix,
                    replace=False,
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
                            cluster, key=lambda x: costs[x - 1]
                        )
                        best_clusters_nodes.update(
                            cluster_nodes_sorted[:nodes_num]
                        )

            middle = (inv_costs.min() + inv_costs.max()) / 2
            diff = inv_costs.max() - inv_costs.min()

            def get_ranking(node):
                if node == self.nodes[0]:
                    return 0.0
                elif node in best_nodes:
                    return inv_costs[node - 1] + np.random.uniform(
                        middle, inv_costs.max()
                    )
                elif node in closest_nodes:
                    return inv_costs[node - 1] + np.random.uniform(
                        inv_costs.min(), middle
                    )
                # elif node in best_clusters_nodes:
                #     return np.random.uniform(inv_costs.min(), inv_costs.max())
                else:
                    return np.random.uniform(inv_costs.min(), inv_costs.max())
                    # return 0.0

            return [[get_ranking(node) for node in self.nodes]]

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
        self.matrix_pheromones = self.create_pheromones_matrix(
            initial_pheromones=self.t_max, lst_clusters=self.lst_clusters
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

                # Update pheromones matrix
                if self.type_pheromones_update == "all_ants":
                    for ant_solution in iterations_solutions:
                        self.matrix_pheromones = self.add_pheromones_to_matrix(
                            self.matrix_pheromones,
                            ant_solution["routes_arcs"],
                            ant_solution["cost"],
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

                # Evaporate pheromones matrix
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
