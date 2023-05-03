from typing import Tuple, List
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np

from src.new.helpers import get_route_arcs
from src.new.ants import AntSolution


class DisplayModel():
    @staticmethod
    def render_problem(nodes,
                       demands,
                       matrix_coords,
                       name='',
                       output_file=None,
                       figsize=(14, 6)):
        plt.figure(figsize=figsize)

        depot_x = matrix_coords[0][0]
        depot_y = matrix_coords[0][1]
        clients = nodes[1:]

        plt.subplot(1, 2, 1)
        plt.title(f'Demands ({name})')

        plt.bar(nodes, demands, color='g')

        plt.subplot(1, 2, 2)
        plt.title(f'Coordinates ({name})')

        for node in clients:
            x = matrix_coords[node][0]
            y = matrix_coords[node][1]
            desc_x = x + (x * 0.01)
            desc_y = y - (y * 0.03)

            plt.plot(x, y, c='g', marker='o')
            plt.annotate('$q_{%d}=%d$' %
                         (node, demands[node]), (desc_x, desc_y))

        plt.plot(depot_x, depot_y, c='r', marker='s')
        plt.annotate('DEPOT', (depot_x + (depot_x * 0.01),
                     depot_y - (depot_y * 0.03)))
        plt.grid()

        if output_file:
            plt.savefig(output_file)

        plt.show()

    @staticmethod
    def render_solution(solution: AntSolution,
                        matrix_coords: np.ndarray,
                        name: str,
                        solutions: List[AntSolution] = None,
                        avg_costs: List[float] = None,
                        median_costs: List[float] = None,
                        output_file: str = None,
                        base_figsize: Tuple[int] = (14, 6)):
        figsize = list(base_figsize)
        if solutions is not None or avg_costs is not None or \
                median_costs is not None:
            figsize = [14, 6]

        plt.figure(figsize=figsize)

        plt.subplot(1, 2, 1)
        plt.title(f'Solution ({name})')

        k = len(solution['routes'])
        color_palette = cm.jet(np.linspace(0, 1, k + 1))

        depot_x = matrix_coords[0][0]
        depot_y = matrix_coords[0][1]
        plt.plot(depot_x, depot_y, c='r', marker='s')

        for idx, route in enumerate(solution['routes']):
            for node in route:
                if node == 0:
                    continue

                x = matrix_coords[node][0]
                y = matrix_coords[node][1]
                desc_x = x + (x * 0.01)
                desc_y = y - (y * 0.03)

                plt.plot(x, y, c='c', marker='o', alpha=0.3)
                plt.annotate(f'n{node}',
                             (desc_x, desc_y),
                             alpha=0.3,
                             fontsize=8)

            route_arcs = get_route_arcs(route)
            for i, j in route_arcs:
                i_x = matrix_coords[i][0]
                i_y = matrix_coords[i][1]
                j_x = matrix_coords[j][0]
                j_y = matrix_coords[j][1]

                if i == 0 or j == 0:
                    plt.plot(
                        (i_x, j_x),
                        (i_y, j_y),
                        c=color_palette[idx],
                        alpha=0.5,
                        linestyle='--')
                else:
                    plt.plot(
                        (i_x, j_x), (i_y, j_y), c=color_palette[idx])
        plt.grid()

        if median_costs is not None:
            plt.subplot(1, 2, 2)

            iterations = [it + 1 for it in range(len(solutions))]
            costs = [cost for cost in median_costs]

            plt.plot(iterations, costs, c='y',
                     linewidth=2.0, label='It. Median Cost')

        if avg_costs is not None:
            plt.subplot(1, 2, 2)

            iterations = [it + 1 for it in range(len(solutions))]
            costs = [cost for cost in avg_costs]

            plt.plot(iterations, costs, c='g',
                     linewidth=2.0, label='It. Avg Cost')

        if solutions is not None:
            plt.subplot(1, 2, 2)

            iterations = [it + 1 for it in range(len(solutions))]
            solutions_reverse = solutions
            costs = [sol['cost'] for sol in solutions_reverse]
            it_max_cost = np.argmax(costs)
            max_cost = costs[it_max_cost]
            it_min_cost = np.argmin(costs)
            min_cost = costs[it_min_cost]

            plt.plot(iterations, costs, c='b',
                     linewidth=2.0, label='It. Best Cost')
            plt.plot(it_max_cost + 1, max_cost, c='r',
                     marker='o', label='Max Cost')
            plt.plot(it_min_cost + 1, min_cost, c='g',
                     marker='o', label='Min Cost')

        if solutions is not None or avg_costs is not None or \
                median_costs is not None:
            plt.title(f'Costs ({name})')
            plt.legend(loc='upper right')

        if output_file:
            plt.savefig(output_file)

        plt.show()
