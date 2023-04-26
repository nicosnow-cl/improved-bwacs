import matplotlib.pyplot as plt


class DisplayModel():
    @staticmethod
    def render_problem(nodes,
                       demands,
                       matrix_coords,
                       name='',
                       output_file=None):
        plt.rc('font', size=16)
        plt.rc('figure', titlesize=18)
        plt.figure(figsize=(12, 12))
        # plt.scatter(matrix_coords[:, 0], matrix_coords[:, 1], s=10)

        depot_x = matrix_coords[0][0]
        depot_y = matrix_coords[0][1]
        clients = nodes[1:]

        for node in clients:
            x = matrix_coords[node][0]
            y = matrix_coords[node][1]

            plt.plot(x * 0.2, y * 0.2, c='g', marker='o', markersize=17)
            plt.annotate('$q_{%d}=%d$' %
                         (node, demands[node]), (x * 0.2, (y - 3) * 0.2))

        plt.plot(depot_x * 0.2, depot_y * 0.2,
                 c='r', marker='s', markersize=17)
        plt.annotate('DEPOT', ((depot_x - 1.5) * 0.2, (depot_y - 3) * 0.2))

        plt.title(f'Problem {name}')

        if output_file:
            plt.savefig(output_file)

        plt.show()

    @staticmethod
    def render_solutions_descend(solutions, name, output_file=None):
        plt.rc('font', size=16)
        plt.rc('figure', titlesize=18)
        plt.figure(figsize=(12, 12))

        solutions_reverse = solutions
        iterations = [it + 1 for it in range(len(solutions))]
        costs = [sol[1] for sol in solutions_reverse]

        plt.plot(iterations, costs, linewidth=2.0)

        plt.title(f'Costs VS Iterations ({name})')

        if output_file:
            plt.savefig(output_file)

        plt.show()
