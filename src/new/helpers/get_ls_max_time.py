def get_ls_max_time(num_nodes,
                    actual_iteration,
                    max_iterations,
                    factor=0.001,
                    alpha=1):
    intensity_percentage = 1 - (actual_iteration / max_iterations)
    intensity_factor = 0.0005 + intensity_percentage * (factor - 0.0005)

    max_time = num_nodes * intensity_factor * alpha

    return max_time
