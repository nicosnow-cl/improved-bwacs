def save_results_on_file(folder, file_name, ant_type, cluster_type, solution_list=[], instance='default', file_type='csv', header=None):
    try:
        file = open(f'{folder}/{file_name}.{file_type}', 'a')

        if header:
            file.write(
                f'{instance} {ant_type.upper()} {cluster_type.upper()}\n\n')
            file.write(f'{header}\n')

        for i, solution in enumerate(solution_list):
            line_to_write = str(solution).replace(
                '(', '').replace(')', '').replace(' ', '')
            file.write(f'{line_to_write}\n')

        file.close()
    except Exception as e:
        print(e)

        if (file):
            file.close()
