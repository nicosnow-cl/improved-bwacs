import pandas as pd
import csv


def csv_to_dataframe(csv_file_path):
    with open(csv_file_path, 'r') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        header = []
        rows = []

        for row in csv_reader:
            if line_count < 2:
                line_count += 1
                continue

            if line_count == 2:
                header = [row[0], row[2], row[3]]
                line_count += 1
                continue

            row_without_arcs = [row[0], row[-2], row[-1]]
            rows.append(row_without_arcs)
            line_count += 1

        df = pd.DataFrame(rows, columns=header)
        return df.apply(pd.to_numeric, errors='ignore')
