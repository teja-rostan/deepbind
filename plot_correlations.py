import time
import sys
import numpy as np
import pandas as pd
import os
from matplotlib import pyplot as plt


def main():
    start = time.time()
    arguments = sys.argv[1:]

    if len(arguments) < 2:
        print("Not enough arguments stated! Usage: \n"
              "python plot_correlations.py <input_directory_path> <delimiter>.")
        sys.exit(0)

    input_dir = arguments[0]
    delimiter = arguments[1]

    plots_names = []
    plots_time = []

    rows = os.listdir(input_dir)
    # cols = get_data_target.get_cols(input_dir + "/" + rows[0][:-1], delimiter,  1, 20)
    for row in rows:
        if row[:8] == "spearman":
            scores_file = input_dir + row
            plots_names.append(row[13:-15])
            print(scores_file)
            df = pd.read_csv(scores_file, sep=delimiter)
            matrix = df.as_matrix()
            matrix = matrix[np.arange(0, len(matrix), step=2)][:, 1:]  # get only rhos without name rho
            print(np.mean(matrix))
            plots_time.append(matrix)

    plt.figure("Correlation by time interval - classification")
    plt.subplot(211)
    plt.plot(np.mean(np.array(plots_time)[[0, 1, 2]].T, axis=1))
    plt.errorbar(np.std(np.array(plots_time).astype(float)[[0, 1, 2]].T, axis=1))
    plt.xticks(np.arange(13), ['0h', '2h', '4h', '6h', '8h', '10h', '12h', '14h', '16h', '18h', '20h', '22h', '24h'])
    plt.legend(np.array(plots_names)[[0, 1, 2]], loc='upper left')
    plt.axis([0, 12, 0, 1])
    plt.grid()

    plt.subplot(212)
    plt.plot(np.mean(np.array(plots_time)[[5, 6]].T, axis=1))
    plt.xticks(np.arange(13), ['0h', '2h', '4h', '6h', '8h', '10h', '12h', '14h', '16h', '18h', '20h', '22h', '24h'])
    plt.legend(np.array(plots_names)[[5, 6]], loc='upper left')
    plt.axis([0, 12, 0, 1])
    plt.grid()

    plt.figure("Correlation by time interval - regression")

    plt.subplot(211)
    plt.plot(np.mean(np.array(plots_time)[[3, 4]].T, axis=1))
    plt.xticks(np.arange(13), ['0h', '2h', '4h', '6h', '8h', '10h', '12h', '14h', '16h', '18h', '20h', '22h', '24h'])
    plt.legend(np.array(plots_names)[[3, 4]], loc='upper left')
    plt.axis([0, 12, 0, 1])
    plt.grid()

    plt.subplot(212)
    plt.plot(np.mean(np.array(plots_time)[[7, 8]].T, axis=1))
    plt.xticks(np.arange(13), ['0h', '2h', '4h', '6h', '8h', '10h', '12h', '14h', '16h', '18h', '20h', '22h', '24h'])
    plt.legend(np.array(plots_names)[[7, 8]], loc='upper left')
    plt.axis([0, 12, 0, 1])
    plt.grid()

    plt.show()
    end = time.time() - start
    print("Program run for %.2f seconds." % end)

if __name__ == '__main__':
    main()