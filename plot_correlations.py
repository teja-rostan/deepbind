import time
import sys
import numpy as np
import pandas as pd
import os
from matplotlib import pyplot as plt

help = \
    """
    Program for plotting correlation results at input_directory_path.

    Usage:
        python plot_correlations.py <input_directory_path> <delimiter>.
"""


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

    for row in rows:
        if row[:8] == "spearman":
            scores_file = input_dir + row
            # plots_names.append(row[13:-15])
            plots_names.append(row[13:-14])
            print(scores_file)
            df = pd.read_csv(scores_file, sep=delimiter)
            matrix = df.as_matrix()
            matrix = matrix[np.arange(0, len(matrix), step=2)][:, 1:]  # get only rhos without name rho
            matrix = np.nan_to_num(matrix)
            print(matrix.shape)
            if matrix.shape[0] > 3:
                matrix = matrix[8:11]
            print("mean matrix", np.mean(matrix))
            plots_time.append(matrix.T)

    print(np.array(plots_time))
    plots_time = np.array(plots_time).astype(float)
    print(plots_time[0].shape[1])
    time_ints = plots_time[0].shape[1]

    plt.figure("Correlation of expressions on time interval")
    # plt.subplot(211)

    def new_mean(x, axis=0):
        a = np.apply_along_axis(lambda v: np.mean(v[np.nonzero(v)]), axis, x)
        print(a.shape)
        print(a)
        return a

    def new_std(x, axis=0):
        a = np.apply_along_axis(lambda v: np.std(v[np.nonzero(v)]), axis, x)
        print(a.shape)
        print(a)
        return a

    plt.errorbar(np.arange(time_ints).reshape(-1, 1), new_mean(plots_time[[0]].T, axis=1), yerr=new_std(plots_time[[0]].T, axis=1), color='r')
    plt.errorbar(np.arange(time_ints).reshape(-1, 1), new_mean(plots_time[[2]].T, axis=1), yerr=new_std(plots_time[[2]].T, axis=1), color='g')

    eb1 = plt.errorbar(np.arange(time_ints).reshape(-1, 1), new_mean(plots_time[[1]].T, axis=1),
                       yerr=new_std(plots_time[[1]].T, axis=1), ls='--', color='r')
    eb1[-1][0].set_linestyle('--')
    eb1 = plt.errorbar(np.arange(time_ints).reshape(-1, 1), new_mean(plots_time[[3]].T, axis=1),
                       yerr=new_std(plots_time[[3]].T, axis=1), ls='--', color='g')
    eb1[-1][0].set_linestyle('--')

    plt.xticks(np.arange(time_ints),['16h', '18h', '20h'])
    #           ['0h', '2h', '4h', '6h', '8h', '10h', '12h', '14h', '16h', '18h', '20h', '22h', '24h'])
    # plt.legend(np.array(plots_names)[np.arange(16, 24)], loc='upper left')
    plots_names = np.array(plots_names)
    a = [0, 2, 1, 3]
    print(plots_names[a])
    plt.legend(np.array(plots_names[a]), loc='lower left')
    # plt.axis([0, 12, 0, 1])
    plt.grid()


    plt.show()
    end = time.time() - start
    print("Program run for %.2f seconds." % end)

if __name__ == '__main__':
    main()