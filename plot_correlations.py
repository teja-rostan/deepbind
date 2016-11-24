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

    plots_time = np.array(plots_time).astype(float)
    time_ints = 13

    plt.figure("Correlation by time interval - classification")
    plt.subplot(211)
    plt.errorbar(np.arange(time_ints).reshape(-1, 1), np.mean(plots_time[[0]].T, axis=1), yerr=np.std(plots_time[[0]].T, axis=1))
    plt.errorbar(np.arange(time_ints).reshape(-1, 1), np.mean(plots_time[[1]].T, axis=1), yerr=np.std(plots_time[[1]].T, axis=1))
    plt.errorbar(np.arange(time_ints).reshape(-1, 1), np.mean(plots_time[[2]].T, axis=1), yerr=np.std(plots_time[[2]].T, axis=1))
    plt.errorbar(np.arange(time_ints).reshape(-1, 1), np.mean(plots_time[[3]].T, axis=1), yerr=np.std(plots_time[[3]].T, axis=1))

    plt.errorbar(np.arange(time_ints).reshape(-1, 1), np.mean(plots_time[[4]].T, axis=1), yerr=np.std(plots_time[[4]].T, axis=1))
    plt.errorbar(np.arange(time_ints).reshape(-1, 1), np.mean(plots_time[[5]].T, axis=1), yerr=np.std(plots_time[[5]].T, axis=1))
    plt.errorbar(np.arange(time_ints).reshape(-1, 1), np.mean(plots_time[[6]].T, axis=1), yerr=np.std(plots_time[[6]].T, axis=1))
    plt.errorbar(np.arange(time_ints).reshape(-1, 1), np.mean(plots_time[[7]].T, axis=1), yerr=np.std(plots_time[[7]].T, axis=1))
    plt.xticks(np.arange(time_ints), ['0h', '2h', '4h', '6h', '8h', '10h', '12h', '14h', '16h', '18h', '20h', '22h', '24h'])
    plt.legend(np.array(plots_names)[np.arange(9)], loc='upper left')
    plt.axis([0, 12, 0, 1])
    plt.grid()

    plt.subplot(211)
    plt.errorbar(np.arange(time_ints).reshape(-1, 1), np.mean(plots_time[[8]].T, axis=1), yerr=np.std(plots_time[[8]].T, axis=1))
    plt.errorbar(np.arange(time_ints).reshape(-1, 1), np.mean(plots_time[[9]].T, axis=1), yerr=np.std(plots_time[[9]].T, axis=1))
    plt.errorbar(np.arange(time_ints).reshape(-1, 1), np.mean(plots_time[[10]].T, axis=1), yerr=np.std(plots_time[[10]].T, axis=1))
    plt.errorbar(np.arange(time_ints).reshape(-1, 1), np.mean(plots_time[[11]].T, axis=1), yerr=np.std(plots_time[[11]].T, axis=1))

    plt.errorbar(np.arange(time_ints).reshape(-1, 1), np.mean(plots_time[[12]].T, axis=1), yerr=np.std(plots_time[[12]].T, axis=1))
    plt.errorbar(np.arange(time_ints).reshape(-1, 1), np.mean(plots_time[[13]].T, axis=1), yerr=np.std(plots_time[[13]].T, axis=1))
    plt.errorbar(np.arange(time_ints).reshape(-1, 1), np.mean(plots_time[[14]].T, axis=1), yerr=np.std(plots_time[[14]].T, axis=1))
    plt.errorbar(np.arange(time_ints).reshape(-1, 1), np.mean(plots_time[[15]].T, axis=1), yerr=np.std(plots_time[[15]].T, axis=1))
    plt.xticks(np.arange(time_ints), ['0h', '2h', '4h', '6h', '8h', '10h', '12h', '14h', '16h', '18h', '20h', '22h', '24h'])
    plt.legend(np.array(plots_names)[np.arange(9, 16)], loc='upper left')
    plt.axis([0, 12, 0, 1])
    plt.grid()

    plt.figure("Correlation by time interval - regression")
    plt.subplot(211)
    plt.errorbar(np.arange(time_ints).reshape(-1, 1), np.mean(plots_time[[16]].T, axis=1), yerr=np.std(plots_time[[16]].T, axis=1))
    plt.errorbar(np.arange(time_ints).reshape(-1, 1), np.mean(plots_time[[17]].T, axis=1), yerr=np.std(plots_time[[17]].T, axis=1))
    plt.errorbar(np.arange(time_ints).reshape(-1, 1), np.mean(plots_time[[18]].T, axis=1), yerr=np.std(plots_time[[18]].T, axis=1))
    plt.errorbar(np.arange(time_ints).reshape(-1, 1), np.mean(plots_time[[19]].T, axis=1), yerr=np.std(plots_time[[19]].T, axis=1))

    plt.errorbar(np.arange(time_ints).reshape(-1, 1), np.mean(plots_time[[20]].T, axis=1), yerr=np.std(plots_time[[20]].T, axis=1))
    plt.errorbar(np.arange(time_ints).reshape(-1, 1), np.mean(plots_time[[21]].T, axis=1), yerr=np.std(plots_time[[21]].T, axis=1))
    plt.errorbar(np.arange(time_ints).reshape(-1, 1), np.mean(plots_time[[22]].T, axis=1), yerr=np.std(plots_time[[22]].T, axis=1))
    plt.errorbar(np.arange(time_ints).reshape(-1, 1), np.mean(plots_time[[23]].T, axis=1), yerr=np.std(plots_time[[23]].T, axis=1))
    plt.xticks(np.arange(time_ints), ['0h', '2h', '4h', '6h', '8h', '10h', '12h', '14h', '16h', '18h', '20h', '22h', '24h'])
    plt.legend(np.array(plots_names)[np.arange(16, 24)], loc='upper left')
    plt.axis([0, 12, 0, 1])
    plt.grid()

    plt.subplot(211)
    plt.errorbar(np.arange(time_ints).reshape(-1, 1), np.mean(plots_time[[24]].T, axis=1), yerr=np.std(plots_time[[24]].T, axis=1))
    plt.errorbar(np.arange(time_ints).reshape(-1, 1), np.mean(plots_time[[25]].T, axis=1), yerr=np.std(plots_time[[25]].T, axis=1))
    plt.errorbar(np.arange(time_ints).reshape(-1, 1), np.mean(plots_time[[26]].T, axis=1), yerr=np.std(plots_time[[26]].T, axis=1))
    plt.errorbar(np.arange(time_ints).reshape(-1, 1), np.mean(plots_time[[27]].T, axis=1), yerr=np.std(plots_time[[27]].T, axis=1))

    plt.errorbar(np.arange(time_ints).reshape(-1, 1), np.mean(plots_time[[28]].T, axis=1), yerr=np.std(plots_time[[28]].T, axis=1))
    plt.errorbar(np.arange(time_ints).reshape(-1, 1), np.mean(plots_time[[29]].T, axis=1), yerr=np.std(plots_time[[29]].T, axis=1))
    plt.errorbar(np.arange(time_ints).reshape(-1, 1), np.mean(plots_time[[30]].T, axis=1), yerr=np.std(plots_time[[30]].T, axis=1))
    plt.errorbar(np.arange(time_ints).reshape(-1, 1), np.mean(plots_time[[31]].T, axis=1), yerr=np.std(plots_time[[31]].T, axis=1))
    plt.xticks(np.arange(time_ints), ['0h', '2h', '4h', '6h', '8h', '10h', '12h', '14h', '16h', '18h', '20h', '22h', '24h'])
    plt.legend(np.array(plots_names)[np.arange(24, 32)], loc='upper left')
    plt.axis([0, 12, 0, 1])
    plt.grid()

    plt.show()
    end = time.time() - start
    print("Program run for %.2f seconds." % end)

if __name__ == '__main__':
    main()