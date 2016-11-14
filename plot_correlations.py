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
            matrix = matrix[np.arange(0, len(matrix), step=2)][:, 1:]  # get only rhos (without p-values)
            print(np.mean(matrix))
            plots_time.append(matrix)

    plt.figure("Correlation by time interval - classification")
    plt.subplot(231)
    plt.plot(np.array(plots_time)[[4, 16]].T[0])
    plt.xticks(np.arange(13), ['0h', '2h', '4h', '6h', '8h', '10h', '12h', '14h', '16h', '18h', '20h', '22h', '24h'])
    plt.legend(np.array(plots_names)[[4, 16]], loc='lower left')

    plt.subplot(232)
    plt.plot(np.array(plots_time)[[3, 15]].T[0])
    plt.xticks(np.arange(13), ['0h', '2h', '4h', '6h', '8h', '10h', '12h', '14h', '16h', '18h', '20h', '22h', '24h'])
    plt.legend(np.array(plots_names)[[3, 15]], loc='lower left')

    plt.subplot(233)
    plt.plot(np.array(plots_time)[[5, 17]].T[0])
    plt.xticks(np.arange(13), ['0h', '2h', '4h', '6h', '8h', '10h', '12h', '14h', '16h', '18h', '20h', '22h', '24h'])
    plt.legend(np.array(plots_names)[[5, 17, 11, 23]], loc='lower left')

    plt.subplot(234)
    plt.plot(np.array(plots_time)[[1, 13]].T[0])
    plt.xticks(np.arange(13), ['0h', '2h', '4h', '6h', '8h', '10h', '12h', '14h', '16h', '18h', '20h', '22h', '24h'])
    plt.legend(np.array(plots_names)[[1, 13, 7, 19]], loc='lower left')

    plt.subplot(235)
    plt.plot(np.array(plots_time)[[0, 12]].T[0])
    plt.xticks(np.arange(13), ['0h', '2h', '4h', '6h', '8h', '10h', '12h', '14h', '16h', '18h', '20h', '22h', '24h'])
    plt.legend(np.array(plots_names)[[0, 12]], loc='lower left')

    plt.subplot(236)
    plt.plot(np.array(plots_time)[[2, 14]].T[0])
    plt.xticks(np.arange(13), ['0h', '2h', '4h', '6h', '8h', '10h', '12h', '14h', '16h', '18h', '20h', '22h', '24h'])
    plt.legend(np.array(plots_names)[[2, 14]], loc='lower left')

    plt.figure("Correlation by time interval - regression")

    plt.subplot(231)
    plt.plot(np.array(plots_time)[[10, 22]].T[0])
    plt.xticks(np.arange(13), ['0h', '2h', '4h', '6h', '8h', '10h', '12h', '14h', '16h', '18h', '20h', '22h', '24h'])
    plt.legend(np.array(plots_names)[[10, 22]], loc='lower left')

    plt.subplot(232)
    plt.plot(np.array(plots_time)[[9, 21]].T[0])
    plt.xticks(np.arange(13), ['0h', '2h', '4h', '6h', '8h', '10h', '12h', '14h', '16h', '18h', '20h', '22h', '24h'])
    plt.legend(np.array(plots_names)[[9, 21]], loc='lower left')

    plt.subplot(233)
    plt.plot(np.array(plots_time)[[11, 23]].T[0])
    plt.xticks(np.arange(13), ['0h', '2h', '4h', '6h', '8h', '10h', '12h', '14h', '16h', '18h', '20h', '22h', '24h'])
    plt.legend(np.array(plots_names)[[11, 23]], loc='lower left')

    plt.subplot(234)
    plt.plot(np.array(plots_time)[[7, 19]].T[0])
    plt.xticks(np.arange(13), ['0h', '2h', '4h', '6h', '8h', '10h', '12h', '14h', '16h', '18h', '20h', '22h', '24h'])
    plt.legend(np.array(plots_names)[[7, 19]], loc='lower left')

    plt.subplot(235)
    plt.plot(np.array(plots_time)[[6, 18]].T[0])
    plt.xticks(np.arange(13), ['0h', '2h', '4h', '6h', '8h', '10h', '12h', '14h', '16h', '18h', '20h', '22h', '24h'])
    plt.legend(np.array(plots_names)[[6, 18]], loc='lower left')

    plt.subplot(236)
    plt.plot(np.array(plots_time)[[8, 20]].T[0])
    plt.xticks(np.arange(13), ['0h', '2h', '4h', '6h', '8h', '10h', '12h', '14h', '16h', '18h', '20h', '22h', '24h'])
    plt.legend(np.array(plots_names)[[8, 20]], loc='lower left')
    plt.show()

    end = time.time() - start
    print("Program run for %.2f seconds." % end)

if __name__ == '__main__':
    main()