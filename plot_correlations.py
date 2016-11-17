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
    plt.xticks(np.arange(13), ['0h', '2h', '4h', '6h', '8h', '10h', '12h', '14h', '16h', '18h', '20h', '22h', '24h'])
    plt.legend(np.array(plots_names)[[0, 1, 2]], loc='lower left')
    plt.axis([0, 12, 0, 1])

    plt.subplot(212)
    plt.plot(np.mean(np.array(plots_time)[[5, 6]].T, axis=1))
    plt.xticks(np.arange(13), ['0h', '2h', '4h', '6h', '8h', '10h', '12h', '14h', '16h', '18h', '20h', '22h', '24h'])
    plt.legend(np.array(plots_names)[[5, 6]], loc='lower left')
    plt.axis([0, 12, 0, 1])

    # plt.subplot(313)
    # plt.plot(np.mean(np.array(plots_time)[[2]].T, axis=1))
    # plt.xticks(np.arange(13), ['0h', '2h', '4h', '6h', '8h', '10h', '12h', '14h', '16h', '18h', '20h', '22h', '24h'])
    # plt.legend(np.array(plots_names)[[2]], loc='lower left')
    # plt.axis([0, 12, 0, 1])
    #
    # plt.subplot(234)
    # plt.plot(np.array(plots_time)[[1, 13]].T[0])
    # plt.xticks(np.arange(13), ['0h', '2h', '4h', '6h', '8h', '10h', '12h', '14h', '16h', '18h', '20h', '22h', '24h'])
    # plt.legend(np.array(plots_names)[[1, 13, 7, 19]], loc='lower left')
    # plt.axis([0.0, 1.0])

    # plt.subplot(235)
    # plt.plot(np.array(plots_time)[[0, 12]].T[0])
    # plt.xticks(np.arange(13), ['0h', '2h', '4h', '6h', '8h', '10h', '12h', '14h', '16h', '18h', '20h', '22h', '24h'])
    # plt.legend(np.array(plots_names)[[0, 12]], loc='lower left')
    # plt.axis([0.0, 1.0])
    #
    # plt.subplot(236)
    # plt.plot(np.array(plots_time)[[2, 14]].T[0])
    # plt.xticks(np.arange(13), ['0h', '2h', '4h', '6h', '8h', '10h', '12h', '14h', '16h', '18h', '20h', '22h', '24h'])
    # plt.legend(np.array(plots_names)[[2, 14]], loc='lower left')
    # plt.axis([0.0, 1.0])

    plt.figure("Correlation by time interval - regression")

    plt.subplot(211)
    plt.plot(np.mean(np.array(plots_time)[[3, 4]].T, axis=1))
    plt.xticks(np.arange(13), ['0h', '2h', '4h', '6h', '8h', '10h', '12h', '14h', '16h', '18h', '20h', '22h', '24h'])
    plt.legend(np.array(plots_names)[[3, 4]], loc='lower left')
    plt.axis([0, 12, 0, 1])

    plt.subplot(212)
    plt.plot(np.mean(np.array(plots_time)[[7, 8]].T, axis=1))
    plt.xticks(np.arange(13), ['0h', '2h', '4h', '6h', '8h', '10h', '12h', '14h', '16h', '18h', '20h', '22h', '24h'])
    plt.legend(np.array(plots_names)[[7, 8]], loc='lower left')
    plt.axis([0, 12, 0, 1])
    #
    # plt.subplot(233)
    # plt.plot(np.array(plots_time)[[11, 23]].T[0])
    # plt.xticks(np.arange(13), ['0h', '2h', '4h', '6h', '8h', '10h', '12h', '14h', '16h', '18h', '20h', '22h', '24h'])
    # plt.legend(np.array(plots_names)[[11, 23]], loc='lower left')
    # plt.axis([0.0, 1.0])

    # plt.subplot(234)
    # plt.plot(np.array(plots_time)[[7, 19]].T[0])
    # plt.xticks(np.arange(13), ['0h', '2h', '4h', '6h', '8h', '10h', '12h', '14h', '16h', '18h', '20h', '22h', '24h'])
    # plt.legend(np.array(plots_names)[[7, 19]], loc='lower left')
    # plt.axis([0.0, 1.0])

    # plt.subplot(235)
    # plt.plot(np.array(plots_time)[[6, 18]].T[0])
    # plt.xticks(np.arange(13), ['0h', '2h', '4h', '6h', '8h', '10h', '12h', '14h', '16h', '18h', '20h', '22h', '24h'])
    # plt.legend(np.array(plots_names)[[6, 18]], loc='lower left')
    # plt.axis([0.0, 1.0])
    #
    # plt.subplot(236)
    # plt.plot(np.array(plots_time)[[8, 20]].T[0])
    # plt.xticks(np.arange(13), ['0h', '2h', '4h', '6h', '8h', '10h', '12h', '14h', '16h', '18h', '20h', '22h', '24h'])
    # plt.legend(np.array(plots_names)[[8, 20]], loc='lower left')
    # plt.axis([0.0, 1.0])

    plt.show()
    end = time.time() - start
    print("Program run for %.2f seconds." % end)

if __name__ == '__main__':
    main()