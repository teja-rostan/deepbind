import pandas as pd
import numpy as np
import time
import sys

help = \
    """
    The program horizontally stacks matrices [sequences * n_components] of num_of_best TF.
    Usage:
        python concat_best_pca.py <input_dir> <sort_file> <n_components> <num_of_best> <delimiter>.
"""


def main():
    start = time.time()
    arguments = sys.argv[1:]

    if len(arguments) < 5:
        print("Not enough arguments stated! Usage: \n"
              "python concat_best_pca.py <input_dir> <sort_file> <n_components> <num_of_best> <delimiter>.")
        return

    # input_dir = "/Users/tejarostan/deepbindProject/results/pca/pca_13_2_2017"
    # sort_file = "/Users/tejarostan/deepbindProject/results/pca/pca_29_12_2016/ratio_10.tab"

    input_dir = arguments[0]
    sort_file = arguments[1]
    n_components = arguments[2]
    num_of_best_pca = arguments[3]
    delimiter = arguments[4]

    sort_data = pd.read_csv(sort_file, sep=" ").as_matrix()[:, 0]

    all_pcas = []
    for i, f in enumerate(sort_data):
        if i < num_of_best_pca:
            fname_in = input_dir + "/new_" + str(n_components) + "/" + str(f[:11]) + str(n_components) + 'X_new.csv'
            print(fname_in)
            all_pcas.append(pd.read_csv(fname_in, delimiter=delimiter).drop('ID', axis=1).as_matrix())
        else:
            break
    seq_ids = pd.read_csv(fname_in, delimiter=delimiter)['ID'].as_matrix()
    df = pd.DataFrame(np.hstack(all_pcas), index=seq_ids)
    df.index.name = 'ID'
    df.to_csv(input_dir + "/new_" + str(n_components) + "/best_concat" + str(num_of_best_pca) + ".csv", sep='\t')

    end = time.time() - start
    print("Program has successfully written scores at " + input_dir + ".")
    print("Program run for %.2f seconds." % end)


if __name__ == '__main__':
    main()
