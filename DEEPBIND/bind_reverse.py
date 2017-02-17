import pandas as pd
import numpy as np
import time
import sys


help = \
    """
    The program reverses data in next steps:
        for new_file i = 0 to n
            takes a row i in all old_files and writes into a new_file i

    Usage:
        python bind_reverse.py <input_dir> <output_dir> <sort_file>.
"""


def main():
    start = time.time()
    arguments = sys.argv[1:]
    max_size = 877
    at_once = 100

    if len(arguments) < 4:
        print("Not enough arguments stated! Usage: \n"
              "python bind_reverse.py <input_dir> <output_dir> <sort_file> <delimiter>.")
        return

    input_dir = arguments[0]
    output_dir = arguments[1]
    sort_file = arguments[2]
    delimiter = arguments[3]

    sort_data = pd.read_csv(sort_file, sep=" ").as_matrix()[:, 0]

    for i in range(0, len(sort_data), at_once):
        if len(sort_data) - i < at_once:
            at_once = len(sort_data) - i

        all_best_tfs_names = []
        all_best_tfs = np.zeros((at_once, max_size, max_size))

        for j, f in enumerate(sort_data):
            if j < 877:
                all_best_tfs_names.append(str(f[:10]))

                fname_in = input_dir + "/" + str(f[:10]) + '_10X_org.csv'
                print(fname_in)

                binding_score = pd.read_csv(fname_in, delimiter=delimiter).drop('ID', axis=1).as_matrix()
                for k in range(len(binding_score)):
                    if i + k < len(sort_data):
                        all_best_tfs[k, j] = binding_score[i + k, :max_size]
            else:
                break
        seq_ids = pd.read_csv(fname_in, delimiter=delimiter)['ID'].as_matrix()[i:i + at_once]

        for k in range(at_once):
            df = pd.DataFrame(all_best_tfs[k], index=all_best_tfs_names)
            df.index.name = 'ID'

            fname_out = output_dir + '/' + str(seq_ids[k]) + ".csv"
            print(fname_out)
            df.to_csv(fname_out, sep='\t')

    end = time.time() - start
    print("Program has successfully written scores at " + output_dir + ".")
    print("Program run for %.2f seconds." % end)


if __name__ == '__main__':
    main()
