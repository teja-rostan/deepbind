import sys
import time

import numpy as np
import os
import pandas as pd

from NNET import nnet_class
from NNET import nnet_reg

help = \
    """
    The program uses a Neural Network learning model to predict expressions (target variables) from binding scores of
    transcription factors (evaluated by deepbind)... MORE LATER...

    Usage:
        python nnet_score.py <input_path> <output_path> <learning_type> <delimiter> <target_size>,\n"
        "where learning_type=class or learning_type=reg".

    Example:
        python nnet_score.py input_files.txt results.csv reg $'\\t' 14
"""


def main():
    start = time.time()
    arguments = sys.argv[1:]

    if len(arguments) < 5:
        print("Error: Not enough arguments stated! Usage: \n"
              "python nnet_score.py <input_path> <output_path> <learning_type> <delimiter> <target_size>")
        sys.exit(0)

    scores_list = arguments[0]
    nn_scores_file = arguments[1]
    learning_type = arguments[2]
    delimiter = arguments[3]
    target_size = int(arguments[4])

    nn_scores = []
    col_names = []
    index_names = []

    list_file = open(scores_list, "r")
    data_dir, _ = os.path.split(scores_list)

    for row in list_file:
        scores_file = data_dir + "/" + row[:-1]
        df = pd.read_csv(scores_file, sep=delimiter)
        index_names = list(np.array(list(df))[-target_size:])
        col_names.append(row[:-4])
        print(scores_file)

        if learning_type == "class":
            all_mc, all_nn = nnet_class.learn_and_score(scores_file, delimiter, target_size)
            nn_scores.append(all_nn)
        elif learning_type == "reg":
            all_mc, all_nn = nnet_reg.learn_and_score(scores_file, delimiter, target_size)
            nn_scores.append(all_nn)
        else:
            print("Error: learning_type unknown! Usage: \n"
                  "python nnet_score.py <input_path> <output_path> <learning_type> <delimiter> <target_size>,\n"
                  "where learning_type=class or learning_type=reg")

    df = pd.DataFrame(data=np.array(nn_scores).T, columns=col_names, index=index_names)
    df.to_csv(nn_scores_file, sep=delimiter)

    end = time.time() - start
    print("Program run for %.2f seconds." % end)

if __name__ == '__main__':
    main()
