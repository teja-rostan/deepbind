import sys
import time
import numpy as np
import os
import pandas as pd
from NNET import nnet_class, nnet_reg, nnet_ord, nnet_reg_one, nnet_class_one, nnet_class_one_time, nnet_reg_one_time

help = \
    """
    The program uses a Neural Network learning model to predict expressions (target variables) from binding scores of
    transcription factors (evaluated by deepbind) and calculates the sperman relative correlation between true and
    predicted data.

    Usage:
        python nnet_score.py <input_path> <output_path> <learning_type> <delimiter> <target_size>,\n"
        "where learning_type={class, class_one, clas_one_time, reg, reg_one, reg_one_time, ord}".

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
    corr_scores = []

    data_dir, _ = os.path.split(scores_list)
    data_dir_nn, data_file = os.path.split(nn_scores_file)

    list_file = open(scores_list, "r")
    rows = list_file.readlines()
    list_file.close()

    for row in rows:
        scores_file = data_dir + "/" + row[:-1]

        col_names.append(row[:-4])
        print(scores_file)

        if learning_type == "class":
            all_mc, all_nn, probs, ids = nnet_class.learn_and_score(scores_file, delimiter, target_size)
            nn_scores.append(all_nn)

        elif learning_type == "reg":
            all_mc, all_nn = nnet_reg.learn_and_score(scores_file, delimiter, target_size)
            nn_scores.append(all_nn)

        elif learning_type == "class_one":
            rhos, p_values, probs, ids = nnet_class_one.learn_and_score(scores_file, delimiter, target_size)
            corr_scores.append(rhos)
            corr_scores.append(p_values)
            nn_scores.append(np.hstack([probs, ids]))

        elif learning_type == "reg_one":
            rhos, p_values, probs, ids = nnet_reg_one.learn_and_score(scores_file, delimiter, target_size)
            corr_scores.append(rhos)
            corr_scores.append(p_values)
            nn_scores.append(np.hstack([probs, ids]))

        elif learning_type == "ord":
            probs, ids, spear = nnet_ord.learn_and_score(scores_file, delimiter, target_size)
            nn_probs_file = data_dir_nn + "/prob" + row[8:11] + "_" + data_file
            nn_spear_file = data_dir_nn + "/spearman" + row[8:11] + "_" + data_file

        elif learning_type == "class_one_time":
            rhos, p_values, probs, ids = nnet_class_one_time.learn_and_score(scores_file, data_dir, rows, delimiter, target_size)
            corr_scores.append(rhos)
            corr_scores.append(p_values)
            nn_scores.append(np.hstack([probs, ids]))

        elif learning_type == "reg_one_time":
            rhos, p_values, probs, ids = nnet_reg_one_time.learn_and_score(scores_file, data_dir, rows, delimiter, target_size)
            corr_scores.append(rhos)
            corr_scores.append(p_values)
            nn_scores.append(np.hstack([probs, ids]))

        else:
            print("Error: learning_type unknown! Usage: \n"
                  "python nnet_score.py <input_path> <output_path> <learning_type> <delimiter> <target_size>,\n"
                  "where learning_type is class/ord/reg/reg_one.")

    if learning_type == "reg_one" or learning_type == "class_one" or learning_type == "class_one_time" or learning_type == "reg_one_time":
        nn_one_file = data_dir_nn + "/spearman_one" + "_" + data_file
        # df = pd.DataFrame(data=np.array(corr_scores),
        #                   index=(['rho', 'p-value'] * len(rows)), columns=cols[-target_size:])
        df = pd.DataFrame(data=np.array(corr_scores), index=(['rho', 'p-value'] * len(rows)))
        df = df.round(decimals=4)
        df.to_csv(nn_one_file, sep=delimiter)

        # nn_scores.append(np.hstack([probs, ids]))
        # nn_probs_file = data_dir_nn + "/prob_one" + "_" + data_file
        # print(nn_probs_file)
        # df = pd.DataFrame(data=np.vstack(nn_scores), columns=cols)
        # df = df.round(decimals=2)
        # df.to_csv(nn_probs_file, sep=delimiter)

    end = time.time() - start
    print("Program run for %.2f seconds." % end)

if __name__ == '__main__':
    main()
