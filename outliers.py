import time
import sys
import numpy as np
import pandas as pd
import os

from NNET import get_data_target


help = \
    """
    Reads the expressions. Counts number of instances that are outside of 2 * sigma (outliers).
    The program ranks the expression (higher that is the number of instances that are outside of 2 * sigma,
    lower is the rank)


    Usage:
        python outliers.py <input_file> <output_file> <delimiter> <target_size>.

    Example:
        python outliers.py path/to/expressions_or_join_scores_output path/to/outliers_table $"\\t" 14.
"""


def main():
    start = time.time()
    arguments = sys.argv[1:]

    if len(arguments) < 5:
        print("Not enough arguments stated! Usage: \n"
              "python outliers.py <input_file> <output_file> <delimiter> <target_size>.")
        sys.exit(0)

    input_file = arguments[0]
    output_file = arguments[1]
    delimiter = arguments[2]
    target_size = int(arguments[3])
    date = arguments[4]

    data_dir, _ = os.path.split(input_file)
    data_dir_1, data_file = os.path.split(output_file)

    list_file = open(input_file, "r")
    rows = list_file.readlines()
    list_file.close()

    # id = get_data_target.get_ids(data_dir + "/" + rows[0][:-1], delimiter, 'ID')
    cols = get_data_target.get_cols(data_dir + "/" + rows[0][:-1], delimiter, 0, -target_size)[2:]

    scores_file = data_dir + "/" + rows[0][:-1]
    print(scores_file)
    data, target, _, _ = get_data_target.get_original_data(scores_file, delimiter, target_size, "reg")

    all_cands = np.zeros(data.shape[1]).astype(int)
    left_thresh = np.zeros(data.shape[1])
    right_thresh = np.zeros(data.shape[1])
    for i in range(data.shape[1]):
        feature = data[:, i]
        sd = np.std(feature)
        all_cands[i] = np.sum(feature >= np.mean(feature) + 2 * sd) + np.sum(feature <= np.mean(feature) - 2 * sd)
        left_thresh[i] = np.mean(feature) - 2 * sd
        right_thresh[i] = np.mean(feature) + 2 * sd
        print(np.mean(feature), sd, np.mean(feature) + 2 * sd, np.mean(feature) - 2 * sd, all_cands[i], len(feature))

    """ Ranking... """
    temp = all_cands.argsort()[::-1]
    ranks = np.empty(len(all_cands), int)
    ranks[temp] = np.arange(len(all_cands))
    print(ranks.shape, all_cands.shape, np.vstack([ranks, all_cands]).T.shape)

    df = pd.DataFrame(data=np.vstack([ranks, all_cands, left_thresh, right_thresh]).T, index=cols,
                      columns=["rank", "num of outliers", "left threshold", "right threshold"])
    new_scores_file = data_dir_1 + "/" + date + "_" + data_file
    print(new_scores_file)

    df.to_csv(new_scores_file, sep=",")
    end = time.time() - start
    print("Program run for %.2f seconds." % end)

if __name__ == '__main__':
    main()




