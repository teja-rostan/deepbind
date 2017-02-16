import pandas as pd
import time
import sys

help = \
    """
    The program in sorted order opens files in input_dir and fills nan values with the median of row values. New files
     are then written in output_dir.
    Usage:
        python fill_missing_values.py <input_dir> <output_dir> <sort_file> <delimiter>.
"""


def main():
    start = time.time()
    arguments = sys.argv[1:]

    if len(arguments) < 4:
        print("Not enough arguments stated! Usage: \n"
              "python fill_missing_values.py <input_dir> <output_dir> <sort_file> <delimiter>.")
        return

    # input_dir = "/Users/tejarostan/deepbindProject/results/pca/pca_13_2_2017/org_10"
    # # input_dir = "/Users/tejarostan/deepbindProject/results/binding_scores/bind_20_12_2016"
    # output_dir = "/Users/tejarostan/deepbindProject/results/matlab/pca_14_2_2017"
    # # output_dir = "/Users/tejarostan/deepbindProject/results/matlab/org_14_2_2017"
    # sort_file = "/Users/tejarostan/deepbindProject/results/pca/pca_29_12_2016/ratio_10.tab"

    input_dir = arguments[0]
    output_dir = arguments[1]
    sort_file = arguments[2]
    delimiter = arguments[3]

    sort_data = pd.read_csv(sort_file, sep=" ").as_matrix()[:, 0]

    for i, tf in enumerate(sort_data):
        if i >= 157:
            print(tf[:10])
            tf_scores = pd.read_csv(input_dir + "/" + tf[:10] + "_10X_org.csv", delimiter=delimiter, na_values='?')
            # tf_scores = pd.read_csv(input_dir + "/" + tf[:10] + ".csv", delimiter=delimiter, na_values='?')
            tf_scores = tf_scores.T.fillna(tf_scores.median(axis=1, numeric_only=True)).T
            tf_scores = tf_scores.fillna(0)
            tf_scores.to_csv(output_dir + "/" + tf[:10] + ".csv", header=False, index=False)

    end = time.time() - start
    print("Program has successfully written scores at " + input_dir + ".")
    print("Program run for %.2f seconds." % end)


if __name__ == '__main__':
    main()
