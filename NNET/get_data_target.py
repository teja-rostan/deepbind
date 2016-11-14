import pandas as pd
import numpy as np
import sys
import time
from sklearn.preprocessing import OneHotEncoder


def one_hot_decoder_prediction(y):
    """ Computing argmax for every feature trinity."""

    k = 3  # k neurons per feature
    new_y = np.zeros(y.shape)
    for i in range(int(y.shape[1] / k)):
        col = y[:, i * k:i * k + k]
        col_max = np.argmax(col, axis=1)
        col = np.zeros(col.shape)
        col[range(len(col_max)), col_max] = 1
        new_y[:, i * k:i * k + k] = col
    return new_y


def one_hot_encoder_target(y):
    """ one hot encoding of all target variables"""

    k = 3  # k neurons per feature
    new_y = np.zeros((y.shape[0], y.shape[1] * k))
    for i in range(y.shape[1]):
        col = y[:, i]
        enc = OneHotEncoder(sparse=False)
        one_hot = enc.fit_transform(col.reshape(-1, 1))
        new_y[:, i * k:i * k + k] = one_hot
    return new_y


def classification(target):
    """ change to classification problem """

    new_target = np.zeros(target.shape)
    for i, expression in enumerate(target.T):
        down_10 = np.percentile(expression, 10)  # 10th percentile, decides over analysis on Orange
        up_10 = np.percentile(expression, 90)  # 10th percentile, decides over analysis on Orange
        new_expression = np.ones(expression.shape)
        new_expression -= (expression <= down_10)
        new_expression += (expression >= up_10)
        new_target[:, i] = new_expression
    return new_target


def ordinal(target):
    """ change to ordinal problem """

    # low       00
    # no change 10
    # high      11
    k = 2  # k neurons per feature
    new_target = np.zeros((target.shape[0], target.shape[1] * k))
    for i in np.arange(0, new_target.shape[1], step=k):
        expression = target[:, int(i/2)]
        down_10 = np.percentile(expression, 10)
        up_10 = np.percentile(expression, 90)
        new_expression = np.ones((expression.shape[0], k))
        new_expression[:, 0] -= (expression <= down_10)
        new_expression[:, 1] -= (expression < up_10)
        new_target[:, i:i+k] = new_expression
    return new_target


def get_original_data(scores_file, delimiter, target_size, nn_type):
    """Reads a complete file, splits to data and target, preprocessing... """

    """ get data """
    df = pd.read_csv(scores_file, sep=delimiter)
    input_matrix = df.select_dtypes(include=['float64']).as_matrix()

    """ split data and target """
    data = input_matrix[:, :-target_size]
    target = input_matrix[:, -target_size:]
    raw_expressions = np.copy(target)
    target_class = classification(target)

    if nn_type == "class":

        target = one_hot_encoder_target(target_class)
    elif nn_type == "ord":
        target = ordinal(target)

    return data, target, raw_expressions, target_class


def get_ids(scores_file, delimiter, id_name):
    df = pd.read_csv(scores_file, sep=delimiter)
    return df[id_name].as_matrix()


def get_cols_target(scores_file, delimiter, target_size):
    df = pd.read_csv(scores_file, sep=delimiter)
    col_names = list(df)[-target_size:]
    col_names1 = [x for pair in zip(col_names, col_names) for x in pair]
    col_names1 = list(col_names1[i] + '_' + str(i % 2) for i in range(len(col_names1)))
    return col_names1 + col_names


def get_cols(scores_file, delimiter, _from, _to):
    df = pd.read_csv(scores_file, sep=delimiter)
    col_names = list(df)[_from:_to]
    return col_names


def main():
    start = time.time()
    arguments = sys.argv[1:]

    if len(arguments) < 4:
        print("Not enough arguments stated! Usage: \n"
              "python get_data_target.py <scores_file_path> <delimiter> <target_size> <nn_type>.")
        sys.exit(0)

    scores_file = arguments[0]
    delimiter = arguments[1]
    target_size = int(arguments[2])
    nn_type = arguments[3]

    get_original_data(scores_file, delimiter, target_size, nn_type)

    end = time.time() - start
    print("Program run for %.2f seconds." % end)

if __name__ == '__main__':
    main()
