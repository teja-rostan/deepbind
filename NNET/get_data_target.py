import pandas as pd
import numpy as np
import sys
import time
from scipy.stats import scoreatpercentile
from sklearn.preprocessing import OneHotEncoder


def one_hot_decoder_prediction(y):
    """ Computing argmax for every feature trinity."""

    new_y = np.zeros(y.shape)
    for i in range(int(y.shape[1] / 3)):
        col = y[:, i * 3:i * 3 + 3]
        col_max = np.argmax(col, axis=1)
        col = np.zeros(col.shape)
        col[range(len(col_max)), col_max] = 1
        new_y[:, i * 3:i * 3 + 3] = col
    return new_y


def one_hot_encoder_target(y):
    """ one hot encoding of all target variables"""

    new_y = np.zeros((y.shape[0], y.shape[1] * 3))
    for i in range(y.shape[1]):
        col = y[:, i]
        enc = OneHotEncoder(sparse=False)
        trinity = enc.fit_transform(col.reshape(-1, 1))
        new_y[:, i * 3:i * 3 + 3] = trinity
    return new_y


def classification(target):
    """ change to classification problem """

    new_target = np.zeros(target.shape)
    for i, expression in enumerate(target.T):
        down_10 = np.percentile(expression, 10)
        up_10 = np.percentile(expression, 90)
        new_expression = np.ones(expression.shape)
        new_expression -= (expression <= down_10)
        new_expression += (expression >= up_10)
        new_target[:, i] = new_expression
    return new_target


def get_original_data(scores_file, delimiter, target_size, is_classification):
    """Reads a complete file, splits to data and target, preprocessing... """

    """ get data """
    df = pd.read_csv(scores_file, sep=delimiter)
    input_matrix = df.select_dtypes(include=['float64']).as_matrix()

    """ split data and target """
    data = input_matrix[:, :-target_size]
    target = input_matrix[:, -target_size:]

    """ classification or regression"""
    if is_classification:
        target = classification(target)

        """ one hot encoding of classes """
        target = one_hot_encoder_target(target)

    """ feature values reduction """
    # for i in range(data.shape[1]):
    #     feature = data[:, i]
    #     s1 = scoreatpercentile(feature, 5)
    #     s2 = scoreatpercentile(feature, 95)
    #     cool = np.add((feature > s2).astype(int), (feature < s1).astype(int))
    #     feature = np.multiply(feature, cool)
    #     data[:, i] = feature
    return data, target


def main():
    start = time.time()
    arguments = sys.argv[1:]

    if len(arguments) < 3:
        print("Not enough arguments stated! Usage: \n"
              "python nnet_class.py <true_scores_path> <predicted_scores_path> <delimiter> <target_size>.")
        sys.exit(0)

    scores_file = arguments[0]
    delimiter = arguments[1]
    target_size = int(arguments[2])

    data, target = get_original_data(scores_file, delimiter, target_size)

    end = time.time() - start
    print("Program run for %.2f seconds." % end)

if __name__ == '__main__':
    main()
