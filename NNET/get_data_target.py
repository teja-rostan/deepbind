import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder


def one_hot_decoder_prediction(y, k):
    """
    Computing argmax for every feature group (can handle multiple targets with k classes).
    :param y: target variables.
    :param k: number of classes per target.
    :return: argmax per target.
    """

    new_y = np.zeros(y.shape)
    for i in range(int(y.shape[1] / k)):
        col = y[:, i * k:i * k + k]
        col_max = np.argmax(col, axis=1)
        col = np.zeros(col.shape)
        col[range(len(col_max)), col_max] = 1
        new_y[:, i * k:i * k + k] = col
    return new_y


def one_hot_encoder_target(y, k):
    """
    One hot encoding of all target variables (can handle multiple targets with k classes).
    :param y: target variables.
    :param k: number of classes per target.
    :return: one hot per target.
    """

    new_y = np.zeros((y.shape[0], y.shape[1] * k))
    for i in range(y.shape[1]):
        col = y[:, i]
        enc = OneHotEncoder(sparse=False)
        one_hot = enc.fit_transform(col.reshape(-1, 1))
        new_y[:, i * k:i * k + k] = one_hot
    return new_y


def classification(target, down_per, up_per):
    """
    Changes to a classification problem up to three classes.
    :param target: Raw target values that we want to classify.
    :param down_per: lower threshold as percentile.
    :param up_per: upper threshold as percentile.
    :return: new target with classes.
    """

    new_target = np.zeros(target.shape)
    for i, expression in enumerate(target.T):
        down_10 = np.percentile(expression, down_per)  # 10th percentile, decides over analysis on Orange
        up_10 = np.percentile(expression, up_per)  # 10th percentile, decides over analysis on Orange
        new_expression = np.ones(expression.shape)
        new_expression -= (expression <= down_10)
        new_expression += (expression >= up_10)
        new_target[:, i] = new_expression
    return new_target


def ordinal(target, down_per, up_per):
    """
    Change to an ordinal problem, where low_expression=00, no change=10 and high_expression=11 (2 neurons per feature).
    :param target: Raw target values that we want to classifiy to three classes.
    :return: New target for an ordinal learning model.
    """

    k = 2  # k neurons per feature
    new_target = np.zeros((target.shape[0], target.shape[1] * k))
    for i in np.arange(0, new_target.shape[1], step=k):
        expression = target[:, int(i/2)]
        down_10 = np.percentile(expression, down_per)
        up_10 = np.percentile(expression, up_per)
        new_expression = np.ones((expression.shape[0], k))
        new_expression[:, 0] -= (expression <= down_10)
        new_expression[:, 1] -= (expression < up_10)
        new_target[:, i:i+k] = new_expression
    return new_target


def get_original_data(scores_file, delimiter, target_size, nn_type):
    """
    Reads a complete file with pandas, splits to data and target and makes wished preprocessing of the target variables.
    :param scores_file: The file with data and targets for neural network learning.
    :param delimiter: The delimiter for the scores_file.
    :param target_size: Number of columns at the end of scores_file that represent target variables.
    :param nn_type: The type of learning problem for neural networks {class, reg, ord}.
    :return: data that represent the input of neural network, target that represent the output of neural network,
    raw target expressions and classified target.
    """

    """ get data """
    df = pd.read_csv(scores_file, sep=delimiter)
    input_matrix = df.select_dtypes(include=['float64']).as_matrix()

    """ split data and target """
    data = input_matrix[:, :-target_size]
    target = input_matrix[:, -target_size:]
    raw_expressions = np.copy(target)
    target_class = classification(target, 10, 90)

    if nn_type == "class":
        target = one_hot_encoder_target(target_class, 3)

    elif nn_type == "ord":
        target = ordinal(target, 10, 90)

    return data, target, raw_expressions, target_class


def get_ids(scores_file, delimiter, id_name):
    """
    Gets the column of scores_file with ID names id_name.
    :param scores_file: The file from which we want to extract a column.
    :param delimiter: The delimiter for scores_file.
    :param id_name: the name of the column we want to extract.
    :return: The column in scores_file with the name id_name.
    """

    df = pd.read_csv(scores_file, sep=delimiter)
    return df[id_name].as_matrix()


def get_cols_target(scores_file, delimiter, target_size):
    """
    Gets column names (of targets) in scores_file and makes their twins.
    :param scores_file: The file from which we want column names and double them.
    :param delimiter: The delimiter for scores_file.
    :param target_size: Number of targets in scores_file (the number of columns from the end of scores_file that we want
    to extract and double).
    :return: the twins of column names.
    """

    df = pd.read_csv(scores_file, sep=delimiter)
    col_names = list(df)[-target_size:]
    col_names1 = [x for pair in zip(col_names, col_names) for x in pair]
    col_names1 = list(col_names1[i] + '_' + str(i % 2) for i in range(len(col_names1)))
    return col_names1 + col_names


def get_cols(scores_file, delimiter, _from, _to):
    """
    Gets column names from scores_file in range (_from, _to).
    :param scores_file: The file from where we want to extract column names.
    :param delimiter: The delimiter for scores_file.
    :param _from: The first index of a column.
    :param _to: The last index of a column (not included).
    :return: the column names from specific range.
    """
    df = pd.read_csv(scores_file, sep=delimiter)
    col_names = list(df)[_from:_to]
    return col_names
