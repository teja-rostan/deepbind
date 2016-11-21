import sys
import time
import numpy as np
from sklearn.cross_validation import KFold
from NNET import get_data_target, NnetRegLearner


def rmse_score(y_predicted, y_true):
    """ Computes root mean squared error on every target variable separately. """

    results = []
    for i in range(y_true.shape[1]):
        col_true = y_true[:, i]
        col_predicted = y_predicted[:, i]
        results.append(np.sqrt(np.sum(np.square(col_true - col_predicted)) / col_true.size))
    return results


def mean_score(tr_y, te_y):
    """ Computes mean value (Mean learner) on every target variable separately and scores with RMSE."""

    predicted = np.ones(te_y.shape)
    for i in range(tr_y.shape[1]):
        predicted[:, i] *= np.median(tr_y[:, i])
    return rmse_score(predicted, te_y)


def learn_and_score(scores_file, delimiter, target_size):
    """Learning and scoring input data with Neural Network (and Mean Learner for reference). """

    """ Get data and target tables. """
    data, target = get_data_target.get_original_data(scores_file, delimiter, target_size, "reg")

    """ Neural network architecture initialisation. """
    n_hidden_n = int(max(data.shape[1], target.shape[1]) * 2 / 3)
    n_hidden_l = 2
    class_size = target_size * 3
    net = NnetRegLearner.NnetRegLearner(data.shape[1], class_size, n_hidden_l, n_hidden_n)

    """ Split to train and test 10-fold Cross-Validation """
    all_MC = []
    all_NN = []

    skf = KFold(target.shape[0], n_folds=10, shuffle=True)
    for train_index, test_index in skf:
        trX, teX = data[train_index], data[test_index]
        trY, teY = target[train_index], target[test_index]

        net.fit(trX, trY)
        prY = net.predict(teX)

        """ Scoring... """
        md_score = mean_score(trY, teY)
        nn_score = rmse_score(prY, teY)

        all_MC.append(md_score)
        all_NN.append(nn_score)
    print("MD:", np.mean(all_MC, axis=0))
    print("NN:", np.mean(all_NN, axis=0))
    return np.mean(all_MC), np.mean(all_NN, axis=0)


def main():
    start = time.time()
    arguments = sys.argv[1:]

    if len(arguments) < 3:
        print("Not enough arguments stated! Usage: \n"
              "python nnet_reg.py <true_scores_path> <predicted_scores_path> <delimiter> <target_size>.")
        sys.exit(0)

    scores_file = arguments[0]
    delimiter = arguments[1]
    target_size = int(arguments[2])

    learn_and_score(scores_file, delimiter, target_size)

    end = time.time() - start
    print("Program run for %.2f seconds." % end)

if __name__ == '__main__':
    main()
