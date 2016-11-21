import sys
import time
import numpy as np
from sklearn.cross_validation import KFold
from sklearn.metrics import accuracy_score
from NNET import get_data_target,  NnetClassLearner


def score_ca_and_prob(y_predicted, y_true):
    """ Multi-target scoring with classification accuracy. """
    true_prob = []
    pred_prob = []
    all_ca = []
    k = 3
    for i in range(int(y_true.shape[1] / k)):
        col_true = y_true[:, i * k:i * k + k]
        col_predicted = y_predicted[:, i * k:i * k + k]

        col_true = np.argmax(col_true, axis=1)
        pred_prob.append(col_predicted[range(len(col_true)), col_true])

        col_predicted = np.argmax(col_predicted, axis=1)

        u, cn = np.unique(col_true, return_counts=True)
        probs = cn/col_true.shape[0]

        true_prob.append(probs[col_true])

        ca = accuracy_score(col_true, col_predicted)
        all_ca.append(ca)
    return all_ca, np.array(true_prob).T, np.array(pred_prob).T


def majority(tr_y, te_y):
    """ Classification accuracy of majority classifier. """
    k = 3
    mc = []
    for i in range(int(tr_y.shape[1] / k)):
        col_train = tr_y[:, i * k:i * k + k]
        col_test = te_y[:, i * k:i * k + k]

        col_train = np.argmax(col_train, axis=1)
        col_test = np.argmax(col_test, axis=1)
        counts = np.bincount(col_train)
        predicted = np.argmax(counts)

        maj = np.sum(col_test == predicted)/len(col_test)
        mc.append(maj)
    return mc


def learn_and_score(scores_file, delimiter, target_size):
    """Learning and scoring input data with neural network (and majority classifier for reference). """

    """ Get data and target tables. """
    data, target = get_data_target.get_original_data(scores_file, delimiter, target_size, "class")

    """ Neural network architecture initialisation. """
    n_hidden_n = int(max(data.shape[1], target.shape[1]) * 2 / 3)
    n_hidden_l = 2
    class_size = target_size * 3
    net = NnetClassLearner.NnetClassLearner(data.shape[1], class_size, n_hidden_l, n_hidden_n)

    all_MC = []
    all_NN = []
    idx = 0
    id = get_data_target.get_ids(scores_file, delimiter, 'ID')
    probabilities = np.zeros((len(id), target_size*2))
    prob_ids = []

    """ Split to train and test 10-fold Cross-Validation """
    skf = KFold(target.shape[0], n_folds=10, shuffle=True)
    for train_index, test_index in skf:
        trX, teX = data[train_index], data[test_index]
        trY, teY = target[train_index], target[test_index]

        prob_ids.extend(id[test_index])
        # print(trX.shape, trY.shape, teX.shape, teY.shape)

        net.fit(trX, trY)
        prY = net.predict(teX)

        """ Scoring... """
        mc_score = majority(trY, teY)
        nn_score, true_p, pred_p = score_ca_and_prob(prY, teY)

        probabilities[idx:idx+len(teY), :target_size] = true_p
        probabilities[idx:idx+len(teY), target_size:] = pred_p

        idx += len(teY)

        all_MC.append(mc_score)
        all_NN.append(nn_score)
    print("MC:", np.mean(all_MC, axis=0))
    print("NN:", np.mean(all_NN, axis=0))

    return np.mean(all_MC), np.mean(all_NN, axis=0), probabilities, prob_ids


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

    learn_and_score(scores_file, delimiter, target_size)

    end = time.time() - start
    print("Program run for %.2f seconds." % end)

if __name__ == '__main__':
    main()
