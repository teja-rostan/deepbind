import numpy as np
from sklearn.cross_validation import KFold
from scipy.stats import spearmanr
from NNET import get_data_target, CnnetClassLearner
import pandas as pd
import os


def learn_and_score(scores_file, delimiter, target_size):
    """
    Covnolutional Neural network learning and correlation scoring. Learning and predicting one target
    per time on balanced or unbalanced data.
    :param scores_file: The file with data and targets for neural network learning.
    :param delimiter: The delimiter for the scores_file.
    :param target_size: Number of targets in scores_file (the number of columns from the end of scores_file that we want
    to extract and double).
    :return: rhos and p-values of relative Spearman correlation, predictions and ids of instances that were included
    in learning and testing.
    """

    """ Get data and target tables. """
    data, target, raw_target, target_class = get_data_target.get_original_data(scores_file, delimiter, target_size, "class")

    wild_type = 11  # eleventh target attribute is a wild-type

    """ Neural network architecture initialisation. """
    class_size = 3
    n_hidden_n = int(max(data.shape[1], target.shape[1]) * 5 / 3)

    net = CnnetClassLearner.CnnetClassLearner(class_size, n_hidden_n)

    rhos = []
    p_values = []
    all_probs = []
    all_ids = []
    ids = get_data_target.get_ids(scores_file, delimiter, 'ID')
    # max_len = get_max_len(target, target_class, target_size)  # balanced data
    max_len = target.shape[0] / class_size  # unbalanced data

    nn_scores = []
    for t in range(target_size):
        target_c = target_class[:, t]

        """ Ignore missing attributes """
        if len(np.unique(target_c)) == 1:
            rhos.append(0)
            p_values.append(0)
            all_probs.append(np.zeros((max_len * class_size, 2)))
            all_ids.append(np.zeros((max_len * class_size, 1)).astype(str))
            print(max_len * class_size, 2)
            continue

        # targets, ids_b, data_b = get_balanced_data(t, class_size, target_c, data_c, target, max_len, ids)  # balanced data
        targets, data, ids_b = target[:, class_size * t:class_size * t + class_size], data, ids  # unbalanced data

        probs = np.zeros((targets.shape[0], 5))
        ids_end = np.zeros((targets.shape[0], 1)).astype(str)

        """ Split to train and test 10-fold Cross-Validation """
        skf = KFold(targets.shape[0], n_folds=10, shuffle=True)
        idx = 0
        for train_index, test_index in skf:
            trX, teX = data[train_index], data[test_index]
            trY, teY = targets[train_index], targets[test_index]

            teY2 = np.argmax(teY, axis=1)
            not_ones = teY2[teY2 != 1]
            not_ones_arg = [teY2 != 1]

            if len(not_ones) == 0:
                continue
            """ Learning and predicting """
            net.fit(trX, trY)
            prY = net.predict(teX)
            prY2 = np.argmax(prY, axis=1)

            maj = np.ones(np.shape(not_ones)) * (np.bincount(teY2 + 1).argmax() - 1)
            print(np.mean(not_ones == maj), np.mean(not_ones == prY2[not_ones_arg]), '|', majority(trY, teY)[0], np.mean(np.argmax(teY, axis=1) == prY2))
            print(not_ones, maj.astype(int), prY2[not_ones_arg])

            nn_scores.append(np.mean(not_ones == prY2[not_ones_arg]))
            print(majority(trY, teY)[0], np.mean(np.argmax(teY, axis=1) == prY))

            """ Storing results... """
            probs[idx:idx+len(teY), 0:-2] = prY
            probs[idx:idx+len(teY), -2] = np.argmax(teY, axis=1).flatten()
            probs[idx:idx+len(teY), -1] = prY2.flatten()
            ids_end[idx:idx+len(teY), 0] = ids_b[test_index].flatten()
            idx += len(teY)

        all_probs.append(np.around(probs, decimals=2))
        all_ids.append(ids_end)
        rho, p = spearmanr(probs[:, -1], probs[:, -2])
        rhos.append(rho)
        p_values.append(p)
    print(np.mean(nn_scores))
    return rhos, p_values, np.hstack(all_probs), np.hstack(all_ids)


def get_max_len(target, target_class, target_size):
    """
    Get maximum possible number of instances of one class to get balanced data through all target features.
    :param target: all feature targets of raw expressions.
    :param target_size: number of target features.
    :return: max class size.
    """

    max_len = len(target)
    for t in range(target_size):
        targets = target_class[:, t]
        if len(np.unique(targets)) == 1:
            continue
        down_10 = np.count_nonzero(np.array(targets == 0).astype(int))
        up_10 = np.count_nonzero(np.array(targets == 2).astype(int))
        if max_len > min(down_10, up_10):
            max_len = min(down_10, up_10)
    return max_len


def get_balanced_data(t, class_size, targets, data, target, max_len, ids):
    """
    Balancing data (Make a set of instances of every class to equal size).
    :param targets: target feature that we want to balance.
    :param data: the input attributes that we sample based on target balancing.
    :param max_len: maximum number of instances per class.
    :param ids: attribute names that we sample based on target balancing.
    :return: returns balanced data, target and ids.
    """

    nc = np.array(targets == 1)
    len_nc = np.count_nonzero(nc)

    id_nc = ids[nc]
    data_nc = data[nc]
    target_nc = target[nc, class_size * t:class_size * t + class_size]

    shuffle = np.random.permutation(len_nc)
    target_nc = target_nc[shuffle][:max_len]
    id_nc = id_nc[shuffle][:max_len]
    data_nc = data_nc[shuffle][:max_len]

    ids_prob = np.vstack([ids[targets == 0][:max_len].reshape(-1, 1),
                          ids[targets == 2][:max_len].reshape(-1, 1),
                          id_nc.reshape(-1, 1)])
    target_nc = np.vstack([target[targets == 0, class_size * t:class_size * t + class_size][:max_len],
                           target[targets == 2, class_size * t:class_size * t + class_size][:max_len],
                           target_nc])
    data_nc = np.vstack([data[targets == 0][:max_len],
                         data[targets == 2][:max_len],
                         data_nc])

    shuffle = np.random.permutation(len(target_nc))
    targets = target_nc[shuffle]
    ids_prob = ids_prob[shuffle]
    datas = data_nc[shuffle]
    return targets, ids_prob, datas


def majority(tr_y, te_y):
    """
    Classification accuracy of majority classifier.
    :param tr_y: train target Y.
    :param te_y: test target Y.
    :return: classification accuracy.
    """

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
