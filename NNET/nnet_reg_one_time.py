import numpy as np
from sklearn.cross_validation import KFold
from scipy.stats import spearmanr
from NNET import get_data_target, NnetRegLearner, nnet_reg_one


def learn_and_score(scores_file, data_dir, rows, delimiter, target_size):
    """
    Neural network learning and correlation scoring. Regressional Neural Network, learning and predicting one target
    per time on balanced or unbalanced data. It takes expressions from all time intervals at once as attributes to
    neural network.
    :param scores_file: The file with data and targets for neural network learning.
    :param delimiter: The delimiter for the scores_file.
    :param target_size: Number of targets in scores_file (the number of columns from the end of scores_file that we want
    to extract and double).
    :return: rhos and p-values of relative Spearman correlation, predictions and ids of instances that were included
    in learning and testing.
    """

    """ Get data and target tables. """
    data, target, _, _ = get_data_target.get_original_data(scores_file, delimiter, target_size, "reg")

    all_targets = []
    for row in rows:
        scores_file = data_dir + "/" + row[:-1]
        _, raw_target, _, _ = get_data_target.get_original_data(scores_file, delimiter, target_size, "reg")
        all_targets.append(raw_target)

    all_targets = np.array(all_targets)
    print(all_targets.shape)

    wild_type = 11  # eleventh target attribute is a wild-type

    """ Neural network architecture initialisation. """
    class_size = 3
    n_hidden_n = int(max(data.shape[1], target.shape[1]) * 2 / 3)
    n_hidden_l = 2

    # net = NnetRegLearner.NnetRegLearner(data.shape[1] + (target_size - 1), 1, n_hidden_l, n_hidden_n)  # protwildtime
    # net = NnetRegLearner.NnetRegLearner(data.shape[1] + (target_size - 1) * len(rows), 1, n_hidden_l, n_hidden_n)  # protwildexptime
    # net = NnetRegLearner.NnetRegLearner(len(rows), 1, n_hidden_l, n_hidden_n)  # wildtime
    net = NnetRegLearner.NnetRegLearner((target_size - 1) * len(rows), 1, n_hidden_l, n_hidden_n)  # wildexptime

    rhos = []
    p_values = []
    all_probs = []
    all_ids = []

    ids_prob = get_data_target.get_ids(scores_file, delimiter, 'ID')

    down_per = 10  # 10th percentile, decides over analysis on Orange (obvious groups of expressions)
    up_per = 90  # 90th percentile, decides over analysis on Orange (obvious groups of expressions)

    # max_len = nnet_reg_one.get_max_len(target, target_size, down_per, up_per)  # balanced data
    max_len = target.shape[0] / class_size  # unbalanced data

    # for t in range(target_size):
    for t in np.hstack([range(wild_type), range(wild_type+1, target_size)]):
        target_r = target[:, t]
        data_c = data
        # data_c = np.hstack([data_c, all_targets[:, :, wild_type].T])  # protwildtime
        # data_c = np.hstack([data_c, np.hstack(all_targets[:, :, :t]), np.hstack(all_targets[:, :, t + 1:])])  # protwildexptime
        # data_c = all_targets[:, :, wild_type].T  # wildtime
        data_c = np.hstack([np.hstack(all_targets[:, :, :t]), np.hstack(all_targets[:, :, t + 1:])])  # wildexptime
        print(data_c.shape)

        """ Ignore missing attributes """
        if len(np.unique(target_r)) == 1:
            rhos.append(0)
            p_values.append(0)
            all_probs.append(np.zeros((int(max_len * class_size), 2)))
            all_ids.append(np.zeros((int(max_len * class_size), 1)).astype(str))
            print(max_len * class_size, 2)
            continue

        # data_b, targets, ids_b = nnet_reg_one.get_balanced_data(target_r, data_c, max_len, ids_prob, down_per, up_per)  # balanced data
        targets, data_b, ids_b = target_r.reshape(-1, 1), data_c, ids_prob  # unbalanced data

        probs = np.zeros((targets.shape[0], 2))
        ids_end = np.zeros((targets.shape[0], 1)).astype(str)

        print(probs.shape)
        """ Split to train and test 10-fold Cross-Validation """
        skf = KFold(targets.shape[0], n_folds=10, shuffle=True)
        idx = 0
        for train_index, test_index in skf:
            # trX, teX = data_b, data_b  # for testing on training data (overfitting)
            trX, teX = data_b[train_index], data_b[test_index]
            # trY, teY = targets, targets  # for testing on training data (overfitting)
            trY, teY = targets[train_index], targets[test_index]

            """ Learning and predicting """
            net.fit(trX, trY)
            prY = net.predict(teX)
            print(np.sqrt(np.mean(np.square(teY - np.mean(trY)))), np.sqrt(np.mean(np.square(teY - prY))))

            """ Storing results... """
            probs[idx:idx+len(teY), 0] = np.around(prY, decimals=2).flatten()
            probs[idx:idx+len(teY), 1] = teY.flatten()
            ids_end[idx:idx+len(teY), 0] = ids_b[test_index].flatten()
            # ids_end[idx:idx+len(teY), 0] = ids_b.flatten()  # for testing on training data (overfitting)
            idx += len(teY)

        all_probs.append(np.around(probs, decimals=2))
        all_ids.append(ids_end)
        rho, p = spearmanr(probs[:, 0], probs[:, 1])
        rhos.append(rho)
        p_values.append(p)
    return rhos, p_values, np.hstack(all_probs), np.hstack(all_ids)

