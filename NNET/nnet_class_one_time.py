import numpy as np
from sklearn.cross_validation import KFold
from scipy.stats import spearmanr
from NNET import get_data_target, NnetClassLearner, nnet_class_one


def learn_and_score(scores_file, data_dir, rows, delimiter, target_size):
    """
    Neural network learning and correlation scoring. Classificational Neural Network, learning and predicting one target
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
    data, target, raw_target, target_class = get_data_target.get_original_data(scores_file, delimiter, target_size,
                                                                               "class")

    all_targets = []
    for row in rows:
        scores_file = data_dir + "/" + row[:-1]
        _, _, raw_target, _ = get_data_target.get_original_data(scores_file, delimiter, target_size, "class")
        all_targets.append(raw_target)

    all_targets = np.array(all_targets)
    print(all_targets.shape)

    wild_type = 11  # eleventh target attribute is a wild-type

    """ Neural network architecture initialisation. """
    class_size = 3
    n_hidden_l = 2
    n_hidden_n = int(max(data.shape[1], target.shape[1]) * 2 / 3)

    # net = NnetClassLearner.NnetClassLearner(data.shape[1] + (target_size - 1), class_size, n_hidden_l, n_hidden_n)  # protwildtime
    # net = NnetClassLearner.NnetClassLearner(data.shape[1] + (target_size - 1) * len(rows), class_size, n_hidden_l, n_hidden_n)  # protwildtexptime
    # net = NnetClassLearner.NnetClassLearner(len(rows), class_size, n_hidden_l, n_hidden_n)  # wildtime
    net = NnetClassLearner.NnetClassLearner((target_size - 1) * len(rows), class_size, n_hidden_l, n_hidden_n)  # wildtexptime

    rhos = []
    p_values = []
    all_probs = []
    all_ids = []

    ids = get_data_target.get_ids(scores_file, delimiter, 'ID')

    # max_len = nnet_class_one.get_max_len(target, target_class, target_size)  # balanced data
    max_len = target.shape[0] / class_size  # unbalanced data

    # for t in range(target_size):
    for t in np.hstack([range(wild_type), range(wild_type+1, target_size)]):
        target_c = target_class[:, t]
        data_c = data
        # data_c = np.hstack([data_c, all_targets[:, :, wild_type].T])  # protwildtime
        # data_c = np.hstack([data_c, np.hstack(all_targets[:, :, :t]), np.hstack(all_targets[:, :, t + 1:])])  # protwildexptime
        # data_c = all_targets[:, :, wild_type].T # wildtime
        data_c = np.hstack([np.hstack(all_targets[:, :, :t]), np.hstack(all_targets[:, :, t + 1:])])  # wildexptime
        print(data_c.shape)

        """ Ignore missing attributes """
        if len(np.unique(target_c)) == 1:
            rhos.append(0)
            p_values.append(0)
            all_probs.append(np.zeros((max_len * class_size, 2)))
            all_ids.append(np.zeros((max_len * class_size, 1)).astype(str))
            print(max_len * class_size, 2)
            continue

        # targets, ids_b, data_b = nnet_class_one.get_balanced_data(t, class_size, target_c, data_c, target, max_len, ids)  # balanced data
        targets, data_b, ids_b = target[:, class_size * t:class_size * t + class_size], data_c, ids  # unbalanced data

        probs = np.zeros((targets.shape[0], 2))
        ids_end = np.zeros((targets.shape[0], 1)).astype(str)

        print(probs.shape)
        """ Split to train and test 10-fold Cross-Validation """
        skf = KFold(targets.shape[0], n_folds=10, shuffle=True)
        idx = 0
        for train_index, test_index in skf:
            trX, teX = data_b[train_index], data_b[test_index]
            # trX, teX = data_b, data_b  # for testing on training data (overfitting)
            trY, teY = targets[train_index], targets[test_index]
            # trY, teY = targets, targets  # for testing on training data (overfitting)

            """ Learning and predicting """
            net.fit(trX, trY)
            prY = net.predict(teX)
            print(nnet_class_one.majority(trY, teY)[0], np.mean(np.argmax(teY, axis=1) == prY))

            """ Storing results... """
            probs[idx:idx+len(teY), 0] = prY.flatten()
            probs[idx:idx+len(teY), 1] = np.argmax(teY, axis=1).flatten()
            ids_end[idx:idx+len(teY), 0] = ids_b[test_index].flatten()
            # ids_end[idx:idx+len(teY), 0] = ids_b.flatten()  # for testing on training data (overfitting)
            idx += len(teY)
        all_probs.append(np.around(probs, decimals=2))
        all_ids.append(ids_end)
        rho, p = spearmanr(probs[:, 0], probs[:, 1])
        rhos.append(rho)
        p_values.append(p)
    return rhos, p_values, np.hstack(all_probs), np.hstack(all_ids)
