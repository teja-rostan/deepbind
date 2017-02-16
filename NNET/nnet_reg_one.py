import numpy as np
from sklearn.cross_validation import KFold
from scipy.stats import spearmanr
from NNET import get_data_target, NnetRegLearner


def learn_and_score(scores_file, delimiter, target_size):
    """
    Neural network learning and correlation scoring. Regressional Neural Network, learning and predicting one target
    per time on balanced or unbalanced data.
    :param scores_file: The file with data and targets for neural network learning.
    :param delimiter: The delimiter for the scores_file.
    :param target_size: Number of targets in scores_file (the number of columns from the end of scores_file that we want
    to extract and double).
    :return: rhos and p-values of relative Spearman correlation, predictions and ids of instances that were included
    in learning and testing.
    """

    """ Get data and target tables. """
    data, target, _, _ = get_data_target.get_original_data(scores_file, delimiter, target_size, "reg")

    wild_type = 11  # eleventh target attribute is a wild-type
    # prot, exp, protexp
    # wild, wildexp, protwild, protwildexp

    # data = np.hstack([data, target[:, :wild_type], target[:, wild_type + 1:]]) # protexp.
    # data = np.hstack([target[:, :wild_type], target[:, wild_type + 1:]])  # exp

    # data = np.hstack([data, target[:, wild_type].reshape(-1, 1)])  # protwild
    # data = np.hstack([target[:, wild_type].reshape(-1, 1)])  # wild

    """ Neural network architecture initialisation. """
    class_size = 3
    n_hidden_l = 2
    n_hidden_n = int(max(data.shape[1], target.shape[1]) * 2 / 3)

    net = NnetRegLearner.NnetRegLearner(data.shape[1], 1, n_hidden_l, n_hidden_n)  # wild/protwild or exp/protexp
    # net = NnetRegLearner.NnetRegLearner(data.shape[1] + target_size - 1, 1, n_hidden_l, n_hidden_n)  # protwildexp
    # net = NnetRegLearner.NnetRegLearner(target_size - 1, 1, n_hidden_l, n_hidden_n)  # wildexp

    rhos = []
    p_values = []
    all_probs = []
    all_ids = []
    ids_prob = get_data_target.get_ids(scores_file, delimiter, 'ID')

    down_per = 10  # 10th percentile, decides over analysis on Orange (obvious groups of expressions)
    up_per = 90  # 90th percentile, decides over analysis on Orange (obvious groups of expressions)

    max_len = get_max_len(target, target_size, down_per, up_per)  # balanced data
    # max_len = target.shape[0] / class_size  # unbalanced data

    for t in range(target_size):
    # for t in np.hstack([range(wild_type), range(wild_type+1, target_size)]):
        target_r = target[:, t]
        # data_c = np.hstack([data, target[:, :t], target[:, t + 1:]])  # protwildexp
        # data_c = np.hstack([target[:, :t], target[:, t + 1:]])  # wildexp
        data_c = data  # wild/protwild or exp/protexp

        """ Ignore missing attributes """
        if len(np.unique(target_r)) == 1:
            rhos.append(0)
            p_values.append(0)
            all_probs.append(np.zeros((int(max_len * class_size), 2)))
            all_ids.append(np.zeros((int(max_len * class_size), 1)).astype(str))
            print(max_len * class_size, 2)
            continue

        data_b, targets, ids_b = get_balanced_data(target_r, data_c, max_len, ids_prob, down_per, up_per)  # balanced data
        # targets, data_b, ids_b = target_r.reshape(-1, 1), data_c, ids_prob  # unbalanced data

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


def get_max_len(target, target_size, down_per, up_per):
    """
    Get maximum possible number of instances of one class to get balanced data through all target features.
    :param target: all feature targets of raw expressions
    :param target_size: number of target features
    :param down_per: lower threshold for classification of targets as percentile
    :param up_per: upper threshold for classification of targets as percentile
    :return: max class size
    """

    max_len = len(target)
    for t in range(target_size):
        targets = target[:, t]
        if len(np.unique(targets)) == 1:
            continue
        down_10 = np.percentile(targets, down_per)
        up_10 = np.percentile(targets, up_per)
        down_10 = np.sum(targets < down_10)
        up_10 = np.sum(targets > up_10)
        if max_len > min(down_10, up_10):
            max_len = min(down_10, up_10)
    return max_len


def get_balanced_data(targets, data, max_len, ids, down_per, up_per):
    """
    Balancing data (Make a set of instances of every class to equal size).
    :param targets: target feature that we want to balance
    :param data: the input attributes that we sample based on target balancing
    :param max_len: maximum number of instances per class
    :param ids: attribute names that we sample based on target balancing
    :param down_per: lower threshold for classification of targets as percentile
    :param up_per: upper threshold for classification of targets as percentile
    :return: returns balanced data, target and ids
    """

    down_10 = np.percentile(targets, down_per)
    up_10 = np.percentile(targets, up_per)
    nc = np.logical_and(down_10 <= targets, targets <= up_10)
    len_nc = np.sum(nc)

    id_nc = ids[nc]
    data_nc = data[nc]
    target_nc = targets[nc]

    shuffle = np.random.permutation(len_nc)
    target_nc = target_nc[shuffle][:max_len]
    id_nc = id_nc[shuffle][:max_len]
    data_nc = data_nc[shuffle][:max_len]

    ids_prob = np.vstack([ids[down_10 >= targets][:max_len].reshape(-1, 1),
                          ids[targets >= up_10][:max_len].reshape(-1, 1),
                          id_nc.reshape(-1, 1)])
    target_nc = np.vstack([targets[down_10 >= targets][:max_len].reshape(-1, 1),
                           targets[targets >= up_10][:max_len].reshape(-1, 1),
                           target_nc.reshape(-1, 1)])
    data_nc = np.vstack([data[down_10 >= targets][:max_len],
                         data[targets >= up_10][:max_len],
                         data_nc])

    shuffle = np.random.permutation(len(target_nc))
    targets = target_nc[shuffle]
    ids_prob = ids_prob[shuffle]
    datas = data_nc[shuffle]
    return datas, targets, ids_prob

