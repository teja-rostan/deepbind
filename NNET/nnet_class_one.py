import sys
import time
import numpy as np
from sklearn.cross_validation import KFold
from scipy.stats import spearmanr
from NNET import get_data_target, NnetClassLearner


def learn_and_score(scores_file, delimiter, target_size):
    """Learning and correlation scoring input data with classificational Neural Network, one target per time. """

    """ Get data and target tables. """
    data, target, raw_target, target_class = get_data_target.get_original_data(scores_file, delimiter, target_size, "class")

    wild_type = 11  # eleventh target attribute is a wild-type

    # data = np.hstack([data, raw_target[:, :wild_type], raw_target[:, wild_type + 1:]]) # protexp
    # data = np.hstack([raw_target[:, :wild_type], raw_target[:, wild_type + 1:]])  # exp

    # data = np.hstack([data, raw_target[:, wild_type].reshape(-1, 1)])  # protwild
    # data = np.hstack([raw_target[:, wild_type].reshape(-1, 1)])  # wild

    """ Neural network architecture initialisation. """
    class_size = 3
    n_hidden_l = 2
    n_hidden_n = int(max(data.shape[1], target.shape[1]) * 2 / 3)

    # net = NnetClassLearner.NnetClassLearner(data.shape[1], class_size, n_hidden_l, n_hidden_n) # wild/protwild or exp/protexp
    # net = NnetClassLearner.NnetClassLearner(data.shape[1] + target_size - 1, class_size, n_hidden_l, n_hidden_n)  # protwildexp
    net = NnetClassLearner.NnetClassLearner(target_size - 1, class_size, n_hidden_l, n_hidden_n)  # wildexp

    rhos = []
    p_values = []
    all_probs = []
    all_ids = []
    ids = get_data_target.get_ids(scores_file, delimiter, 'ID')
    # max_len = get_max_len(target, target_class, target_size)  # balanced data
    max_len = target.shape[0] / class_size  # unbalanced data

    # for t in range(target_size):
    for t in np.hstack([range(wild_type), range(wild_type+1, target_size)]):
        target_c = target_class[:, t]
        # data_c = np.hstack([data, raw_target[:, :t], raw_target[:, t + 1:]])  # protwildexp
        data_c = np.hstack([raw_target[:, :t], raw_target[:, t + 1:]])  # wildexp
        # data_c = data  # # wild/protwild or exp/protexp

        """ Ignore missing attributes """
        if len(np.unique(target_c)) == 1:
            rhos.append(0)
            p_values.append(0)
            all_probs.append(np.zeros((max_len * class_size, 2)))
            all_ids.append(np.zeros((max_len * class_size, 1)).astype(str))
            print(max_len * class_size, 2)
            continue

        # targets, ids_b, data_b = get_balanced_data(t, class_size, target_c, data_c, target, max_len, ids)  # balanced data
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
            print(majority(trY, teY)[0], np.mean(np.argmax(teY, axis=1) == prY))

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


def get_max_len(target, target_class, target_size):
    """ Get length of one class in balanced data """

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
    """ Balancing data (Make a set of instances of every class to equal size). """

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


def data_selection(data, size):
    """ Selecting up to 80 attributes with highest number of outliers. (fast fix)"""

    data = data[:, [517, 625, 622, 105, 602, 130, 816, 77, 375, 209, 274, 687, 514, 548, 760, 493, 733, 615, 768, 135,
                    156, 95, 679, 264, 680, 327, 923, 504, 515, 924, 57, 838, 589, 195, 306, 398, 248, 546, 117, 632,
                    726, 372, 345, 823, 80, 29, 706, 669, 74, 720, 177, 412, 174, 91, 829, 877, 811, 419, 145, 761,
                    165, 509, 847, 777, 532, 475, 137, 782, 700, 445, 513, 478, 118, 54, 499, 104, 166, 738, 26, 146]]
    return data[:, :size]


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


def main():
    start = time.time()
    arguments = sys.argv[1:]

    if len(arguments) < 3:
        print("Not enough arguments stated! Usage: \n"
              "python nnet_reg_one.py <scores_file_path> <predicted_scores_path> <delimiter> <target_size>.")
        sys.exit(0)

    scores_file = arguments[0]
    delimiter = arguments[1]
    target_size = int(arguments[2])

    learn_and_score(scores_file, delimiter, target_size)

    end = time.time() - start
    print("Program run for %.2f seconds." % end)

if __name__ == '__main__':
    main()
