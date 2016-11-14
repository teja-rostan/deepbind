import sys
import time

import numpy as np
import theano
from theano import tensor as T

from NNET import nnet
from NNET import get_data_target

from sklearn.cross_validation import KFold
from scipy.stats import spearmanr


def ac_score(y_true, y_predicted):
    """ Multi-target scoring with classification accuracy. """

    # low    00
    # medium 01
    # high   11

    k = 2
    pred_prob = []
    true_class = []
    predicted_class = []
    print(np.mean(y_predicted, axis=0))
    for i in range(int(y_true.shape[1] / k)):
        col_true = y_true[:, i * k:i * k + k]
        col_predicted = y_predicted[:, i * k:i * k + k]
        col_true = np.sum(col_true, axis=1)

        first = col_predicted[:, 0] > 0.65
        second = col_predicted[:, 1] > 0.5
        col_predicted = np.zeros(col_predicted.shape[0])
        col_predicted += first
        col_predicted += np.logical_and(first, second)

        true_class.append(col_true)
        predicted_class.append(col_predicted)
    return pred_prob, true_class, predicted_class


def learn_and_score(scores_file, delimiter, target_size):
    """Learning and scoring input data with neural network (and majority classifier for reference). """

    """ Get data and target tables. """
    data, target, raw_exps, target_class = get_data_target.get_original_data(scores_file, delimiter, target_size, "ord")

    """ Neural network architecture initialisation. """
    n_hidden = int(max(data.shape[1], target.shape[1]) * 2 / 3)

    _, counts = np.unique(target_class.flatten(), return_counts=True)
    counts = max(counts) / np.array(counts)
    class_prob = counts[target_class.astype(int)]
    class_prob = np.repeat(class_prob, 2, axis=1)

    X = T.fmatrix()
    Y = T.fmatrix()
    W = T.fmatrix()

    w_h = nnet.init_weights((data.shape[1], n_hidden))
    w_h2 = nnet.init_weights((n_hidden, n_hidden))
    # w_h3 = nnet.init_weights((n_hidden, n_hidden))
    w_o = nnet.init_weights((n_hidden, target.shape[1]))

    # noise_py_x = nnet.model_sig(X, w_h, w_h2, w_h3, w_o, 0.2, 0.5)
    # py_x = nnet.model_sig(X, w_h, w_h2, w_h3, w_o, 0., 0.)
    noise_py_x = nnet.model_sig(X, w_h, w_h2, w_o, 0.2, 0.5)
    py_x = nnet.model_sig(X, w_h, w_h2, w_o, 0., 0.)

    # cost = T.mean(T.nnet.categorical_crossentropy(noise_py_x, Y))
    cost = T.mean(nnet.weighted_entropy(noise_py_x, Y, W))

    # params = [w_h, w_h2, w_h3, w_o]
    params = [w_h, w_h2, w_o]
    updates = nnet.RMSprop(cost, params, lr=0.001)

    train = theano.function(inputs=[X, Y, W], outputs=cost, updates=updates, allow_input_downcast=True)
    predict = theano.function(inputs=[X], outputs=py_x, allow_input_downcast=True)

    """ Leave one out Cross-Validation """
    id = get_data_target.get_ids(scores_file, delimiter, 'ID')
    ids_prob = []

    probs = np.zeros((len(target), target_size * 3))
    idx = 0
    skf = KFold(target.shape[0], n_folds=10, shuffle=True)
    for train_index, test_index in skf:
        trX, teX = data[train_index], data[test_index]
        trY, teY = target[train_index], target[test_index]

        """ Learning... """
        for i in range(100):
            shuffle = np.random.permutation(len(trY))
            trYs = trY[shuffle]
            trXs = trX[shuffle]
            class_probs = class_prob[shuffle]
            for start, end in zip(range(0, len(trXs), 128), range(128, len(trXs), 128)):
                cost = train(trXs[start:end], trYs[start:end], class_probs[start:end])
        predictions = predict(teX)

        ids_prob.extend(id[test_index])
        probs[idx:idx+len(teY), :target_size*2] = predictions
        probs[idx:idx+len(teY), target_size*2:target_size*3] = raw_exps[test_index]
        idx += len(teY)

        """ Scoring... """
        # score = ac_score(teY, predictions)

        """ Randomizing weights for new fold (to overcome overfitting)."""
        w_h.set_value(nnet.rand_weights((data.shape[1], n_hidden)))
        w_h2.set_value(nnet.rand_weights((n_hidden, n_hidden)))
        # w_h3.set_value(nnet.rand_weights((n_hidden, n_hidden)))
        w_o.set_value(nnet.rand_weights((n_hidden, target.shape[1])))

    """ Spreaman correlation """
    probs0 = probs[:, 0:target_size*2:2]
    probs1 = probs[:, 1:target_size*2:2]
    spears = []
    for i in range(target_size):
        rho0, p0 = spearmanr(target_class[:, i], probs0[:, i])
        rho1, p1 = spearmanr(target_class[:, i], probs1[:, i])

        spear = np.stack([rho0, p0, rho1, p1], axis=0)
        spears.append(spear)

    return probs, ids_prob, spears


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
