import theano
from theano import tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from theano.tensor.nnet.conv import conv2d
from theano.tensor.signal.downsample import max_pool_2d
import numpy as np


srng = RandomStreams()


def floatX(X):
    return np.asarray(X, dtype=theano.config.floatX)


def init_weights(shape):
    return theano.shared(floatX(np.random.randn(*shape) * 0.01))


def rand_weights(shape):
    return (np.random.randn(*shape) * 0.01).astype(np.float32)


def rectify(X):
    return T.maximum(X, 0.)


def softmax(X):
    e_x = T.exp(X - X.max(axis=1).dimshuffle(0, 'x'))
    return e_x / e_x.sum(axis=1).dimshuffle(0, 'x')


def sigmoid(X):
    e_x = T.exp(X - X.max(axis=1).dimshuffle(0, 'x'))
    return 1 / (1 + e_x)


def RMSprop(cost, params, lr=0.001, rho=0.9, epsilon=1e-6):
    grads = T.grad(cost=cost, wrt=params)
    updates = []
    for p, g in zip(params, grads):
        acc = theano.shared(p.get_value() * 0.)
        acc_new = rho * acc + (1 - rho) * g ** 2
        gradient_scaling = T.sqrt(acc_new + epsilon)
        g = g / gradient_scaling
        updates.append((acc, acc_new))
        updates.append((p, p - lr * g))
    return updates


def dropout(X, p=0.):
    if p > 0:
        retain_prob = 1 - p
        X *= srng.binomial(X.shape, p=retain_prob, dtype=theano.config.floatX)
        X /= retain_prob
    return X


def weighted_entropy(noise_py_x, Y, W):
    return -T.sum((Y * T.log(noise_py_x) + (1 - Y) * T.log(1 - noise_py_x)) * W, axis=noise_py_x.ndim - 1)


def relative_entropy(noise_py_x, Y):
    return -T.sum(Y * T.log(noise_py_x) + (1 - Y) * T.log(1 - noise_py_x), axis=noise_py_x.ndim - 1)
    # return -T.sum(Y * T.log(noise_py_x) + (1 - Y), axis=noise_py_x.ndim - 1)


def rmse(noise_py_x, Y):
    return T.sqrt(T.mean(T.sqr(Y - noise_py_x)))


def model_sig(X, w_h, w_h2, w_o, p_drop_input, p_drop_hidden):
    X = dropout(X, p_drop_input)
    h = rectify(T.dot(X, w_h))

    h = dropout(h, p_drop_hidden)
    h2 = rectify(T.dot(h, w_h2))

    h2 = dropout(h2, p_drop_hidden)
    py_x = sigmoid(T.dot(h2, w_o))
    return py_x


def model2(X, w_h, w_h2, w_o, p_drop_input, p_drop_hidden):
    X = dropout(X, p_drop_input)
    h = rectify(T.dot(X, w_h))

    h = dropout(h, p_drop_hidden)
    h2 = rectify(T.dot(h, w_h2))

    h2 = dropout(h2, p_drop_hidden)
    py_x = softmax(T.dot(h2, w_o))
    return py_x


def model3(X, w_h, w_h2, w_h3, w_o, p_drop_input, p_drop_hidden):
    X = dropout(X, p_drop_input)
    h = rectify(T.dot(X, w_h))

    h = dropout(h, p_drop_hidden)
    h2 = rectify(T.dot(h, w_h2))

    h2 = dropout(h2, p_drop_hidden)
    h3 = rectify(T.dot(h2, w_h3))

    h3 = dropout(h3, p_drop_hidden)
    py_x = softmax(T.dot(h3, w_o))
    return py_x


def model4(X, w_h, w_h2, w_h3, w_h4, w_o, p_drop_input, p_drop_hidden):
    X = dropout(X, p_drop_input)
    h = rectify(T.dot(X, w_h))

    h = dropout(h, p_drop_hidden)
    h2 = rectify(T.dot(h, w_h2))

    h2 = dropout(h2, p_drop_hidden)
    h3 = rectify(T.dot(h2, w_h3))

    h3 = dropout(h3, p_drop_hidden)
    h4 = rectify(T.dot(h3, w_h4))

    h4 = dropout(h4, p_drop_hidden)
    py_x = softmax(T.dot(h4, w_o))
    return py_x


def model_reg(X, w_h, w_h2, w_o, p_drop_input, p_drop_hidden):
    X = dropout(X, p_drop_input)
    h = rectify(T.dot(X, w_h))

    h = dropout(h, p_drop_hidden)
    h2 = rectify(T.dot(h, w_h2))

    h2 = dropout(h2, p_drop_hidden)
    py_x = (T.dot(h2, w_o))
    return py_x


def model_reg3(X, w_h, w_h2, w_h3, w_o, p_drop_input, p_drop_hidden):
    X = dropout(X, p_drop_input)
    h = rectify(T.dot(X, w_h))

    h = dropout(h, p_drop_hidden)
    h2 = rectify(T.dot(h, w_h2))

    h2 = dropout(h2, p_drop_hidden)
    h3 = rectify(T.dot(h2, w_h3))

    h3 = dropout(h3, p_drop_hidden)
    py_x = (T.dot(h3, w_o))
    return py_x


def conv_model(X, w_h, w_h2, w_h3, w_h4, w_h5, w_o, p_drop_conv, p_drop_hidden):
    h1 = rectify(conv2d(X, w_h, border_mode='full'))
    h1 = max_pool_2d(h1, (1, 6))

    h1 = dropout(h1, p_drop_conv)
    h2 = rectify(conv2d(h1, w_h2))
    h2 = max_pool_2d(h2, (1, 8))

    h2 = dropout(h2, p_drop_conv)
    h3 = rectify(conv2d(h2, w_h3))
    h3 = max_pool_2d(h3, (1, 10))
    h3 = T.flatten(h3, outdim=2)

    h3 = dropout(h3, p_drop_conv)
    h4 = rectify(T.dot(h3, w_h4))

    h4 = dropout(h4, p_drop_conv)
    h5 = rectify(T.dot(h4, w_h5))

    h5 = dropout(h5, p_drop_hidden)
    pyx = softmax(T.dot(h5, w_o))
    return pyx
