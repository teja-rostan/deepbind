"""
    Artificial neural network with 2 or 3 hidden layers
"""
from NNET import nnet
import numpy as np
import theano
from theano import tensor as T


class NnetClassLearner:

    def __init__(self, attribute_size, class_size, n_hidden_layers=2, n_hidden_neurons=30):
        """
        Initialization of Classification neural network.
        :param attribute_size: Number of input attributes for neural network.
        :param class_size: Number of output classes for neural network.
        :param n_hidden_layers: Number of hidden layers in neural network architecture.
        :param n_hidden_neurons: Number of hidden neurons in every hidden layer in neural network architecture.

        """
        self.n_hidden_layers = n_hidden_layers
        self.class_size = class_size
        self.n_hidden_neurons = n_hidden_neurons
        self.attribute_size = attribute_size

        X = T.fmatrix()
        Y = T.fmatrix()

        self.w_h = nnet.init_weights((self.attribute_size, self.n_hidden_neurons))
        self.w_h2 = nnet.init_weights((self.n_hidden_neurons, self.n_hidden_neurons))
        self.w_o = nnet.init_weights((self.n_hidden_neurons, self.class_size))

        if self.n_hidden_layers == 2:

            noise_py_x = nnet.model2(X, self.w_h, self.w_h2, self.w_o, 0.2, 0.5)
            py_x = nnet.model2(X, self.w_h, self.w_h2, self.w_o, 0., 0.)

            cost = T.mean(T.nnet.categorical_crossentropy(noise_py_x, Y))
            params = [self.w_h, self.w_h2, self.w_o]
            updates = nnet.RMSprop(cost, params, lr=0.001)

            self.train = theano.function(inputs=[X, Y], outputs=cost, updates=updates, allow_input_downcast=True)
            self.predict_ = theano.function(inputs=[X], outputs=py_x, allow_input_downcast=True)

        elif self.n_hidden_layers == 3:

            self.w_h3 = nnet.init_weights((self.n_hidden_neurons, self.n_hidden_neurons))

            noise_py_x = nnet.model3(X, self.w_h, self.w_h2, self.w_h3, self.w_o, 0.2, 0.5)
            py_x = nnet.model3(X, self.w_h, self.w_h2, self.w_h3, self.w_o, 0., 0.)

            cost = T.mean(T.nnet.categorical_crossentropy(noise_py_x, Y))
            params = [self.w_h, self.w_h2, self.w_h3, self.w_o]
            updates = nnet.RMSprop(cost, params, lr=0.001)

            self.train = theano.function(inputs=[X, Y], outputs=cost, updates=updates, allow_input_downcast=True)
            self.predict_ = theano.function(inputs=[X], outputs=py_x, allow_input_downcast=True)

        elif self.n_hidden_layers == 4:

            self.w_h3 = nnet.init_weights((self.n_hidden_neurons, self.n_hidden_neurons))
            self.w_h4 = nnet.init_weights((self.n_hidden_neurons, self.n_hidden_neurons))

            noise_py_x = nnet.model4(X, self.w_h, self.w_h2, self.w_h3, self.w_h4, self.w_o, 0.2, 0.5)
            py_x = nnet.model4(X, self.w_h, self.w_h2, self.w_h3, self.w_h4, self.w_o, 0., 0.)

            cost = T.mean(T.nnet.categorical_crossentropy(noise_py_x, Y))
            params = [self.w_h, self.w_h2, self.w_h3, self.w_h4, self.w_o]
            updates = nnet.RMSprop(cost, params, lr=0.001)

            self.train = theano.function(inputs=[X, Y], outputs=cost, updates=updates, allow_input_downcast=True)
            self.predict_ = theano.function(inputs=[X], outputs=py_x, allow_input_downcast=True)

    def fit(self, trX, trY):
        """
        Neural Network learning.
        :param trX: Input data for training (train X)
        :param trY: Output data for training (train y)

        """
        for i in range(100):
            shuffle = np.random.permutation(len(trY))
            trYs = trY[shuffle]
            trXs = trX[shuffle]
            for start, end in zip(range(0, len(trX), 128), range(128, len(trX), 128)):
                cost = self.train(trXs[start:end], trYs[start:end])

    def predict(self, teX):
        """
        Neural network predicting.
        :param teX: Input data for predicting (test X)
        :return: Predictions
        """
        predictions = self.predict_(teX)
        prY = np.argmax(predictions, axis=1)

        """ Randomize weights after training and predicting for new round"""
        self.w_h.set_value(nnet.rand_weights((self.attribute_size, self.n_hidden_neurons)))
        self.w_h2.set_value(nnet.rand_weights((self.n_hidden_neurons, self.n_hidden_neurons)))
        self.w_o.set_value(nnet.rand_weights((self.n_hidden_neurons, self.class_size)))

        if self.n_hidden_layers > 2:
            self.w_h3.set_value(nnet.rand_weights((self.n_hidden_neurons, self.n_hidden_neurons)))

        if self.n_hidden_layers == 4:
            self.w_h4.set_value(nnet.rand_weights((self.n_hidden_neurons, self.n_hidden_neurons)))
        return prY
