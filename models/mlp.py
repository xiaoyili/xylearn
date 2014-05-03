"""
This tutorial introduces the multilayer perceptron using Theano.

 A multilayer perceptron is a logistic regressor where
instead of feeding the input to the logistic regression you insert a
intermediate layer, called the hidden layer, that has a nonlinear
activation function (usually tanh or sigmoid) . One can use many such
hidden layers making the architecture deep. The tutorial will also tackle
the problem of MNIST digit classification.

.. math::

    f(x) = G( b^{(2)} + W^{(2)}( s( b^{(1)} + W^{(1)} x))),

References:

    - textbooks: "Pattern Recognition and Machine Learning" -
                 Christopher M. Bishop, section 5

"""
__docformat__ = 'restructedtext en'

import numpy

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
from xylearn.models.softmax_regression import SoftmaxRegression

from xylearn.utils.theano_util import toSharedX


# for debug
from xylearn.visualizer.terminal_printer import TerminalPrinter


class HiddenLayer(object):
    def __init__(self, rng, input=None, n_in=None, n_out=None, W=None, b=None,
                 activation=T.tanh, identity=0, dropout=None):
        """
        Typical hidden layer of a MLP: units are fully-connected and have
        sigmoidal activation function. Weight matrix W is of shape (n_in,n_out)
        and the bias vector b is of shape (n_out,).

        NOTE : The nonlinearity used here is tanh

        Hidden unit activation is given by: tanh(dot(input,W) + b)

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dmatrix
        :param input: a symbolic tensor of shape (n_examples, n_in)

        :type n_in: int
        :param n_in: dimensionality of input

        :type n_out: int
        :param n_out: number of hidden units

        :type activation: theano.Op or function
        :param activation: Non linearity to be applied in the hidden
                           layer

        :type identity: int
        :param identity: do the transformation or not?

        :type dropout: 0 ~ 1 prob
        :param dropout: dropout x% neurons at random
        """

        if input is None:
            self.x = T.matrix('x_hl')  # symbolic input
        else:
            self.x = input

        self.identity = identity

        if not self.identity:

            if W is None:
                W_values = numpy.asarray(rng.uniform(
                    low=-numpy.sqrt(6. / (n_in + n_out)),
                    high=numpy.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)), dtype=theano.config.floatX)
                if activation == theano.tensor.nnet.sigmoid:
                    W_values *= 4

                W = theano.shared(value=W_values, name='W_hl', borrow=True)

            if b is None:
                b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
                b = theano.shared(value=b_values, name='b_hl', borrow=True)

            self.W = W
            self.b = b

            lin_output = T.dot(self.x, self.W) + self.b

            self.output = (lin_output if activation is None
                           else activation(lin_output))

            if dropout is not None:
                # dropout is just a mask
                theano_rng = RandomStreams(rng.randint(2 ** 30))
                mask = theano_rng.binomial(size=self.output.shape,
                                           n=1, p=1 - dropout,
                                           dtype=theano.config.floatX)
                self.output *= mask

            # parameters of the model
            self.params = [self.W, self.b]

        else:
            self.output = self.x
            self.params = []


class MLP(object):
    """Multi-Layer Perceptron Class

    A multilayer perceptron is a feedforward artificial neural network model
    that has one layer or more of hidden units and nonlinear activations.
    Intermediate layers usually have as activation function thanh or the
    sigmoid function (defined here by a ``SigmoidalLayer`` class)  while the
    top layer is a softamx layer (defined here by a ``SoftmaxRegression``
    class).
    """

    def __init__(self, rng=None, input=None,
                 n_in=None, n_hidden=None, n_out=None):
        """Initialize the parameters for the multilayer perceptron

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
        architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
        which the datapoints lie

        :type n_hidden: int
        :param n_hidden: number of hidden units

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
        which the labels lie

        """

        if input is None:
            self.x = T.matrix('x')  # symbolic input
        else:
            self.x = input

        self.y = T.ivector('y')  # symbolic target

        self.hidden_layer = HiddenLayer(rng=rng, input=self.x,
                                        n_in=n_in, n_out=n_hidden,
                                        activation=T.tanh)


        # The logistic regression layer gets as input the hidden units
        # of the hidden layer
        self.softmax_layer = SoftmaxRegression(
            input=self.hidden_layer.output,
            n_in=n_hidden,
            n_out=n_out)

        self.p_y_given_x = self.softmax_layer.p_y_given_x

        self.y_pred = T.argmax(self.p_y_given_x, axis=1)

        # the parameters of the model are the parameters of the two layer it is
        # made out of
        self.params = self.hidden_layer.params + self.softmax_layer.params

    def _negative_log_likelihood(self, y):
        # negative log likelihood of the MLP is given by the negative
        # log likelihood of the output of the model, computed in the
        # logistic regression layer
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

    def errors(self, y):
        """Return a float representing the number of errors in the minibatch
        over the total number of examples of the minibatch ; zero one
        loss over the size of the minibatch

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label
        """
        # check if y has same dimension of y_pred
        if y.ndim != self.y_pred.ndim:
            raise TypeError('y should have the same shape as self.y_pred',
                            ('y', y.type, 'y_pred', self.y_pred.type))
            # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()

    def _get_cost_update(self, lr=0.01):

        # L1 norm ; one regularization option is to enforce L1 norm to
        # be small
        self.L1 = abs(self.hidden_layer.W).sum() \
                  + abs(self.softmax_layer.W).sum()

        # square of L2 norm ; one regularization option is to enforce
        # square of L2 norm to be small
        self.L2_sqr = (self.hidden_layer.W ** 2).sum() \
                      + (self.softmax_layer.W ** 2).sum()

        # the total cost
        cost = self._negative_log_likelihood(self.y)

        # get individual update
        gparams = []
        for param in self.params:
            gparam = T.grad(cost, param)
            gparams.append(gparam)

        from collections import OrderedDict

        updates = OrderedDict()
        for param, gparam in zip(self.params, gparams):
            updates[param] = param - gparam * T.cast(lr, dtype=theano.config.floatX)

        return cost, updates

    def get_train_fn(self,
                     dataX,
                     dataY=None,
                     batch_size=1):
        """
        dataX: theano shared data

        dataY: theano shared label
        """
        learning_rate = T.scalar('lr')

        cost, updates = self._get_cost_update(learning_rate)

        idx = T.lscalar('index')

        fn = theano.function(inputs=[idx, theano.Param(learning_rate, default=0.01)],
                             outputs=cost,
                             updates=updates,
                             givens={
                                 self.x: dataX[idx * batch_size:(idx + 1) * batch_size],
                                 self.y: dataY[idx * batch_size:(idx + 1) * batch_size]})
        return fn

    def get_prediction(self, testX):
        # (test_set_x, test_set_y) = datasets['test']
        prediction = theano.function([], theano.Out(self.p_y_given_x, borrow=True),
                                       givens={self.x: testX})
        # don't return prediction()
        # since each time we need a prediction with different testset
        return prediction()

    def get_error_rate(self, testX, testY):
        if not isinstance(testX, theano.sandbox.cuda.var.CudaNdarraySharedVariable):
            testX = toSharedX(testX)
        mat = self.get_prediction(testX)
        y_pred = numpy.argmax(mat, axis=1)

        return numpy.mean(y_pred != testY)


def begin(data=None,
          learning_rate=0.01,
          n_epochs=100,
          n_hidden=20,
          batch_size=20):
    """
    1) load dataset
    """

    Tp = TerminalPrinter(debug=True, verbose=False)
    """
    1) load data
    """
    train_x, train_y = data['train']

    num_of_dims = train_x.shape[1]
    num_of_classes = max(train_y) - min(train_y) + 1
    n_train_batches = train_x.shape[0] / batch_size

    train_x = toSharedX(train_x, borrow=True)
    train_y = T.cast(toSharedX(train_y, borrow=True), 'int32')

    """
    2) build model
    """
    Tp.Print('... building symbolic model')
    numpy_rng = numpy.random.RandomState(123)

    classifier = MLP(rng=numpy_rng,
                     n_hidden=n_hidden,
                     n_in=num_of_dims,
                     n_out=num_of_classes)

    """
    3) compile training function
    """
    Tp.Print('... linking symbolic model')

    train_fn = classifier.get_train_fn(train_x, train_y, batch_size)

    """
    4) begin training
    """
    Tp.Print('... training the model')
    import time

    start_time = time.clock()
    for epoch in xrange(n_epochs):
        for batch_index in xrange(n_train_batches):
            err_rate = train_fn(index=batch_index, lr=learning_rate)
            Tp.Print(err_rate)

    end_time = time.clock()
    Tp.Print('total time is: ' + str((end_time - start_time) / 60.))

    """
    5) begin testing
    """
    Tp.Print('... testing the model')
    test_x, test_y = data['test']

    return classifier.get_error_rate(test_x, test_y)


def kfold_train(learning_rate=1,
                n_epochs=100,
                batch_size=20,
                num_folds=5,
                n_hidden=20,
                normalize_idx=0):
    """
    kfold_begin() training on 10 fold cross validation
    """

    """
    1) load dataset
    """
    from sklearn.datasets import load_iris

    dataset = load_iris()

    """
    2) split by KFold (in a ratio of NumOfFolds:1)
    """
    from xylearn.utils.data_util import get_train_test
    from xylearn.utils.data_normalization import rescale, standardize, whiten

    norm_factory = [rescale, standardize, whiten]

    mean_err = []

    for i in range(num_folds):
        data_i = get_train_test(norm_factory[normalize_idx](dataset.data), dataset.target, fold_id=i,
                                num_fold=num_folds, useGPU=0)

        mean_err.append(begin(data_i,
                              n_epochs=n_epochs,
                              n_hidden=n_hidden,
                              learning_rate=learning_rate,
                              batch_size=batch_size))

    return numpy.mean(mean_err)


if __name__ == '__main__':
    print kfold_train(learning_rate=1.5,
                      n_epochs=100,
                      batch_size=20)
