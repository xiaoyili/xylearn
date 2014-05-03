"""

.. math::
  P(Y=i|x, W,b) &= softmax_i(W x + b) \\
                &= \frac {e^{W_i x + b_i}} {\sum_j e^{W_j x + b_j}}


The output of the model or prediction is then done by taking the argmax of
the vector whose i'th element is P(Y=i|x).

.. math::

  y_{pred} = argmax_i P(Y=i|x,W,b)


References:

    - textbooks: "Pattern Recognition and Machine Learning" -
                 Christopher M. Bishop, section 4.3.2

"""
__docformat__ = 'restructedtext en'
__author__ = 'eric'

import numpy
import theano
import theano.tensor as T

from xylearn.utils.theano_util import toSharedX

# for debug
from xylearn.visualizer.terminal_printer import TerminalPrinter


class SoftmaxRegression(object):
    """Multi-class Logistic Regression Class

    The logistic regression is fully described by a weight matrix :math:`W`
    and bias vector :math:`b`. Classification is done by projecting data
    points onto a set of hyperplanes, the distance to which is used to
    determine a class membership probability.
    """

    def __init__(self, input=None, n_in=None, n_out=None, W=None, b=None):
        """ Initialize the parameters of the logistic regression

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
                      architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
                     which the datapoints lie

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
                      which the labels lie

        """
        if W is None:
            # initialize with 0 the weights W as a matrix of shape (n_in, n_out)
            W = theano.shared(value=numpy.zeros((n_in, n_out),
                                                dtype=theano.config.floatX),
                              name='W', borrow=True)
        if b is None:
            # initialize the baises b as a vector of n_out 0s
            b = theano.shared(value=numpy.zeros((n_out,),
                                                dtype=theano.config.floatX),
                              name='b', borrow=True)

        self.W = W
        self.b = b

        if input is None:
            self.x = T.matrix('x')  # symbolic input
        else:
            self.x = input

        self.y = T.ivector('y')  # symbolic target

        # compute vector of class-membership probabilities in symbolic form
        self.p_y_given_x = T.nnet.softmax(T.dot(self.x, self.W) + self.b)

        # compute prediction as class whose probability is maximal in
        # symbolic form
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)

        # parameters of the model
        self.params = [self.W, self.b]

    def _negative_log_likelihood(self, y):
        """Return the mean of the negative log-likelihood of the prediction
        of this model under a given target distribution.

        .. math::

            \frac{1}{|\mathcal{D}|} \mathcal{L} (\theta=\{W,b\}, \mathcal{D}) =
            \frac{1}{|\mathcal{D}|} \sum_{i=0}^{|\mathcal{D}|} \log(P(Y=y^{(i)}|x^{(i)}, W,b)) \\
                \ell (\theta=\{W,b\}, \mathcal{D})

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label

        Note: we use the mean instead of the sum so that
              the learning rate is less dependent on the batch size
        """
        # y.shape[0] is (symbolically) the number of rows in y, i.e.,
        # number of examples (call it n) in the minibatch
        #
        # T.arange(y.shape[0]) is a symbolic vector which will contain
        # [0,1,2,... n-1].
        #
        # T.log(self.p_y_given_x) is a matrix of
        # Log-Probabilities (call it LP) with one row per example and
        # one column per class
        #
        # LP[T.arange(y.shape[0]),y] is a vector
        # v containing [LP[0,y[0]], LP[1,y[1]], LP[2,y[2]], ...,
        # LP[n-1,y[n-1]]] and T.mean(LP[T.arange(y.shape[0]),y]) is
        # the mean (across minibatch examples) of the elements in v,
        # i.e., the mean log-likelihood across the minibatch.
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

    def errors(self, y):
        """
        Warning: DeprecationWarning will generated when numpy1.8+scipy 1.3.3

        Return a float representing the number of errors in the minibatch
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

            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()

    def _get_cost_update(self, lr=0.01):
        cost = self._negative_log_likelihood(self.y)
        g_W = T.grad(cost, self.W)
        g_b = T.grad(cost, self.b)

        from collections import OrderedDict

        updates = OrderedDict({self.W: self.W - g_W * T.cast(lr, dtype=theano.config.floatX),
                               self.b: self.b - g_b * T.cast(lr, dtype=theano.config.floatX)})

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

        index = T.lscalar('index')

        fn = theano.function(inputs=[index, theano.Param(learning_rate, default=0.01)],
                             outputs=cost,
                             updates=updates,
                             givens={
                                 self.x: dataX[index * batch_size:(index + 1) * batch_size],
                                 self.y: dataY[index * batch_size:(index + 1) * batch_size]})
        return fn

    def get_prediction(self, testX):
        # (test_set_x, test_set_y) = datasets['test']
        prediction = theano.function([], theano.Out(self.p_y_given_x, borrow=True),
                                       givens={self.x: testX})
        # don't return prediction()
        # since each time we need a prediction with different testset
        return prediction()

    def get_error_rate(self, testX, testY):

        testX = toSharedX(testX)
        mat = self.get_prediction(testX)
        y_pred = numpy.argmax(mat, axis=1)

        return numpy.mean(y_pred != testY)


def begin(data=None,
          learning_rate=0.13,
          n_epochs=100,
          batch_size=600):
    """
    1) load dataset
    """

    Tp = TerminalPrinter(debug=True, verbose=True)
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
    Tp.Print('... building & linking symbolic model')
    classifier = SoftmaxRegression(n_in=num_of_dims, n_out=num_of_classes)

    """
    3) compile training function
    """
    train_fn = classifier.get_train_fn(train_x, train_y, batch_size)

    """
    4) begin training
    """
    Tp.Print('... training the model')
    import time

    Tp.set_counter(n_epochs, stepsize=n_epochs / 10 + 1)
    start_time = time.clock()
    for epoch in xrange(n_epochs):
        for batch_index in xrange(n_train_batches):
            err_rate = train_fn(index=batch_index, lr=learning_rate)
            Tp.interval_print(err_rate)

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
                              learning_rate=learning_rate,
                              batch_size=batch_size))

    return numpy.mean(mean_err)


if __name__ == '__main__':
    print kfold_train(learning_rate=1.5,
                      n_epochs=100,
                      batch_size=20)
