__author__ = 'eric'

from collections import OrderedDict

import numpy
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
from xylearn.utils.theano_util import shared_normal, toSharedX


class HmmLayer(object):
    """
    Xiaoyi's own HMM. Different from traditional hmm, we don't find the
    hidden state sequence, in other word, we don't need decode step.

    All what we are doing is to maximize the data likelihood given parameter:
        A: transition matrix (between hidden states)
        B: emission matrix (between hidden and visible)
    """

    def __init__(self, input=None, n_hidden=10, n_vis=None):

        if input is None:
            self.x = T.matrix('x_hmm')  # symbolic input
        else:
            self.x = input

        self.x_1 = T.zeros(self.x.shape, dtype=theano.config.floatX)

        self.B = shared_normal(n_hidden, n_vis)
        self.A = shared_normal(n_hidden, n_hidden)

        self.state = T.dot(self.x, self.B.T)

        self.emission = T.dot(T.dot(self.state, self.A), self.B)

        self.cost = T.sqrt(((self.x_1 - self.emission) ** 2).sum()) / self.x.shape[0]

        self.params = [self.A, self.B]

    def _get_cost_update(self, lr=0.1):

        cost = self.cost
        gparams = T.grad(cost, wrt=self.params)

        updates = OrderedDict()
        for gparam, param in zip(gparams, self.params):
            # make sure that the learning rate is of the right dtype
            updates[param] = param - gparam * T.cast(lr,
                                                     dtype=theano.config.floatX)

        return cost, updates

    def get_train_fn(self, dataX, dataX_1, batch_size=1):
        """
        dataX: theano shared data

        dataY: theano shared label
        """

        learning_rate = T.scalar('lr')
        cost, updates = self._get_cost_update(learning_rate)

        index = T.lscalar('index')

        obs = T.dmatrix('data')

        fn = theano.function(inputs=[index, theano.Param(learning_rate, default=0.01)],
                             outputs=cost,
                             updates=updates,
                             givens={self.x: dataX[index * batch_size:(index + 1) * batch_size],
                                     self.x_1: dataX_1[index * batch_size:(index + 1) * batch_size]},
                             name='train_hmm')
        return fn


if __name__ == "__main__":

    rng = numpy.random.RandomState(1234)
    t_rng = RandomStreams(rng.randint(2 ** 30))

    # synthetic data with increasing sigma
    np_data = []
    test = []
    for i in xrange(10):
        np_data.append(numpy.random.normal(0.0, i + 1, (1000, 2)))
        test.append(toSharedX(np_data[-1]))

    training_epoch = 100
    learning_rate = 0.1
    batch_size = 20
    n_train_batches = test[0].get_value(borrow=True).shape[0] / batch_size

    # init model
    hmm = HmmLayer(n_vis=2, n_hidden=3)

    # build fns
    train_fn = hmm.get_train_fn(test[0], test[1], batch_size)

    # go through training epochs
    for epoch in xrange(training_epoch):

        # go through the training set
        mean_cost = []
        for batch_index in xrange(n_train_batches):
            # for each batch, we extract the gibbs chain
            new_cost = train_fn(index=batch_index, lr=learning_rate)
            mean_cost += [new_cost]

        print 'Training epoch %d, cost is ' % epoch, numpy.mean(mean_cost)



    import matplotlib.pyplot as plt

    plt.scatter(np_data[0][:, 0], np_data[0][:, 1])
    plt.show()

    plt.scatter(np_data[2][:, 0], np_data[2][:, 1])
    plt.show()
    plt.scatter(np_data[4][:, 0], np_data[4][:, 1])
    plt.show()
    plt.scatter(np_data[6][:, 0], np_data[6][:, 1])
    plt.show()
    plt.scatter(np_data[9][:, 0], np_data[9][:, 1])
    plt.show()
