__author__ = 'eric'


import numpy
import theano
import theano.tensor as T
from collections import OrderedDict
from theano.tensor.shared_randomstreams import RandomStreams
from xylearn.utils.theano_util import shared_normal


class HmmLayer(object):
    """
    A modified HMM. Different from traditional hmm, we don't find the
    hidden state sequence, in other word, we don't need decode step.

    All what we are doing is to maximize the data likelihood given parameter:
        A: transition matrix (between hidden states)
        B: emission matrix (between hidden and visible)
    """

    def __init__(self, theano_rng=None, x_0=None, x_1=None, n_hidden=10, n_vis=None):

        if x_0:
            self.x = x_0
        else:
            self.x = T.matrix('x_0')

        if x_1:
            self.x_1 = x_1
        else:
            self.x_1 = T.matrix('x_1')

        # make sure the experiment is repeatable
        if theano_rng:
            self.t_rng = theano_rng
        else:
            rng = numpy.random.RandomState(1234)
            self.t_rng = RandomStreams(rng.randint(2 ** 30))

        # transitional
        self.A = shared_normal(n_hidden, n_hidden, theano_rng=self.t_rng)
        # emission
        self.B = shared_normal(n_hidden, n_vis, theano_rng=self.t_rng)

        # visible -> hidden (posterior)
        self.state = T.dot(self.x, self.B.T)

        # hidden -> visible (likelihood)
        self.emission = T.dot(T.dot(self.state, self.A), self.B)

        self.params = [self.A, self.B]

    def _get_cost_update(self, lr=0.1):

        cost = T.sqrt(((self.x_1 - self.emission) ** 2).sum()) / self.x.shape[0]

        # weight_decay = T.sum(self.B ** 2)
        # transition_speed = T.grad(cost, self.A)

        self.cost = cost
        gparams = T.grad(cost, wrt=self.params)

        updates = OrderedDict()
        for gparam, param in zip(gparams, self.params):
            # make sure that the learning rate is of the right dtype
            updates[param] = param - gparam * T.cast(lr,
                                                     dtype=theano.config.floatX)

        return cost, updates

    def get_train_fn(self):
        """
        dataX: theano shared data

        dataY: theano shared label
        """

        learning_rate = T.scalar('lr')
        cost, updates = self._get_cost_update(learning_rate)

        fn = theano.function([self.x, self.x_1, theano.Param(learning_rate, default=0.01)], cost,
                             updates=updates,
                             name='train_hmm')
        return fn

    def get_hidden_state(self, x_0):
        fn = theano.function([],
                               theano.Out(self.state, borrow=True),
                               givens={self.x: x_0})
        return fn()

    def predict(self, x_0):
        fn = theano.function([],
                               theano.Out(self.emission, borrow=True),
                               givens={self.x: x_0})

        return fn()


if __name__ == "__main__":

    rng = numpy.random.RandomState(1234)
    t_rng = RandomStreams(rng.randint(2 ** 30))

    # synthetic data with increasing sigma
    DataDim = 2
    data = []
    for i in xrange(10):
        # numpy.random.normal(0.0, i + 1, (1000, 2)
        # unstable, experiment not repeatable
        tmp = t_rng.normal((1000, DataDim), std=i + 1).eval().astype(theano.config.floatX)
        data.append(tmp / numpy.sum(tmp, axis=0))

    training_epoch = 50
    learning_rate = 0.1
    batch_size = 100
    n_train_batches = data[0].shape[0] / batch_size

    # init model
    hmm = HmmLayer(n_vis=DataDim, n_hidden=5, theano_rng=t_rng)

    # build fns
    train_fn = hmm.get_train_fn()

    # go through training epochs
    for epoch in xrange(training_epoch):
        mean_cost = []
        for i in xrange(8):
            # go through the training set
            for batch_index in xrange(n_train_batches):
                # for each batch, we extract the gibbs chain
                new_cost = train_fn(data[i][batch_index * batch_size:(batch_index + 1) * batch_size],
                                    data[i + 1][batch_index * batch_size:(batch_index + 1) * batch_size],
                                    lr=learning_rate)
                mean_cost += [new_cost]

        print 'Training epoch %d, cost is ' % epoch, numpy.mean(mean_cost)


    print hmm.A.get_value(borrow=True)
    print "================"
    print hmm.B.get_value(borrow=True)

    # see the output

    pred = hmm.predict(data[8])

    print numpy.sum(data[9] - pred)



    ###################
    # begin plotting  #
    ###################
    from matplotlib import pylab as plt

    plt.scatter(data[9][:, 0], data[9][:, 1])
    plt.show()

    plt.scatter(pred[:, 0], pred[:, 1])
    plt.show()
