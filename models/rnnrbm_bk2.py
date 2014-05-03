# Author: Nicolas Boulanger-Lewandowski
# University of Montreal (2012)
# RNN-RBM deep learning tutorial
# More information at http://deeplearning.net/tutorial/rnnrbm.html

import numpy
from xylearn.utils.theano_util import shared_normal, shared_zeros
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
from theano import config
config.warn


#Don't use a python long as this don't work on 32 bits computers.
numpy.random.seed(0xbeef)
rng = RandomStreams(seed=numpy.random.randint(1 << 30))
theano.config.warn.subtensor_merge_bug = False


def build_rbm(v, W, bv, bh, k):
    '''Construct a k-step Gibbs chain starting at v for an RBM.

v : Theano vector or matrix
  If a matrix, multiple chains will be run in parallel (batch).
W : Theano matrix
  Weight matrix of the RBM.
bv : Theano vector
  Visible bias vector of the RBM.
bh : Theano vector
  Hidden bias vector of the RBM.
k : scalar or Theano scalar
  Length of the Gibbs chain.

Return a (v_sample, cost, monitor, updates) tuple:

v_sample : Theano vector or matrix with the same shape as `v`
  Corresponds to the generated sample(s).
cost : Theano scalar
  Expression whose gradient with respect to W, bv, bh is the CD-k approximation
  to the log-likelihood of `v` (training example) under the RBM.
  The cost is averaged in the batch case.
monitor: Theano scalar
  Pseudo log-likelihood (also averaged in the batch case).
updates: dictionary of Theano variable -> Theano variable
  The `updates` object returned by scan.'''

    def gibbs_step(v):
        mean_h = T.nnet.sigmoid(T.dot(v, W) + bh)
        h = rng.binomial(size=mean_h.shape, n=1, p=mean_h,
                         dtype=theano.config.floatX)
        mean_v = T.nnet.sigmoid(T.dot(h, W.T) + bv)
        v = rng.binomial(size=mean_v.shape, n=1, p=mean_v,
                         dtype=theano.config.floatX)
        return mean_v, v

    chain, updates = theano.scan(lambda v: gibbs_step(v)[1], outputs_info=[v],
                                 n_steps=k)
    v_sample = chain[-1]

    mean_v = gibbs_step(v_sample)[0]
    monitor = T.xlogx.xlogy0(v, mean_v) + T.xlogx.xlogy0(1 - v, 1 - mean_v)
    monitor = monitor.sum() / v.shape[0]

    def free_energy(v):
        return -(v * bv).sum() - T.log(1 + T.exp(T.dot(v, W) + bh)).sum()

    cost = (free_energy(v) - free_energy(v_sample)) / v.shape[0]

    return v_sample, cost, monitor, updates


def build_rnnrbm(n_visible, n_hidden, n_hidden_recurrent):
    '''Construct a symbolic RNN-RBM and initialize parameters.

n_visible : integer
  Number of visible units.
n_hidden : integer
  Number of hidden units of the conditional RBMs.
n_hidden_recurrent : integer
  Number of hidden units of the RNN.

Return a (v, v_sample, cost, monitor, params, updates_train, v_t,
          updates_generate) tuple:

v : Theano matrix
  Symbolic variable holding an input sequence (used during training)
v_sample : Theano matrix
  Symbolic variable holding the negative particles for CD log-likelihood
  gradient estimation (used during training)
cost : Theano scalar
  Expression whose gradient (considering v_sample constant) corresponds to the
  LL gradient of the RNN-RBM (used during training)
monitor : Theano scalar
  Frame-level pseudo-likelihood (useful for monitoring during training)
params : tuple of Theano shared variables
  The parameters of the model to be optimized during training.
updates_train : dictionary of Theano variable -> Theano variable
  Update object that should be passed to theano.function when compiling the
  training function.
v_t : Theano matrix
  Symbolic variable holding a generated sequence (used during sampling)
updates_generate : dictionary of Theano variable -> Theano variable
  Update object that should be passed to theano.function when compiling the
  generation function.'''

    W = shared_normal(n_visible, n_hidden, 0.01)
    bv = shared_zeros(n_visible)
    bh = shared_zeros(n_hidden)
    Wuh = shared_normal(n_hidden_recurrent, n_hidden, 0.0001)
    Wuv = shared_normal(n_hidden_recurrent, n_visible, 0.0001)
    Wvu = shared_normal(n_visible, n_hidden_recurrent, 0.0001)
    Wuu = shared_normal(n_hidden_recurrent, n_hidden_recurrent, 0.0001)
    bu = shared_zeros(n_hidden_recurrent)

    params = W, bv, bh, Wuh, Wuv, Wvu, Wuu, bu  # learned parameters as shared
    # variables

    v = T.matrix()  # a training sequence
    u0 = T.zeros((n_hidden_recurrent,))  # initial value for the RNN hidden
    # units

    # If `v_t` is given, deterministic recurrence to compute the variable
    # biases bv_t, bh_t at each time step. If `v_t` is None, same recurrence
    # but with a separate Gibbs chain at each time step to sample (generate)
    # from the RNN-RBM. The resulting sample v_t is returned in order to be
    # passed down to the sequence history.
    def recurrence(v_t, u_tm1):
        bv_t = bv + T.dot(u_tm1, Wuv)
        bh_t = bh + T.dot(u_tm1, Wuh)
        generate = v_t is None
        if generate:
            v_t, _, _, updates = build_rbm(T.zeros((n_visible,)), W, bv_t,
                                           bh_t, k=25)
        u_t = T.tanh(bu + T.dot(v_t, Wvu) + T.dot(u_tm1, Wuu))
        return ([v_t, u_t], updates) if generate else [u_t, bv_t, bh_t]

    # For training, the deterministic recurrence is used to compute all the
    # {bv_t, bh_t, 1 <= t <= T} given v. Conditional RBMs can then be trained
    # in batches using those parameters.
    (u_t, bv_t, bh_t), updates_train = theano.scan(
        lambda v_t, u_tm1, *_: recurrence(v_t, u_tm1),
        sequences=v, outputs_info=[u0, None, None], non_sequences=params)

    v_sample, cost, monitor, updates_rbm = build_rbm(v, W, bv_t[:], bh_t[:],
                                                     k=15)
    updates_train.update(updates_rbm)

    # symbolic loop for sequence generation
    (v_t, u_t), updates_generate = theano.scan(
        lambda u_tm1, *_: recurrence(None, u_tm1),
        outputs_info=[None, u0], non_sequences=params, n_steps=250)

    return (v, v_sample, cost, monitor, params, updates_train, v_t,
            updates_generate)


class RnnRbm:
    '''
        Simple class to train an RNN-RBM from MIDI files and to generate sample
        sequences.
    '''

    def __init__(self, n_hidden=150, n_hidden_recurrent=100, lr=0.001,
                 n_vis=None):
        '''Constructs and compiles Theano functions for training and sequence
            generation.

            n_hidden : integer
                Number of hidden units of the conditional RBMs.
            n_hidden_recurrent : integer
                Number of hidden units of the RNN.
            lr : float
                Learning rate
            n_vis : integer
                Number of vis units
        '''

        assert n_vis is not None

        self.n_vis = n_vis
        (v, v_sample, cost, monitor, params, updates_train, v_t,
         updates_generate) = build_rnnrbm(n_vis, n_hidden,
                                          n_hidden_recurrent)

        gradient = T.grad(cost, params, consider_constant=[v_sample])
        updates_train.update(((p, p - lr * g) for p, g in zip(params,
                                                              gradient)))
        self.train_function = theano.function([v], monitor,
                                              updates=updates_train)
        self.generate_function = theano.function([], v_t,
                                                   updates=updates_generate)


    def train(self, dataset, batch_size=100, num_epochs=200):
        '''Train the RNN-RBM via stochastic gradient descent (SGD) using MIDI
    files converted to piano-rolls.

    files : list of strings
    List of MIDI files that will be loaded as piano-rolls for training.
    batch_size : integer
    Training sequences will be split into subsequences of at most this size
    before applying the SGD updates.
    num_epochs : integer
    Number of epochs (pass over the training set) performed. The user can
    safely interrupt training with Ctrl+C at any time.'''

        assert len(dataset) > 0, 'Training set is empty!'

        # print dataset

        for epoch in xrange(num_epochs):
            numpy.random.shuffle(dataset)
            costs = []

            for s, sequence in enumerate(dataset):

                # print s
                # print len(sequence)
                # write_to_mat(sequence, 'sample.mat')

                for i in xrange(0, len(sequence), batch_size):
                    cost = self.train_function(sequence[i:i + batch_size])
                    costs.append(cost)

            print 'Epoch %i/%i' % (epoch + 1, num_epochs),
            print numpy.mean(costs)


    def generate(self):
        '''
            Generate a sample sequence.

            filename : string
              A MIDI file will be created at this location.
            show : boolean
              If True, a piano-roll of the generated sequence will be shown.
        '''

        return self.generate_function()


def test_rnnrbm(batch_size=100, num_epochs=1):
    print '... initializing model'
    model = RnnRbm(n_vis=100)

    print '... feeding data'
    test = list()
    for i in xrange(10):
        test.append(numpy.random.random((10000, 100)).astype(theano.config.floatX))

    print '... begin training'
    model.train(test,
                batch_size=batch_size,
                num_epochs=num_epochs)

    return model


if __name__ == '__main__':
    model = test_rnnrbm()

    print '... generating samples'
    samples = model.generate()
    print samples
    # write_to_mat(samples, 'output.mat')