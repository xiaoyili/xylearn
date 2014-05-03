__author__ = 'eric'

import time
import os

import numpy
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
from xylearn.utils.theano_util import toSharedX


# for debugging
from xylearn.visualizer.terminal_printer import TerminalPrinter


class RBM(object):
    """Restricted Boltzmann Machine (RBM)  """

    def __init__(self, input=None, n_visible=784, n_hidden=500, \
                 W=None, hbias=None, vbias=None, numpy_rng=None, aFunc=T.nnet.sigmoid,
                 theano_rng=None, isPCD=0):
        """
        RBM constructor. Defines the parameters of the model along with
        basic operations for inferring hidden from visible (and vice-versa),
        as well as for performing CD updates.

        :param input: None for standalone RBMs or symbolic variable if RBM is
        part of a larger graph.

        :param n_visible: number of visible units

        :param n_hidden: number of hidden units

        :param W: None for standalone RBMs or symbolic variable pointing to a
        shared weight matrix in case RBM is part of a DBN network; in a DBN,
        the weights are shared between RBMs and layers of a MLP

        :param hbias: None for standalone RBMs or symbolic variable pointing
        to a shared hidden units bias vector in case RBM is part of a
        different network

        :param vbias: None for standalone RBMs or a symbolic variable
        pointing to a shared visible units bias
        """
        self.isPCD = isPCD
        self.aFunc = aFunc
        self.n_visible = n_visible
        self.n_hidden = n_hidden

        if numpy_rng is None:
            # create a number generator
            numpy_rng = numpy.random.RandomState(1234)

        if theano_rng is None:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))

        if W is None:
            # W is initialized with `initial_W` which is uniformely
            # sampled from -4*sqrt(6./(n_visible+n_hidden)) and
            # 4*sqrt(6./(n_hidden+n_visible)) the output of uniform if
            # converted using asarray to dtype theano.config.floatX so
            # that the code is runable on GPU
            initial_W = numpy.asarray(numpy_rng.uniform(
                low=-4 * numpy.sqrt(6. / (n_hidden + n_visible)) * 0.001,
                high=4 * numpy.sqrt(6. / (n_hidden + n_visible)) * 0.001,
                size=(n_visible, n_hidden)),
                                      dtype=theano.config.floatX)
            # theano shared variables for weights and biases
            W = theano.shared(value=initial_W, name='W', borrow=True)

        if hbias is None:
            # create shared variable for hidden units bias
            hbias = theano.shared(value=numpy.zeros(n_hidden,
                                                    dtype=theano.config.floatX),
                                  name='hbias', borrow=True)

        if vbias is None:
            # create shared variable for visible units bias
            vbias = theano.shared(value=numpy.zeros(n_visible,
                                                    dtype=theano.config.floatX),
                                  name='vbias', borrow=True)

        # initialize input layer for standalone RBM or layer0 of DBN
        self.x = input
        if not input:
            self.x = T.matrix('input')
        else:
            self.x = input

        self.W = W
        self.hbias = hbias
        self.vbias = vbias
        self.theano_rng = theano_rng
        # **** WARNING: It is not a good idea to put things in this list
        # other than shared variables created in this function.
        self.params = [self.W, self.hbias, self.vbias]


    def free_energy(self, v_sample):
        ''' Function to compute the free energy '''
        wx_b = T.dot(v_sample, self.W) + self.hbias
        vbias_term = T.dot(v_sample, self.vbias)
        hidden_term = T.sum(T.log(1 + T.exp(wx_b)), axis=1)
        return -hidden_term - vbias_term

    def propup(self, vis):
        '''This function propagates the visible units activation upwards to
        the hidden units

        Note that we return also the pre-sigmoid activation of the
        layer. As it will turn out later, due to how Theano deals with
        optimizations, this symbolic variable will be needed to write
        down a more stable computational graph (see details in the
        reconstruction cost function)

        '''

        pre_sigmoid_activation = T.dot(vis, self.W) + self.hbias
        return [pre_sigmoid_activation, self.aFunc(pre_sigmoid_activation)]

    def propdown(self, hid):
        '''This function propagates the hidden units activation downwards to
        the visible units

        Note that we return also the pre_sigmoid_activation of the
        layer. As it will turn out later, due to how Theano deals with
        optimizations, this symbolic variable will be needed to write
        down a more stable computational graph (see details in the
        reconstruction cost function)

        '''
        pre_sigmoid_activation = T.dot(hid, self.W.T) + self.vbias
        return [pre_sigmoid_activation, self.aFunc(pre_sigmoid_activation)]

    def _sample_h_given_v(self, v0_sample):
        ''' This function infers state of hidden units given visible units '''
        # compute the activation of the hidden units given a sample of
        # the visibles
        pre_sigmoid_h1, h1_mean = self.propup(v0_sample)
        # get a sample of the hiddens given their activation
        # Note that theano_rng.binomial returns a symbolic sample of dtype
        # int64 by default. If we want to keep our computations in floatX
        # for the GPU we need to specify to return the dtype floatX
        h1_sample = self.theano_rng.binomial(size=h1_mean.shape,
                                             n=1, p=h1_mean,
                                             dtype=theano.config.floatX)
        return [pre_sigmoid_h1, h1_mean, h1_sample]

    def _sample_v_given_h(self, h0_sample):
        ''' This function infers state of visible units given hidden units '''
        # compute the activation of the visible given the hidden sample
        pre_sigmoid_v1, v1_mean = self.propdown(h0_sample)
        # get a sample of the visible given their activation
        # Note that theano_rng.binomial returns a symbolic sample of dtype
        # int64 by default. If we want to keep our computations in floatX
        # for the GPU we need to specify to return the dtype floatX
        v1_sample = self.theano_rng.binomial(size=v1_mean.shape,
                                             n=1, p=v1_mean,
                                             dtype=theano.config.floatX)

        return [pre_sigmoid_v1, v1_mean, v1_sample]

    def _gibbs_hvh(self, h0_sample):
        ''' This function implements one step of Gibbs sampling,
            starting from the hidden state'''
        pre_sigmoid_v1, v1_mean, v1_sample = self._sample_v_given_h(h0_sample)
        pre_sigmoid_h1, h1_mean, h1_sample = self._sample_h_given_v(v1_sample)
        return [pre_sigmoid_v1, v1_mean, v1_sample,
                pre_sigmoid_h1, h1_mean, h1_sample]

    def _gibbs_vhv(self, v0_sample):
        ''' This function implements one step of Gibbs sampling,
            starting from the visible state'''
        pre_sigmoid_h1, h1_mean, h1_sample = self._sample_h_given_v(v0_sample)
        pre_sigmoid_v1, v1_mean, v1_sample = self._sample_v_given_h(h1_sample)
        return [pre_sigmoid_h1, h1_mean, h1_sample,
                pre_sigmoid_v1, v1_mean, v1_sample]

    def _get_cost_update(self, lr=0.1, persistent=None, k=1):
        """This functions implements one step of CD-k or PCD-k

        :param lr: learning rate used to train the RBM

        :param persistent: None for CD. For PCD, shared variable
            containing old state of Gibbs chain. This must be a shared
            variable of size (batch size, number of hidden units).

        :param k: number of Gibbs steps to do in CD-k/PCD-k

        Returns a proxy for the cost and the updates dictionary. The
        dictionary contains the update rules for weights and biases but
        also an update of the shared variable used to store the persistent
        chain, if one is used.

        """
        pre_sigmoid_ph, ph_mean, ph_sample = self._sample_h_given_v(self.x)

        # compute positive phase

        # decide how to initialize persistent chain:
        # for CD, we use the newly generate hidden sample
        # for PCD, we initialize from the old state of the chain
        if persistent is None:
            chain_start = ph_sample
        else:
            chain_start = persistent

        # perform actual negative phase
        # in order to implement CD-k/PCD-k we need to scan over the
        # function that implements one gibbs step k times.
        # Read Theano tutorial on scan for more information :
        # http://deeplearning.net/software/theano/library/scan.html
        # the scan will return the entire Gibbs chain
        [pre_sigmoid_nvs, nv_means, nv_samples,
         pre_sigmoid_nhs, nh_means, nh_samples], updates = \
            theano.scan(self._gibbs_hvh,
                        # the None are place holders, saying that
                        # chain_start is the initial state corresponding to the
                        # 6th output
                        outputs_info=[None, None, None, None, None, chain_start],
                        n_steps=k)

        # determine gradients on RBM parameters
        # note that we only need the sample at the end of the chain
        chain_end = nv_samples[-1]

        # Contrastive Loss, different from AE(cross entropy loss)
        cost = T.mean(self.free_energy(self.x)) - T.mean(
            self.free_energy(chain_end))
        # We must not compute the gradient through the gibbs sampling
        gparams = T.grad(cost, self.params, consider_constant=[chain_end])

        # constructs the update dictionary
        for gparam, param in zip(gparams, self.params):
            # make sure that the learning rate is of the right dtype
            updates[param] = param - gparam * T.cast(lr,
                                                     dtype=theano.config.floatX)
        if persistent:
            # Note that this works only if persistent is a shared variable
            updates[persistent] = nh_samples[-1]
            # pseudo-likelihood is a better proxy for PCD
            monitoring_cost = self._get_pseudo_likelihood_cost(updates)
        else:
            # reconstruction cross-entropy is a better proxy for CD
            monitoring_cost = self._get_reconstruction_cost(pre_sigmoid_nvs[-1])

        return monitoring_cost, updates

    def _get_pseudo_likelihood_cost(self, updates):
        """Stochastic approximation to the pseudo-likelihood"""

        # index of bit i in expression p(x_i | x_{\i})
        bit_i_idx = theano.shared(value=0, name='bit_i_idx')

        # binarize the input image by rounding to nearest integer
        xi = T.round(self.x)

        # calculate free energy for the given bit configuration
        fe_xi = self.free_energy(xi)

        # flip bit x_i of matrix xi and preserve all other bits x_{\i}
        # Equivalent to xi[:,bit_i_idx] = 1-xi[:, bit_i_idx], but assigns
        # the result to xi_flip, instead of working in place on xi.
        xi_flip = T.set_subtensor(xi[:, bit_i_idx], 1 - xi[:, bit_i_idx])

        # calculate free energy with bit flipped
        fe_xi_flip = self.free_energy(xi_flip)

        # equivalent to e^(-FE(x_i)) / (e^(-FE(x_i)) + e^(-FE(x_{\i})))
        cost = T.mean(self.n_visible * T.log(self.aFunc(fe_xi_flip -
                                                        fe_xi)))

        # increment bit_i_idx % number as part of updates
        updates[bit_i_idx] = (bit_i_idx + 1) % self.n_visible

        return cost

    def _get_reconstruction_cost(self, pre_sigmoid_nv):
        """Approximation to the reconstruction error

        Note that this function requires the pre-sigmoid activation as
        input.  To understand why this is so you need to understand a
        bit about how Theano works. Whenever you compile a Theano
        function, the computational graph that you pass as input gets
        optimized for speed and stability.  This is done by changing
        several parts of the subgraphs with others.  One such
        optimization expresses terms of the form log(sigmoid(x)) in
        terms of softplus.  We need this optimization for the
        cross-entropy since sigmoid of numbers larger than 30. (or
        even less then that) turn to 1. and numbers smaller than
        -30. turn to 0 which in terms will force theano to compute
        log(0) and therefore we will get either -inf or NaN as
        cost. If the value is expressed in terms of softplus we do not
        get this undesirable behaviour. This optimization usually
        works fine, but here we have a special case. The sigmoid is
        applied inside the scan op, while the log is
        outside. Therefore Theano will only see log(scan(..)) instead
        of log(sigmoid(..)) and will not apply the wanted
        optimization. We can not go and replace the sigmoid in scan
        with something else also, because this only needs to be done
        on the last step. Therefore the easiest and more efficient way
        is to get also the pre-sigmoid activation as an output of
        scan, and apply both the log and sigmoid outside scan such
        that Theano can catch and optimize the expression.

        """

        cross_entropy = T.mean(
            T.sum(self.x * T.log(self.aFunc(pre_sigmoid_nv)) +
                  (1 - self.x) * T.log(1 - self.aFunc(pre_sigmoid_nv)),
                  axis=1))

        return cross_entropy

    def get_train_fn(self, dataX, batch_size=1, k=1):
        """
        dataX: theano shared data

        dataY: theano shared label
        """

        if self.isPCD:
            persistent_chain = theano.shared(numpy.zeros((batch_size, self.n_hidden),
                                                         dtype=theano.config.floatX),
                                             borrow=True)
        else:
            persistent_chain = None

        learning_rate = T.scalar('lr')
        cost, updates = self._get_cost_update(learning_rate, persistent=persistent_chain, k=k)

        index = T.lscalar('index')

        fn = theano.function(inputs=[index, theano.Param(learning_rate, default=0.01)],
                             outputs=cost,
                             updates=updates,
                             givens={self.x: dataX[index * batch_size:(index + 1) * batch_size]},
                             name='train_rbm')
        return fn

    def get_sampling_fn(self, dataX, test_idx=1, n_chains=20, k=1):
        """
        Sampling from a trained RBM.
        After the model was trained, we can use dataX[test_idx: test_idx + n_chains]
        to initialize a persistent sampling chain. Let it run for k steps, and return
        the reconstructed result. Each sample will run on a individual chain.

        :param dataX: theano shared matrix

        :param test_idx: chose the one that you like

        :param n_chains: run n chains to reconstruct more samples

        :param k: number of gibbs round (up and down).

        if we just want to reconstruct the data. we can call like the following:

           fn = get_sampling_fn(dataX,
                                test_idx=0,
                                n_chains=dataX.get_value(borrow=True).shape[0],
                                k=1)

        """
        if n_chains >= dataX.get_value(borrow=True).shape[0]:
            n_chains = dataX.get_value(borrow=True).shape[0]

        persistent_vis_chain = theano.shared(numpy.asarray(
            dataX.get_value(borrow=True)[test_idx:test_idx + n_chains],
            dtype=theano.config.floatX))

        [presig_hids, hid_mfs, hid_samples, presig_vis,
         vis_mfs, vis_samples], updates = \
            theano.scan(self._gibbs_vhv,
                        outputs_info=[None, None, None, None,
                                      None, persistent_vis_chain],
                        n_steps=k)

        # add to updates the shared variable that takes care of our persistent
        # chain :.
        updates.update({persistent_vis_chain: vis_samples[-1]})
        # construct the function that implements our persistent chain.
        # we generate the "mean field" activations for plotting and the actual
        # samples for reinitializing the state of our persistent chain
        sample_fn = theano.function([], [vis_mfs[-1], vis_samples[-1]],
                                      updates=updates,
                                      givens={self.x: persistent_vis_chain},
                                      name='sample_fn')

        return sample_fn

    def get_reconstruction_error(self, testX):

        [pre_sigmoid_h1, h1_mean] = self.propup(testX)
        [pre_sigmoid_v1, v1_mean] = self.propdown(h1_mean)

        error = theano.function([], theano.Out(self._get_reconstruction_cost(pre_sigmoid_v1),
                                               borrow=True),
                                  givens={self.x: testX},
                                  name='rbm_recon_error')

        return error()

    def reconstruct(self, testX, n_rounds=1, showSample=0):

        fn = self.get_sampling_fn(testX,
                                  test_idx=0,
                                  n_chains=testX.get_value(borrow=True).shape[0],
                                  k=n_rounds)
        vis_mf, vis_sample = fn()
        if showSample:
            return vis_sample
        else:
            return vis_mf

    def project(self, dataX, hidSample=0):
        """
        project dataX into hidden space. In other words, when rbm was trained,
        we can get new representation
        hidSample:  1 means sample out
                    0 means the prob
        """
        [pre_sig, h1_mean, h1_sample] = self._sample_h_given_v(dataX)
        if hidSample:
            fn = theano.function([], theano.Out(h1_sample, borrow=True), name='project')
        else:
            fn = theano.function([], theano.Out(h1_mean, borrow=True), name='project')

        return fn()


def test_rbm_mnist(learning_rate=0.01, training_epochs=10, batch_size=20,
                   n_chains=30, n_samples=5, output_folder=None,
                   n_hidden=500, isPCD=0):
    """
    This is demonstrated on MNIST.

    :param learning_rate: learning rate used for training the RBM

    :param training_epochs: number of epochs used for training

    :param dataset: path the the pickled dataset

    :param batch_size: size of a batch used to train the RBM

    :param n_chains: number of parallel Gibbs chains to be used for sampling

    :param n_samples: number of samples to plot for each chain

    e.g.
        test_rbm_mnist(output_folder='/home/eric/Desktop/rbm_plots')

    """

    assert output_folder is not None

    #################################
    #     Data Constructing         #
    #################################

    from sklearn.datasets import fetch_mldata

    mnist = fetch_mldata('MNIST original')

    from xylearn.utils.data_util import get_train_test
    from xylearn.utils.data_normalization import rescale

    data = get_train_test(rescale(mnist.data), mnist.target, useGPU=1, shuffle=True)

    train_x, train_y = data['train']
    n_vis = train_x.get_value(borrow=True).shape[1]
    n_train_batches = train_x.get_value(borrow=True).shape[0] / batch_size


    # construct the RBM class
    rbm = RBM(n_visible=n_vis, n_hidden=n_hidden, isPCD=isPCD)
    train_fn = rbm.get_train_fn(train_x, batch_size)


    #################################
    #     Training the RBM          #
    #################################
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)
    os.chdir(output_folder)

    plotting_time = 0.
    start_time = time.clock()
    import PIL.Image
    from visualizer import tile_raster_images

    # go through training epochs
    for epoch in xrange(training_epochs):

        # go through the training set
        mean_cost = []
        for batch_index in xrange(n_train_batches):
            # for each batch, we extract the gibbs chain
            new_cost = train_fn(index=batch_index, lr=learning_rate)
            mean_cost += [new_cost]

        print 'Training epoch %d, cost is ' % epoch, numpy.mean(mean_cost)

        # W shape is [784 500]
        # Plot filters after each training epoch
        plotting_start = time.clock()
        # Construct image from the weight matrix
        image = PIL.Image.fromarray(tile_raster_images(
            X=rbm.W.get_value(borrow=True).T,
            img_shape=(28, 28), tile_shape=(10, 10),
            tile_spacing=(1, 1)))
        image.save('filters_at_epoch_%i.png' % epoch)
        plotting_stop = time.clock()
        plotting_time += (plotting_stop - plotting_start)

    end_time = time.clock()
    pretraining_time = (end_time - start_time) - plotting_time
    print ('Training took %f minutes' % (pretraining_time / 60.))


    #################################
    #     Sampling from the RBM     #
    #################################

    test_idx = 1

    test_x, test_y = data['test']
    sample_fn = rbm.get_sampling_fn(test_x, test_idx, n_chains)

    print '... begin sampling'
    # plot initial image first
    orig_img = test_x.get_value(borrow=True)[test_idx:test_idx + 1]
    image = PIL.Image.fromarray(tile_raster_images(
        X=orig_img,
        img_shape=(28, 28), tile_shape=(1, 1),
        tile_spacing=(1, 1)))
    image.save('orig_img.png')

    # create a space to store the image for plotting ( we need to leave
    # room for the tile_spacing as well)
    image_data = numpy.zeros((29 * n_samples + 1, 29 * n_chains - 1),
                             dtype='uint8')
    for idx in xrange(n_samples):
        # generate `plot_every` intermediate samples that we discard,
        # because successive samples in the chain are too correlated
        vis_mf, vis_sample = sample_fn()
        print ' ... plotting sample ', idx
        image_data[29 * idx:29 * idx + 28, :] = tile_raster_images(
            X=vis_mf,
            img_shape=(28, 28),
            tile_shape=(1, n_chains),
            tile_spacing=(1, 1))
        # construct image

    image = PIL.Image.fromarray(image_data)
    image.save('samples.png')
    os.chdir('../')


def begin(data=None,
          learning_rate=0.13,
          n_epochs=100,
          n_hidden=50,
          batch_size=600):
    """
    1) load dataset
    """

    Tp = TerminalPrinter(debug=False, verbose=False)
    """
    1) load data
    """
    train_x, train_y = data['train']
    num_of_dims = train_x.shape[1]
    num_of_classes = max(train_y) - min(train_y) + 1
    n_train_batches = train_x.shape[0] / batch_size

    train_x = toSharedX(train_x, borrow=True)

    """
    2) build model
    """
    Tp.Print('... building & linking symbolic model')
    rbm = RBM(n_visible=num_of_dims, n_hidden=n_hidden)
    """
    3) compiling
    """
    train_fn = rbm.get_train_fn(train_x, batch_size)

    """
    4) training
    """
    Tp.Print('... training the model')
    import time

    start_time = time.clock()
    for epoch in xrange(n_epochs):
        for batch_index in xrange(n_train_batches):
            err_rate = train_fn(index=batch_index, lr=learning_rate)
            Tp.interval_print(err_rate)

    end_time = time.clock()
    Tp.Print('total time is: ' + str((end_time - start_time) / 60.))

    """
    5) testing
    """
    Tp.Print('... testing the model')
    test_x, test_y = data['test']
    test_x = toSharedX(test_x, borrow=True)

    return rbm.get_reconstruction_error(test_x)


def kfold_train(learning_rate=0.01,
                n_epochs=100,
                n_hidden=50,
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
                              n_hidden=n_hidden,
                              batch_size=batch_size))

    return numpy.mean(mean_err)


if __name__ == '__main__':
    test_rbm_mnist(output_folder='/Volumes/HDD750/home/TEMP/rbm_plots',
                   learning_rate=0.01, training_epochs=10, batch_size=20,
                   n_chains=30, n_samples=5,
                   n_hidden=500, isPCD=1)