__author__ = 'eric'

import os
import time

import numpy
import theano
import theano.tensor as T
from xylearn.utils.theano_util import toSharedX

from ae import AE


class AE_L1(AE):
    """
    Autoencoder with L1 norm
    """

    def _get_cost_update(self, corruption_level, learning_rate, beta):
        """ This function computes the cost and the updates for one trainng
        step of the AE """

        tilde_x = self.get_corrupted_input(self.x, corruption_level)

        z = self.propdown(self.propup(tilde_x))
        # note : we sum over the size of a datapoint; if we are using
        #        minibatches, L will be a vector, with one entry per
        #        example in minibatch
        L = - T.sum(self.x * T.log(z) + (1 - self.x) * T.log(1 - z), axis=1)
        # note : L is now a vector, where each element is the
        #        cross-entropy cost of the reconstruction of the
        #        corresponding example of the minibatch. We need to
        #        compute the average of all these to get the cost of
        #        the minibatch
        cost = T.mean(L) + T.cast(beta, dtype=theano.config.floatX) * T.sum(abs(self.W))

        # compute the gradients of the cost of the `AE` with respect
        # to its parameters
        gparams = T.grad(cost, self.params)
        # generate the list of updates
        updates = []
        for param, gparam in zip(self.params, gparams):
            updates.append((param, param - learning_rate * gparam))

        return cost, updates

    def get_train_fn(self, dataX, batch_size, corruption_level=0.0):
        # index to a [mini]batch
        index = T.lscalar('index')  # index to a minibatch
        learning_rate = T.scalar('lr')
        beta = T.scalar('beta')
        cost, updates = self._get_cost_update(corruption_level=corruption_level,
                                              learning_rate=learning_rate,
                                              beta=beta)

        fn = theano.function([index,
                              theano.Param(learning_rate, default=0.01),
                              theano.Param(beta, default=0.001)],
                             outputs=cost,
                             updates=updates,
                             givens={self.x: dataX[index * batch_size:(index + 1) * batch_size]},
                             name='train_ae_L1')

        return fn


class AE_SPARSE_L2(AE):
    """
    Autoencoder with L1 norm
    """

    def _get_cost_update(self, corruption_level, learning_rate, beta,
                         gamma=0.0001, s_constrain=0.05, ):
        """ This function computes the cost and the updates for one trainng
        step of the AE """

        tilde_x = self.get_corrupted_input(self.x, corruption_level)

        hid = self.propup(tilde_x)
        hidden_act = T.mean(hid, axis=1)
        sparse_constrain = T.cast(s_constrain, dtype=theano.config.floatX)
        sparsity_cost = T.sum(sparse_constrain * T.log(sparse_constrain / hidden_act) + (1 - sparse_constrain) * T.log(
            (1 - sparse_constrain) / (1 - hidden_act)))

        z = self.propdown(self.propup(tilde_x))
        L = - T.sum(self.x * T.log(z) + (1 - self.x) * T.log(1 - z), axis=1)

        weight_decay = T.sum(T.sqr(self.W))

        cost = T.mean(L) + T.cast(beta, dtype=theano.config.floatX) * sparsity_cost + \
               T.cast(gamma, dtype=theano.config.floatX) * weight_decay

        # compute the gradients of the cost of the `AE` with respect
        # to its parameters
        gparams = T.grad(cost, self.params)
        # generate the list of updates
        updates = []
        for param, gparam in zip(self.params, gparams):
            updates.append((param, param - learning_rate * gparam))

        return cost, updates


    def get_train_fn(self, dataX, batch_size, corruption_level=0.0):
        learning_rate = T.scalar('lr')
        Beta = T.scalar('beta')
        Gamma = T.scalar('gamma')
        Sparseness = T.scalar('sparseness')

        cost, updates = self._get_cost_update(learning_rate=learning_rate,
                                              beta=Beta,
                                              gamma=Gamma,
                                              s_constrain=Sparseness,
                                              corruption_level=corruption_level)

        index = T.lscalar('index')

        fn = theano.function(inputs=[index,
                                     theano.Param(learning_rate, default=0.01),
                                     theano.Param(Beta, default=0.1),
                                     theano.Param(Gamma, default=0.0001),
                                     theano.Param(Sparseness, default=0.05)],
                             outputs=cost,
                             updates=updates,
                             givens={self.x: dataX[index * batch_size:(index + 1) * batch_size]},
                             name='train_ae_S_L2')
        return fn


class AE_Orthogonal(AE):
    """
    Autoencoder with L1 norm
    """

    def _get_cost_update(self, corruption_level, learning_rate, beta):
        """ This function computes the cost and the updates for one trainng
        step of the AE """

        tilde_x = self.get_corrupted_input(self.x, corruption_level)

        z = self.propdown(self.propup(tilde_x))
        # note : we sum over the size of a datapoint; if we are using
        #        minibatches, L will be a vector, with one entry per
        #        example in minibatch
        L = - T.sum(self.x * T.log(z) + (1 - self.x) * T.log(1 - z), axis=1)
        # note : L is now a vector, where each element is the
        #        cross-entropy cost of the reconstruction of the
        #        corresponding example of the minibatch. We need to
        #        compute the average of all these to get the cost of
        #        the minibatch
        orthogonal_cost = T.sum(T.sqr(T.dot(self.W.T, self.W) - T.eye(self.n_hidden, self.n_hidden)))
        cost = T.mean(L) + T.cast(beta, dtype=theano.config.floatX) * orthogonal_cost

        # compute the gradients of the cost of the `AE` with respect
        # to its parameters
        gparams = T.grad(cost, self.params)
        # generate the list of updates
        updates = []
        for param, gparam in zip(self.params, gparams):
            updates.append((param, param - learning_rate * gparam))

        return cost, updates

    def get_train_fn(self, dataX, batch_size, corruption_level=0.0):
        # index to a [mini]batch
        index = T.lscalar('index')  # index to a minibatch
        learning_rate = T.scalar('lr')
        beta = T.scalar('beta')
        cost, updates = self._get_cost_update(corruption_level=corruption_level,
                                              learning_rate=learning_rate,
                                              beta=beta)

        fn = theano.function([index,
                              theano.Param(learning_rate, default=0.01),
                              theano.Param(beta, default=0.001)],
                             outputs=cost,
                             updates=updates,
                             givens={self.x: dataX[index * batch_size:(index + 1) * batch_size]},
                             name='train_ae_orthogonal')

        return fn


class AE_Linear(AE):
    def __init__(self, numpy_rng=None, theano_rng=None, input=None,
                 n_visible=784, n_hidden=500,
                 W=None, Wd=None, hbias=None, vbias=None, aFunc=T.nnet.sigmoid):
        """

        :type numpy_rng: numpy.random.RandomState
        :param numpy_rng: number random generator used to generate weights

        :type theano_rng: theano.tensor.shared_randomstreams.RandomStreams
        :param theano_rng: Theano random generator; if None is given one is
                     generated based on a seed drawn from `rng`

        :type input: theano.tensor.TensorType
        :param input: a symbolic description of the input or None for
                      standalone AE

        :type n_visible: int
        :param n_visible: number of visible units

        :type n_hidden: int
        :param n_hidden:  number of hidden units

        :type W: theano.tensor.TensorType
        :param W: Theano variable pointing to a set of weights that should be
                  shared belong the AE and another architecture; if AE should
                  be standalone set this to None

        :type hbias: theano.tensor.TensorType
        :param hbias: Theano variable pointing to a set of biases values (for
                     hidden units) that should be shared belong AE and another
                     architecture; if AE should be standalone set this to None

        :type vbias: theano.tensor.TensorType
        :param vbias: Theano variable pointing to a set of biases values (for
                     visible units) that should be shared belong AE and another
                     architecture; if AE should be standalone set this to None


        """

        if numpy_rng is None:
            # create a number generator
            numpy_rng = numpy.random.RandomState(1234)

        super(AE_Linear, self).__init__(numpy_rng=numpy_rng, theano_rng=theano_rng,
                                        input=input, n_visible=n_visible,
                                        n_hidden=n_hidden, W=W, hbias=hbias,
                                        vbias=vbias, aFunc=aFunc)

        if not Wd:
            initial_W = numpy.asarray(numpy_rng.uniform(
                low=-4 * numpy.sqrt(6. / (n_hidden + n_visible)),
                high=4 * numpy.sqrt(6. / (n_hidden + n_visible)),
                size=(n_hidden, n_visible)), dtype=theano.config.floatX)
            Wd = theano.shared(value=initial_W, name='Wd', borrow=True)

        self.Wd = Wd

        self.params = [self.W, self.b, self.vbias]

    def propup(self, input):
        return T.dot(input, self.W) + self.b

    def propdown(self, hid):
        return T.dot(hid, self.W.T) + self.vbias

    def _get_cost_update(self, corruption_level, learning_rate):
        """ This function computes the cost and the updates for one trainng
        step of the AE """

        z = self.propdown(self.propup(self.x))

        cost = T.sqrt(T.sum(T.sqr(self.x - z)) / self.x.shape[0])

        # compute the gradients of the cost of the `AE` with respect
        # to its parameters
        gparams = T.grad(cost, self.params)
        # generate the list of updates
        updates = []
        for param, gparam in zip(self.params, gparams):
            updates.append((param, param - learning_rate * gparam))

        return cost, updates


class AE_Poisson(AE):
    def __init__(self, numpy_rng=None, theano_rng=None, input=None,
                 n_visible=784, n_hidden=500,
                 W=None, Wd=None, hbias=None, vbias=None, aFunc=T.nnet.sigmoid):
        """

        :type numpy_rng: numpy.random.RandomState
        :param numpy_rng: number random generator used to generate weights

        :type theano_rng: theano.tensor.shared_randomstreams.RandomStreams
        :param theano_rng: Theano random generator; if None is given one is
                     generated based on a seed drawn from `rng`

        :type input: theano.tensor.TensorType
        :param input: a symbolic description of the input or None for
                      standalone AE

        :type n_visible: int
        :param n_visible: number of visible units

        :type n_hidden: int
        :param n_hidden:  number of hidden units

        :type W: theano.tensor.TensorType
        :param W: Theano variable pointing to a set of weights that should be
                  shared belong the AE and another architecture; if AE should
                  be standalone set this to None

        :type hbias: theano.tensor.TensorType
        :param hbias: Theano variable pointing to a set of biases values (for
                     hidden units) that should be shared belong AE and another
                     architecture; if AE should be standalone set this to None

        :type vbias: theano.tensor.TensorType
        :param vbias: Theano variable pointing to a set of biases values (for
                     visible units) that should be shared belong AE and another
                     architecture; if AE should be standalone set this to None


        """

        if numpy_rng is None:
            # create a number generator
            numpy_rng = numpy.random.RandomState(1234)

        super(AE_Poisson, self).__init__(numpy_rng=numpy_rng, theano_rng=theano_rng,
                                         input=input, n_visible=n_visible,
                                         n_hidden=n_hidden, W=W, hbias=hbias,
                                         vbias=vbias, aFunc=aFunc)

        if not Wd:
            initial_W = numpy.asarray(numpy_rng.uniform(
                low=-4 * numpy.sqrt(6. / (n_hidden + n_visible)),
                high=4 * numpy.sqrt(6. / (n_hidden + n_visible)),
                size=(n_hidden, n_visible)), dtype=theano.config.floatX)
            Wd = theano.shared(value=initial_W, name='Wd', borrow=True)

        self.Wd = Wd

        self.params = [self.W, self.b, self.vbias]

    def propup(self, input):
        return self.aFunc(T.dot(input, self.W) + self.b)

    def propdown(self, hid):
        return self.aFunc(T.dot(hid, self.W.T) + self.vbias)

    def _Ps(self, n, l):
        return T.exp(-l) * (l ** n) / T.gamma(n + 1)

    def _poisson_mask(self, v0_sample):
        wx_b = self.propup(v0_sample)
        wTx_b = self.propdown(wx_b)

        # un-normalized poisson rate, L
        L = T.exp(wTx_b)
        # now we normalize it wrt length of wordvector and partition function
        L = L * T.sum(v0_sample, axis=1)[:, numpy.newaxis] / \
            T.sum(L, axis=1)[:, numpy.newaxis]

        # v1_mean gives the probability of seen self.x
        v1_mean = self._Ps(v0_sample, L)

        return v1_mean

    def _get_cost_update(self, corruption_level, learning_rate):
        """ This function computes the cost and the updates for one trainng
        step of the AE """

        z = self._poisson_mask(self.x) * self.x

        cost = T.sqrt(T.sum(T.sqr(self.x - z)) / self.x.shape[0])

        # compute the gradients of the cost of the `AE` with respect
        # to its parameters
        gparams = T.grad(cost, self.params)
        # generate the list of updates
        updates = []
        for param, gparam in zip(self.params, gparams):
            updates.append((param, param - learning_rate * gparam))

        return cost, updates


def toy_test(learning_rate=0.01, training_epochs=100, batch_size=50,
             output_folder=None, isPCD=0,
             n_hidden=3):
    assert output_folder is not None
    # toy_data, word count vector, [num_terms, num_doc].
    # each cell represents the number of times a term occurs
    #                          d1 d2 d3 d4 d5
    toy_data = numpy.asarray([[0, 2, 0, 1, 0],
                              [9, 0, 3, 1, 1],
                              [4, 1, 1, 2, 1],
                              [10, 10, 1, 1, 0],
                              [1, 0, 8, 0, 10],
                              [0, 1, 10, 1, 0],
                              [1, 0, 2, 6, 1],
                              [0, 0, 1, 0, 0],
                              [1, 0, 0, 0, 0],
                              [1, 0, 1, 0, 0],
                              [1, 1, 0, 0, 1],
                              [10, 2, 0, 1, 0],
                              [0, 0, 1, 0, 10],
                              [1, 0, 0, 3, 0],
                              [0, 0, 2, 0, 1],
                              [10, 0, 1, 0, 0],
                              [0, 1, 0, 0, 0],
                              [0, 1, 0, 1, 0],
                              [1, 0, 1, 0, 0],
                              [1, 0, 0, 0, 1],
                              [1, 0, 1, 0, 0],
                              [0, 0, 1, 0, 0]])

    from xylearn.models.ae_variants import AE_Poisson as AE


    train_x = toSharedX(toy_data, name="toy_data")

    n_vis = train_x.get_value(borrow=True).shape[1]

    n_samples = train_x.get_value(borrow=True).shape[0]

    if batch_size >= n_samples:
        batch_size = n_samples

    n_train_batches = n_samples / batch_size


    # construct the RBM class
    ae = AE(n_visible=n_vis, n_hidden=n_hidden)
    train_fn = ae.get_train_fn(train_x, batch_size)

    # print "... projecting"
    # print ae.project(train_x)

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
            X=ae.W.get_value(borrow=True).T,
            # weight is [n_vis, n_hidden]
            # so, among 'n_hidden' rows,
            # each row corresponds to propdown one hidden unit
            img_shape=(1, n_vis), tile_shape=(n_hidden, 1),
            tile_spacing=(1, 1)))
        image.save('filters_at_epoch_%i.png' % epoch)
        plotting_stop = time.clock()
        plotting_time += (plotting_stop - plotting_start)

    end_time = time.clock()
    pretraining_time = (end_time - start_time) - plotting_time
    print ('Training took %f minutes' % (pretraining_time / 60.))

    print "... projecting"
    print ae.project(train_x, rounding=1)

    print "... reconstructing"
    print ae.reconstruct(train_x, rounding=1)


def test_ae_mnist(learning_rate=0.2, training_epochs=10, batch_size=20,
                  output_folder=None, n_hidden=500, corruption_lvl=0.1):
    """
    :param learning_rate: learning rate used for training the RBM

    :param training_epochs: number of epochs used for training

    :param dataset: path the the pickled dataset

    :param batch_size: size of a batch used to train the RBM

    :param n_chains: number of parallel Gibbs chains to be used for sampling

    :param n_samples: number of samples to plot for each chain

    e.g.
        test_ae_mnist(output_folder='/home/eric/Desktop/ae_plots')

    """

    assert output_folder is not None

    from ae_variants import AE_Orthogonal as AE

    #################################
    #     Data Constructing         #
    #################################

    from sklearn.datasets import fetch_mldata

    mnist = fetch_mldata('MNIST original')

    from xylearn.utils.data_util import get_train_test
    from xylearn.utils.data_normalization import rescale

    data = get_train_test(rescale(mnist.data), mnist.target, useGPU=1)

    train_x, train_y = data['train']
    n_vis = train_x.get_value(borrow=True).shape[1]
    n_train_batches = train_x.get_value(borrow=True).shape[0] / batch_size


    # construct the AE class
    ae = AE(n_visible=n_vis, n_hidden=n_hidden)
    train_fn = ae.get_train_fn(train_x, batch_size, corruption_level=corruption_lvl)


    #################################
    #     Training the AE           #
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
        # monitor projected rank
        projection = ae.project(train_x)
        print numpy.linalg.matrix_rank(projection)

        # W shape is [784 500]
        # Plot filters after each training epoch
        plotting_start = time.clock()
        # Construct image from the weight matrix
        image = PIL.Image.fromarray(tile_raster_images(
            X=ae.W.get_value(borrow=True).T,
            img_shape=(28, 28), tile_shape=(20, 20),
            tile_spacing=(1, 1)))
        image.save('filters_at_epoch_%i.png' % epoch)
        plotting_stop = time.clock()
        plotting_time += (plotting_stop - plotting_start)

    end_time = time.clock()
    pretraining_time = (end_time - start_time) - plotting_time
    print ('Training took %f minutes' % (pretraining_time / 60.))


    #################################
    #     Reconstruct from AE       #
    #################################

    test_idx = 1
    test_x, test_y = data['test']

    print '... begin sampling'
    # plot initial image first
    orig_img = test_x.get_value(borrow=True)[test_idx:test_idx + 1]
    image = PIL.Image.fromarray(tile_raster_images(
        X=orig_img,
        img_shape=(28, 28), tile_shape=(1, 1),
        tile_spacing=(1, 1)))
    image.save('orig_img.png')

    print '... reconstructing'
    recon_img = ae.reconstruct(test_x)
    image = PIL.Image.fromarray(tile_raster_images(
        X=recon_img,
        img_shape=(28, 28), tile_shape=(1, 1),
        tile_spacing=(1, 1)))
    image.save('samples.png')
    os.chdir('../')


if __name__ == '__main__':
    test_ae_mnist(output_folder='/home/eric/Desktop/ae_plots', learning_rate=0.01,
                  training_epochs=100, batch_size=20,
                  n_hidden=500, corruption_lvl=0)

    # toy_test(output_folder='/Volumes/HDD750/home/TEMP/lae_plots',
    #          training_epochs=100, learning_rate=0.1,
    #          n_hidden=10)