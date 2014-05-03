"""
 References :
   - P. Vincent, H. Larochelle, Y. Bengio, P.A. Manzagol: Extracting and
   Composing Robust Features with Denoising Autoencoders, ICML'08, 1096-1103,
   2008
   - Y. Bengio, P. Lamblin, D. Popovici, H. Larochelle: Greedy Layer-Wise
   Training of Deep Networks, Advances in Neural Information Processing
   Systems 19, 2007

"""

import os
import time

import numpy
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
from xylearn.utils.theano_util import toSharedX




# for debugging
from xylearn.visualizer.terminal_printer import TerminalPrinter


class AE(object):
    """Denoising Auto-Encoder class (AE)

    .. math::

        \tilde{x} ~ q_D(\tilde{x}|x)                                     (1)

        y = s(W \tilde{x} + b)                                           (2)

        x = s(W' y  + b')                                                (3)

        L(x,z) = -sum_{k=1}^d [x_k \log z_k + (1-x_k) \log( 1-z_k)]      (4)

    """

    def __init__(self, numpy_rng=None, theano_rng=None, input=None,
                 n_visible=784, n_hidden=500,
                 W=None, hbias=None, vbias=None, aFunc=T.nnet.sigmoid):
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
        if aFunc is not None:
            self.aFunc = aFunc

        self.n_visible = n_visible
        self.n_hidden = n_hidden

        if numpy_rng is None:
            # create a number generator
            numpy_rng = numpy.random.RandomState(1234)

        # create a Theano random generator that gives symbolic random values
        if not theano_rng:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))

        if not W:
            # W is initialized with `initial_W` which is uniformely sampled
            # from -4*sqrt(6./(n_visible+n_hidden)) and
            # 4*sqrt(6./(n_hidden+n_visible))the output of uniform if
            # converted using asarray to dtype
            # theano.config.floatX so that the code is runable on GPU
            initial_W = numpy.asarray(numpy_rng.uniform(
                low=-4 * numpy.sqrt(6. / (n_hidden + n_visible)),
                high=4 * numpy.sqrt(6. / (n_hidden + n_visible)),
                size=(n_visible, n_hidden)), dtype=theano.config.floatX)
            W = theano.shared(value=initial_W, name='W', borrow=True)

        if not vbias:
            vbias = theano.shared(value=numpy.zeros(n_visible,
                                                    dtype=theano.config.floatX),
                                  borrow=True)

        if not hbias:
            hbias = theano.shared(value=numpy.zeros(n_hidden,
                                                    dtype=theano.config.floatX),
                                  name='b',
                                  borrow=True)

        self.W = W
        # b corresponds to the bias of the hidden
        self.b = hbias
        # vbias corresponds to the bias of the visible
        self.vbias = vbias

        self.theano_rng = theano_rng
        # if no input is given, generate a variable representing the input
        if input == None:
            # we use a matrix because we expect a minibatch of several
            # examples, each example being a row
            self.x = T.matrix('x')
        else:
            self.x = input

        self.params = [self.W, self.b, self.vbias]

    def get_corrupted_input(self, input, corruption_level):
        """This function keeps ``1-corruption_level`` entries of the inputs the
        same and zero-out randomly selected subset of size ``coruption_level``
        Note : first argument of theano.rng.binomial is the shape(size) of
               random numbers that it should produce
               second argument is the number of trials
               third argument is the probability of success of any trial

                this will produce an array of 0s and 1s where 1 has a
                probability of 1 - ``corruption_level`` and 0 with
                ``corruption_level``

                The binomial function return int64 data type by
                default.  int64 multiplicated by the input
                type(floatX) always return float64.  To keep all data
                in floatX when floatX is float32, we set the dtype of
                the binomial to floatX. As in our case the value of
                the binomial is always 0 or 1, this don't change the
                result. This is needed to allow the gpu to work
                correctly as it only support float32 for now.

        """
        return self.theano_rng.binomial(size=input.shape, n=1,
                                        p=1 - corruption_level,
                                        dtype=theano.config.floatX) * input

    def propup(self, input):
        """ Computes the values of the hidden layer """
        return self.aFunc(T.dot(input, self.W) + self.b)

    def propdown(self, hid):
        return self.aFunc(T.dot(hid, self.W.T) + self.vbias)

    def _get_reconstruction_cost(self, recon):

        cross_entropy = T.mean(
            T.sum(self.x * T.log(recon) +
                  (1 - self.x) * T.log(1 - recon),
                  axis=1))

        return cross_entropy

    def _get_cost_update(self, corruption_level, learning_rate):
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
        cost = T.mean(L)

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
        cost, updates = self._get_cost_update(corruption_level=corruption_level,
                                              learning_rate=learning_rate)

        fn = theano.function([index, theano.Param(learning_rate, default=0.01)],
                             outputs=cost,
                             updates=updates,
                             givens={self.x: dataX[index * batch_size:(index + 1) * batch_size]},
                             name='train_ae')

        return fn

    # reproduce data
    def get_reconstruction_error(self, testX):
        z = self.propdown(self.propup(testX))

        error = theano.function([], theano.Out(self._get_reconstruction_cost(z),
                                               borrow=True),
                                  givens={self.x: testX},
                                  name='ae_recon_error')
        return error()

    def project(self, dataX, rounding=0):
        """
        project dataX into hidden space. In other words, when rbm was trained,
        we can get new representation
        """
        if rounding:
            h1_mean = T.iround(self.propup(dataX))
        else:
            h1_mean = self.propup(dataX)
        fn = theano.function([], theano.Out(h1_mean, borrow=True), name='project')

        return fn()

    def reconstruct(self, testX, rounding=0):
        if rounding:
            z = T.iround(self.propdown(self.propup(testX)))
        else:
            z = self.propdown(self.propup(testX))

        fn = theano.function([], theano.Out(z, borrow=True), name='ae_recon')
        return fn()


def test_ae_mnist(learning_rate=0.8, training_epochs=10, batch_size=20,
                  output_folder=None, n_hidden=500, corruption_lvl=0.3):
    """
    Demonstrate how to train and afterwards sample from it using Theano.

    This is demonstrated on MNIST.

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
            # new_cost = train_fn(index=batch_index, lr=learning_rate)
            new_cost = train_fn(index=batch_index)
            mean_cost += [new_cost]

        print 'Training epoch %d, cost is ' % epoch, numpy.mean(mean_cost)

        # W shape is [784 500]
        # Plot filters after each training epoch
        plotting_start = time.clock()
        # Construct image from the weight matrix
        image = PIL.Image.fromarray(tile_raster_images(
            X=ae.W.get_value(borrow=True).T,
            img_shape=(28, 28), tile_shape=(10, 10),
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


def begin(data=None,
          learning_rate=0.13,
          n_epochs=100,
          n_hidden=50,
          batch_size=600):
    """
    1) load dataset
    """

    Tp = TerminalPrinter(debug=False, verbose=True)
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
    ae = AE(n_visible=num_of_dims, n_hidden=n_hidden)
    """
    3) compile training function
    """
    train_fn = ae.get_train_fn(train_x, batch_size)

    """
    4) begin training
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
    5) begin testing
    """
    Tp.Print('... testing the model')
    test_x, test_y = data['test']
    test_x = toSharedX(test_x, borrow=True)

    return ae.get_reconstruction_error(test_x)


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
    # print kfold_train()
    test_ae_mnist(output_folder='/Volumes/HDD750/home/TEMP//ae_plots')