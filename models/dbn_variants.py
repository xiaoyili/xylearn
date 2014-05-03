__author__ = 'eric'

import os
import sys
import time

import numpy
import theano.tensor as T
import theano
from theano.tensor.shared_randomstreams import RandomStreams

from xylearn.utils.theano_util import toSharedX

from dbn import DBN
from mlp import HiddenLayer
from softmax_regression import SoftmaxRegression
from rbm import RBM


class DBN_SPARSE_L2(DBN):
    """
    dbn with sparse and L2 constrain
    """

    def __init__(self, numpy_rng=None, theano_rng=None, n_ins=784, n_outs=None,
                 aFunc=T.nnet.sigmoid, dropout=0.0,
                 hidden_layers_sizes=[500, 500], isPCD=0, externParams=None):
        """This class is made to support a variable number of layers.

        :type numpy_rng: numpy.random.RandomState
        :param numpy_rng: numpy random number generator used to draw initial
                    weights

        :type theano_rng: theano.tensor.shared_randomstreams.RandomStreams
        :param theano_rng: Theano random generator; if None is given one is
                           generated based on a seed drawn from `rng`

        :type n_ins: int
        :param n_ins: dimension of the input to the DBN

        :type hidden_layers_sizes: list of ints
        :param hidden_layers_sizes: intermediate layers size, must contain
                               at least one value

        :type n_outs: int
        :param n_outs: dimension of the output of the network

        :type isPCD: use persistent CD
        :param isPCD: 0 or 1

        :type externParams: load external parameters to initialize
        :param: list of tuple e.g. [(w,b), (w,b), (w,a,b)] from bottom to top

        """
        from rbm_variants import RBM_SPARSE_L2 as RBM

        super(DBN_SPARSE_L2, self).__init__(n_ins=n_ins, n_outs=n_outs,
                                            dropout=dropout, numpy_rng=numpy_rng, aFunc=aFunc,
                                            theano_rng=theano_rng, isPCD=isPCD, RBM=RBM,
                                            hidden_layers_sizes=hidden_layers_sizes, externParams=externParams)

    def get_pretrain_fns(self, train_set_x, batch_size, k=1, dropout=0.0):
        '''Generates a list of functions, for all layers. Used for greedy
        layer-wised pretraining

        :type train_set_x: theano.tensor.TensorType
        :param train_set_x: Shared var. that contains all datapoints used
                            for training the RBM
        :type batch_size: int
        :param batch_size: size of a [mini]batch
        :param k: number of Gibbs steps to do in CD-k / PCD-k

        '''

        # index to a [mini]batch
        index = T.lscalar('index')  # index to a minibatch
        learning_rate = T.scalar('lr')  # learning rate to use
        Beta = T.scalar('beta')
        Gamma = T.scalar('gamma')
        Sparseness = T.scalar('sparseness')

        self.dropout = dropout

        # begining of a batch, given `index`
        batch_begin = index * batch_size
        # ending of a batch given `index`
        batch_end = batch_begin + batch_size

        pretrain_fns = []
        counter = 0
        for rbm in self.rbm_layers:
            # initialize storage for the persistent chain (state = hidden
            # layer of chain)
            if self.isPCD:
                persistent_chain = theano.shared(numpy.zeros((batch_size, rbm.n_hidden),
                                                             dtype=theano.config.floatX),
                                                 borrow=True)
            else:
                persistent_chain = None

            # cost = free_energy(start) - free_energy(chain_end)
            cost, updates = rbm._get_cost_update(learning_rate,
                                                 persistent=persistent_chain, k=k,
                                                 beta=Beta, gamma=Gamma, s_constrain=Sparseness)

            # compile the theano function
            fn = theano.function(inputs=[index,
                                         theano.Param(learning_rate, default=0.1),
                                         theano.Param(Beta, default=0.1),
                                         theano.Param(Gamma, default=0.0001),
                                         theano.Param(Sparseness, default=0.05)],
                                 outputs=cost,
                                 updates=updates,
                                 givens={self.x:
                                             train_set_x[batch_begin:batch_end]},
                                 name=str('dbn_S_L2_pretrain_fn_' + str(counter)))
            # append `fn` to the list of functions
            pretrain_fns.append(fn)

            counter += 1

        return pretrain_fns


class DBN_L1(DBN):
    """
    dbn with L1 constrain
    """

    def __init__(self, numpy_rng=None, theano_rng=None, n_ins=784, n_outs=None,
                 aFunc=T.nnet.sigmoid, dropout=0.0,
                 hidden_layers_sizes=[500, 500], isPCD=0, externParams=None):
        """This class is made to support a variable number of layers.

        :type numpy_rng: numpy.random.RandomState
        :param numpy_rng: numpy random number generator used to draw initial
                    weights

        :type theano_rng: theano.tensor.shared_randomstreams.RandomStreams
        :param theano_rng: Theano random generator; if None is given one is
                           generated based on a seed drawn from `rng`

        :type n_ins: int
        :param n_ins: dimension of the input to the DBN

        :type hidden_layers_sizes: list of ints
        :param hidden_layers_sizes: intermediate layers size, must contain
                               at least one value

        :type n_outs: int
        :param n_outs: dimension of the output of the network

        :type isPCD: use persistent CD
        :param isPCD: 0 or 1

        :type externParams: load external parameters to initialize
        :param: list of tuple e.g. [(w,b), (w,b), (w,a,b)] from bottom to top

        """

        from rbm_variants import RBM_L1

        super(DBN_L1, self).__init__(n_ins=n_ins, n_outs=n_outs,
                                     dropout=dropout, numpy_rng=numpy_rng, aFunc=aFunc,
                                     theano_rng=theano_rng, isPCD=isPCD, RBM=RBM_L1,
                                     hidden_layers_sizes=hidden_layers_sizes, externParams=externParams)

    def get_pretrain_fns(self, train_set_x, batch_size, k=1, dropout=0.0):
        '''Generates a list of functions, for all layers. Used for greedy
        layer-wised pretraining

        :type train_set_x: theano.tensor.TensorType
        :param train_set_x: Shared var. that contains all datapoints used
                            for training the RBM
        :type batch_size: int
        :param batch_size: size of a [mini]batch
        :param k: number of Gibbs steps to do in CD-k / PCD-k

        '''

        # index to a [mini]batch
        index = T.lscalar('index')  # index to a minibatch
        learning_rate = T.scalar('lr')  # learning rate to use
        Beta = T.scalar('beta')

        self.dropout = dropout

        # begining of a batch, given `index`
        batch_begin = index * batch_size
        # ending of a batch given `index`
        batch_end = batch_begin + batch_size

        pretrain_fns = []
        counter = 0
        for rbm in self.rbm_layers:
            # initialize storage for the persistent chain (state = hidden
            # layer of chain)
            if self.isPCD:
                persistent_chain = theano.shared(numpy.zeros((batch_size, rbm.n_hidden),
                                                             dtype=theano.config.floatX),
                                                 borrow=True)
            else:
                persistent_chain = None

            # cost = free_energy(start) - free_energy(chain_end)
            cost, updates = rbm._get_cost_update(learning_rate,
                                                 persistent=persistent_chain, k=k,
                                                 beta=Beta)

            # compile the theano function
            fn = theano.function(inputs=[index,
                                         theano.Param(learning_rate, default=0.1),
                                         theano.Param(Beta, default=0.01)],
                                 outputs=cost,
                                 updates=updates,
                                 givens={self.x:
                                             train_set_x[batch_begin:batch_end]},
                                 name=str('dbn_L1_pretrain_fn_' + str(counter)))
            # append `fn` to the list of functions
            pretrain_fns.append(fn)

            counter += 1

        return pretrain_fns


class MultiTask_DBN(DBN):
    def __init__(self, numpy_rng=None, theano_rng=None, n_ins=784, n_outs=None,
                 aFunc=T.nnet.sigmoid, dropout=0.0, RBM=RBM,
                 hidden_layers_sizes=[500, 500], isPCD=0, externParams=None):
        """This class is made to support a variable number of layers.

        :type numpy_rng: numpy.random.RandomState
        :param numpy_rng: numpy random number generator used to draw initial
                    weights

        :type theano_rng: theano.tensor.shared_randomstreams.RandomStreams
        :param theano_rng: Theano random generator; if None is given one is
                           generated based on a seed drawn from `rng`

        :type n_ins: int
        :param n_ins: dimension of the input to the DBN

        :type hidden_layers_sizes: list of ints
        :param hidden_layers_sizes: intermediate layers size, must contain
                               at least one value

        :type n_outs: list of int, e.g. [2, 2, 2] means 3 tasks with binary classification
        :param n_outs: dimension of the outputs of the network

        :type isPCD: use persistent CD
        :param isPCD: 0 or 1

        :type externParams: load external parameters to initialize
        :param: list of tuple e.g. [(w,b), (w,b), (w,a,b)] from bottom to top

        """

        self.sigmoid_layers = []
        self.rbm_layers = []
        self.params = []
        self.n_layers = len(hidden_layers_sizes)
        self.hidden_layers_sizes = hidden_layers_sizes
        self.n_outs = n_outs
        self.n_ins = n_ins
        self.isPCD = isPCD

        # for plotting
        self.input_length = int(numpy.sqrt(n_ins))
        self.input_height = int(numpy.sqrt(n_ins))

        self.dropout = dropout

        # params_dict: backing up all params
        self.params_dict = dict()

        assert self.n_layers > 0

        if not numpy_rng:
            # create a number generator
            numpy_rng = numpy.random.RandomState(1234)

        if not theano_rng:
            self.theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))


        # allocate symbolic variables for the data
        self.x = T.matrix('x')

        for i in xrange(self.n_layers):

            # the size of the input is either the number of hidden
            # units of the layer below or the input size if we are on
            # the first layer
            if i == 0:
                input_size = self.n_ins
            else:
                input_size = self.hidden_layers_sizes[i - 1]

            # the input to this layer is either the activation of the
            # hidden layer below or the input of the DBN if you are on
            # the first layer
            if i == 0:
                layer_input = self.x
            else:
                layer_input = self.sigmoid_layers[-1].output

            if externParams is None:
                sigmoid_layer = HiddenLayer(rng=numpy_rng,
                                            input=layer_input,
                                            n_in=input_size,
                                            n_out=self.hidden_layers_sizes[i],
                                            activation=aFunc,
                                            dropout=self.dropout)
            else:
                sigmoid_layer = HiddenLayer(rng=numpy_rng,
                                            input=layer_input,
                                            n_in=input_size,
                                            W=externParams[i][0],
                                            b=externParams[i][1],
                                            n_out=self.hidden_layers_sizes[i],
                                            activation=aFunc,
                                            dropout=self.dropout)



            # add the layer to our list of layers
            self.sigmoid_layers.append(sigmoid_layer)

            # the last sigmoid layer
            if i == self.n_layers - 1:
                self.output_layer = sigmoid_layer.output

            # its arguably a philosophical question...  but we are
            # going to only declare that the parameters of the
            # sigmoid_layers are parameters of the DBN. The visible
            # biases in the RBM are parameters of those RBMs, but not
            # of the DBN.
            self.params.extend(sigmoid_layer.params)

            # attach rbm to the current sigmoid layer
            rbm_layer = RBM(numpy_rng=numpy_rng,
                            theano_rng=theano_rng,
                            input=layer_input,
                            n_visible=input_size,
                            aFunc=aFunc,
                            n_hidden=self.hidden_layers_sizes[i],
                            W=sigmoid_layer.W,
                            hbias=sigmoid_layer.b)

            self.rbm_layers.append(rbm_layer)

        # now contains vbias as well

        if isinstance(n_outs, list) and len(n_outs) > 0:
            self.y = dict()
            self.logLayer = dict()
            self.logLayerParams = dict()
            self.finetune_cost = 0
            self.errors = 0

            for i in range(len(n_outs)):
                self.y[i] = T.ivector('y')
                self.logLayer[i] = SoftmaxRegression(
                    input=self.sigmoid_layers[-1].output,
                    n_in=self.hidden_layers_sizes[-1],
                    n_out=self.n_outs[i])
                self.logLayerParams[i] = self.logLayer[i].params

                self.finetune_cost += self.logLayer[i]._negative_log_likelihood(self.y[i])

                self.errors += self.logLayer[i].errors(self.y[i])

            self.finetune_cost /= len(n_outs)
            self.errors /= len(n_outs)

        else:
            exit('n_outs should be a list of integers')

        self.params_dict['numpy_rng'] = numpy_rng
        self.params_dict['n_ins'] = n_ins
        self.params_dict['hidden_layers_sizes'] = hidden_layers_sizes
        self.params_dict['n_layers'] = self.n_layers
        self.params_dict['params'] = self.params
        self.params_dict['theano_rng'] = theano_rng

    def get_finetune_fn(self, train_set_x, train_set_y,
                        batch_size, fineTune='All', dropout=0.0):
        '''Generates a function `train` that implements one step of
        finetuning, a function `validate` that computes the error on a
        batch from the validation set, and a function `test` that
        computes the error on a batch from the testing set

        :type train_set_x: n x m matrix of theano.tensor.TensorType
        :param train_set_x:

        :type train_set_y: list of vector theano.tensor.TensorType
        :param train_set_y:

        :type batch_size: int
        :param batch_size: size of a minibatch

        :type fineTune: fine tuning only the top layer, or the whole network
        :param fineTune: 'Top', 'All'

        '''

        if self.n_outs is None:
            exit('ERROR: Number of output units not defined!')

        if not isinstance(train_set_y, list) and len(train_set_y) != len(self.n_outs):
            exit('ERROR: labels should be a list of vectors')

        index = T.lscalar('index')  # index to a [mini]batch
        learning_rate = T.scalar('lr')

        self.dropout = dropout

        # compute the gradients with respect to the model parameters
        params = []
        if fineTune == 'Top':
            for i in range(len(self.logLayerParams)):
                params.extend(self.logLayerParams[i])
        elif fineTune == 'All':
            params = self.params
            for i in range(len(self.logLayerParams)):
                params.extend(self.logLayerParams[i])

        cost = self.finetune_cost

        gparams = T.grad(cost, params)

        # compute list of fine-tuning updates
        from collections import OrderedDict

        updates = OrderedDict()
        for param, gparam in zip(params, gparams):
            updates[param] = param - gparam * T.cast(learning_rate, dtype=theano.config.floatX)

        given_dict = dict()
        given_dict[self.x] = train_set_x[index * batch_size: (index + 1) * batch_size]
        for i in range(len(self.y)):
            given_dict[self.y[i]] = train_set_y[i][index * batch_size: (index + 1) * batch_size]

        fn = theano.function(inputs=[index, theano.Param(learning_rate, default=0.01)],
                             outputs=self.finetune_cost,
                             updates=updates,
                             givens=given_dict,
                             name='dbn_finetune_fn')

        return fn


    def get_prediction(self, testX, task_id=0):
        """
        get softmax layer output
        """
        prediction = theano.function([], self.logLayer[task_id].p_y_given_x,
                                       givens={self.x: testX},
                                       name='dbn_get_prediction')
        return prediction()


    def get_error_rate(self, testX, testY, task_id=0):
        if not isinstance(testX, theano.sandbox.cuda.var.CudaNdarraySharedVariable):
            testX = toSharedX(testX)
        if isinstance(testY, T.TensorVariable):
            testY = testY.eval()

        mat = self.get_prediction(testX, task_id)
        y_pred = numpy.argmax(mat, axis=1)

        return numpy.mean(y_pred != testY)

    def get_AUC(self, testX, testY, task_id=0):
        if not isinstance(testX, theano.sandbox.cuda.var.CudaNdarraySharedVariable):
            testX = toSharedX(testX)
        if isinstance(testY, T.TensorVariable):
            testY = testY.eval()

        mat = self.get_prediction(testX, task_id)
        y_pred = numpy.argmax(mat, axis=1)

        y_mean = [mat[i, testY[i]] for i in range(len(y_pred))]

        from sklearn.metrics import roc_auc_score

        return roc_auc_score(testY, y_pred)


def test_dbn_mnist(learning_rate=0.01, training_epochs=10, batch_size=20,
                   finetune_lr=0.05, finetune_epoch=10, output_folder=None, dropout=0.2,
                   model_structure=[500, 100], isPCD=0, k=1, plot_weight=False):
    """
        test_dbn_mnist(output_folder='/home/eric/Desktop/dbn_plots',
                   training_epochs=10,
                   model_structure=[500, 100],
                   finetune_epoch=10)
    """
    assert output_folder is not None

    from dbn_variants import DBN as DBN
    #################################
    #     Data Constructing         #
    #################################

    from sklearn.datasets import fetch_mldata

    dataset = fetch_mldata('MNIST original')

    # from sklearn.datasets import load_digits
    # dataset = load_digits()

    from xylearn.utils.data_util import get_train_test
    from xylearn.utils.data_normalization import rescale

    data = get_train_test(rescale(dataset.data), dataset.target, useGPU=1, shuffle=True)

    train_x, train_y = data['train']
    test_x, test_y = data['test']
    num_of_classes = max(train_y.eval()) - min(train_y.eval()) + 1
    n_vis = train_x.get_value(borrow=True).shape[1]
    n_train_batches = train_x.get_value(borrow=True).shape[0] / batch_size

    print '... building the model'

    dbn = DBN(n_ins=n_vis, n_outs=num_of_classes, dropout=dropout,
              hidden_layers_sizes=model_structure, isPCD=isPCD)

    print '... getting the pretraining functions'
    pretrain_fns = dbn.get_pretrain_fns(train_set_x=train_x,
                                        batch_size=batch_size,
                                        k=k)
    #################################
    #     Training the DBN          #
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
        for i in xrange(dbn.n_layers):
            # go through pretraining epochs
            for batch_index in xrange(n_train_batches):
                # for each batch, we extract the gibbs chain
                new_cost = pretrain_fns[i](index=batch_index, lr=learning_rate)
                mean_cost += [new_cost]

            if plot_weight:
                # plot all weights
                weight = dbn.project_back(dbn.sigmoid_layers[i].W.T,
                                          from_layer_i=i)
                # Construct image from the weight matrix
                image = PIL.Image.fromarray(tile_raster_images(
                    X=weight,
                    img_shape=(dbn.input_length, dbn.input_height), tile_shape=(10, 10),
                    tile_spacing=(1, 1)))
                image.save('pretrain_epoch_%i_layer_%i.png' % (epoch, i))

        print 'Training epoch %d, cost is ' % epoch, numpy.mean(mean_cost)

    end_time = time.clock()
    pretraining_time = end_time - start_time
    print ('Training took %f minutes' % (pretraining_time / 60.))

    ########################
    # FINETUNING THE MODEL #
    ########################
    # get the training, validation and testing function for the model
    print '... getting the finetuning functions'
    finetune_fn = dbn.get_finetune_fn(train_x, train_y, batch_size, 'All')

    print '... finetunning the model'

    test_score = 0.

    start_time = time.clock()
    # go through the each batch
    for epoch in xrange(finetune_epoch):
        ## Pre-train layer-wise
        c = []
        for batch_index in xrange(n_train_batches):
            c.append(finetune_fn(index=batch_index, lr=finetune_lr))

        if plot_weight:
            # plot all weights
            weight = dbn.project_back(dbn.sigmoid_layers[-1].W.T,
                                      from_layer_i=dbn.n_layers - 1)
            # Construct image from the weight matrix
            image = PIL.Image.fromarray(tile_raster_images(
                X=weight,
                img_shape=(dbn.input_length, dbn.input_height), tile_shape=(10, 10),
                tile_spacing=(1, 1)))
            image.save('finetune_epoch_%i.png' % (epoch))

        print 'Fine-tuning epoch %i, model cost ' % epoch,
        print numpy.mean(c)
        test_score = numpy.mean(c)

    end_time = time.clock()
    print(('Optimization complete with best test performance %f %%') %
          (100. - test_score * 100.))
    print >> sys.stderr, ('The fine tuning code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time)
                                              / 60.))

    ########################
    #   Test THE MODEL     #
    ########################
    print '\nTest complete with error rate ', dbn.get_error_rate(test_x, test_y)


def test_multitask_dbn_mnist(learning_rate=0.01, training_epochs=10, batch_size=20,
                             finetune_lr=0.05, finetune_epoch=10, output_folder=None, dropout=0.2,
                             model_structure=[500, 100], isPCD=0, k=1, plot_weight=False):
    """
        test_dbn_mnist(output_folder='/home/eric/Desktop/dbn_plots',
                   training_epochs=10,
                   model_structure=[500, 100],
                   finetune_epoch=10)
    """
    assert output_folder is not None

    from dbn_variants import MultiTask_DBN as DBN
    #################################
    #     Data Constructing         #
    #################################

    from sklearn.datasets import fetch_mldata
    from xylearn.utils.data_util import get_train_test
    from xylearn.utils.data_normalization import rescale


    dataset = fetch_mldata('MNIST original')

    data = get_train_test(rescale(dataset.data), dataset.target, useGPU=1, shuffle=True)

    train_x, train_y = data['train']
    test_x, test_y = data['test']

    num_of_classes = max(train_y.eval()) - min(train_y.eval()) + 1

    # man-made task label for testing
    n_outs = [num_of_classes, num_of_classes, num_of_classes]

    n_vis = train_x.get_value(borrow=True).shape[1]
    n_train_batches = train_x.get_value(borrow=True).shape[0] / batch_size

    print '... building the model'

    dbn = DBN(n_ins=n_vis, n_outs=n_outs, dropout=dropout,
              hidden_layers_sizes=model_structure, isPCD=isPCD)

    print '... getting the pretraining functions'
    pretrain_fns = dbn.get_pretrain_fns(train_set_x=train_x,
                                        batch_size=batch_size,
                                        k=k)
    #################################
    #     Training the DBN          #
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
        for i in xrange(dbn.n_layers):
            # go through pretraining epochs
            for batch_index in xrange(n_train_batches):
                # for each batch, we extract the gibbs chain
                new_cost = pretrain_fns[i](index=batch_index, lr=learning_rate)
                mean_cost += [new_cost]

            if plot_weight:
                # plot all weights
                weight = dbn.project_back(dbn.sigmoid_layers[i].W.T,
                                          from_layer_i=i)
                # Construct image from the weight matrix
                image = PIL.Image.fromarray(tile_raster_images(
                    X=weight,
                    img_shape=(dbn.input_length, dbn.input_height), tile_shape=(10, 10),
                    tile_spacing=(1, 1)))
                image.save('pretrain_epoch_%i_layer_%i.png' % (epoch, i))

        print 'Training epoch %d, cost is ' % epoch, numpy.mean(mean_cost)

    end_time = time.clock()
    pretraining_time = end_time - start_time
    print ('Training took %f minutes' % (pretraining_time / 60.))

    ########################
    # FINETUNING THE MODEL #
    ########################
    # get the training, validation and testing function for the model
    print '... getting the finetuning functions'

    # man-made labels, list of labels
    train_ys = [train_y, train_y, train_y]
    finetune_fn = dbn.get_finetune_fn(train_x, train_ys, batch_size)

    print '... finetunning the model'

    test_score = 0.

    start_time = time.clock()
    # go through the each batch
    for epoch in xrange(finetune_epoch):
        ## Pre-train layer-wise
        c = []
        for batch_index in xrange(n_train_batches):
            c.append(finetune_fn(index=batch_index, lr=finetune_lr))

        if plot_weight:
            # plot all weights
            weight = dbn.project_back(dbn.sigmoid_layers[-1].W.T,
                                      from_layer_i=dbn.n_layers - 1)
            # Construct image from the weight matrix
            image = PIL.Image.fromarray(tile_raster_images(
                X=weight,
                img_shape=(dbn.input_length, dbn.input_height), tile_shape=(10, 10),
                tile_spacing=(1, 1)))
            image.save('finetune_epoch_%i.png' % (epoch))

        print 'Fine-tuning epoch %i, model cost ' % epoch,
        print numpy.mean(c)
        test_score = numpy.mean(c)

    end_time = time.clock()
    print(('Optimization complete with best test performance %f %%') %
          (100. - test_score * 100.))
    print >> sys.stderr, ('The fine tuning code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time)
                                              / 60.))

    ########################
    #   Test THE MODEL     #
    ########################
    task_id = 0
    print '\nTest complete with error rate ' + str(dbn.get_error_rate(test_x, test_y, task_id=task_id)) + \
          ' for task_' + str(task_id)


if __name__ == '__main__':
    test_multitask_dbn_mnist(output_folder='/Volumes/HDD750/home/TEMP/dbn_plots',
                             training_epochs=1,
                             learning_rate=0.05,
                             model_structure=[500, 10],
                             finetune_epoch=10,
                             finetune_lr=0.1,
                             dropout=0.0,
                             plot_weight=True)