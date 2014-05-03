__author__ = 'eric'

import os
import sys
import time

import numpy
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
from xylearn.utils.theano_util import toSharedX

from mlp import HiddenLayer
from softmax_regression import SoftmaxRegression
from rbm import RBM




# for debug
from xylearn.visualizer.terminal_printer import TerminalPrinter


class DBN(object):
    """Deep Belief Network

    """

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

        :type n_outs: int
        :param n_outs: dimension of the output of the network

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
        self.y = T.ivector('y')  # the labels are presented as 1D vector

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

        if n_outs is not None:
            # We now need to add a logistic layer on top of the MLP
            self.logLayer = SoftmaxRegression(
                input=self.sigmoid_layers[-1].output,
                n_in=self.hidden_layers_sizes[-1],
                n_out=self.n_outs)
            self.logLayerParams = self.logLayer.params

            self.finetune_cost = self.logLayer._negative_log_likelihood(self.y)

            self.errors = self.logLayer.errors(self.y)

        self.params_dict['numpy_rng'] = numpy_rng
        self.params_dict['n_ins'] = n_ins
        self.params_dict['hidden_layers_sizes'] = hidden_layers_sizes
        self.params_dict['n_layers'] = self.n_layers
        self.params_dict['params'] = self.params
        self.params_dict['theano_rng'] = theano_rng

    def get_pretrain_fns(self, train_set_x, batch_size,
                         k=1, dropout=0.0):
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
                                                 persistent=persistent_chain, k=k)

            # compile the theano function
            fn = theano.function(inputs=[index, theano.Param(learning_rate, default=0.1)],
                                 outputs=cost,
                                 updates=updates,
                                 givens={self.x:
                                             train_set_x[batch_begin:batch_end]},
                                 name=str('dbn_pretrain_fn_' + str(counter)))
            # append `fn` to the list of functions
            pretrain_fns.append(fn)

            counter += 1

        return pretrain_fns

    def get_finetune_fn(self, train_set_x, train_set_y,
                        batch_size, fineTune='All', dropout=0.0):
        '''Generates a function `train` that implements one step of
        finetuning, a function `validate` that computes the error on a
        batch from the validation set, and a function `test` that
        computes the error on a batch from the testing set

        :type train_set: list of pairs of theano.tensor.TensorType
        :param train_set: (train_set_x, train_set_y)

        :type batch_size: int
        :param batch_size: size of a minibatch

        :type fineTune: fine tuning only the top layer, or the whole network
        :param fineTune: 'Top', 'All'

        '''

        if self.n_outs is None:
            print 'ERROR: Number of output units not defined!'
            return

        index = T.lscalar('index')  # index to a [mini]batch
        learning_rate = T.scalar('lr')

        self.dropout = dropout

        # compute the gradients with respect to the model parameters
        params = []
        if fineTune == 'Top':
            params.extend(self.logLayerParams)
        elif fineTune == 'All':
            params = self.params
            params.extend(self.logLayerParams)

        cost = self.finetune_cost

        gparams = T.grad(cost, params)

        # compute list of fine-tuning updates
        from collections import OrderedDict

        updates = OrderedDict()
        for param, gparam in zip(params, gparams):
            updates[param] = param - gparam * T.cast(learning_rate, dtype=theano.config.floatX)

        fn = theano.function(inputs=[index, theano.Param(learning_rate, default=0.01)],
                             outputs=self.finetune_cost,
                             updates=updates,
                             givens={self.x: train_set_x[index * batch_size:
                             (index + 1) * batch_size],
                                     self.y: train_set_y[index * batch_size:
                                     (index + 1) * batch_size]},
                             name='dbn_finetune_fn')

        return fn

    def get_prediction(self, testX):
        """
        get softmax layer output
        """
        prediction = theano.function([], self.logLayer.p_y_given_x,
                                       givens={self.x: testX},
                                       name='dbn_get_prediction')
        return prediction()

    def project(self, dataX=None, toTensor=0, to_layer_i=None):
        """
        project input data to hidden space.

        toTensor: don't change tensor type if set to 1

        to_layer_i: project to predefined layer

        """
        assert dataX is not None
        v = dataX

        if to_layer_i is None:
            # propup to the top
            for rbm in self.rbm_layers:
                [pre, v] = rbm.propup(v)
        elif to_layer_i > 0:
            for i in xrange(to_layer_i):
                [pre, v] = self.rbm_layers[i].propup(v)

        if toTensor:
            return v
        else:
            fn = theano.function([], theano.Out(v, borrow=True), name='project')
            return fn()

    def project_back(self, dataX=None, toTensor=0, from_layer_i=None):
        """
        project hidden space back to visible layer.

        toTensor: don't change tensor type if set to 1

        from_layer_i: project_back from predefined layer

        """
        assert dataX is not None
        v = dataX
        if from_layer_i is None:
            for rbm in reversed(self.rbm_layers):
                [pre, v] = rbm.propdown(v)
        elif from_layer_i > 0:
            for i in reversed(xrange(from_layer_i)):
                [pre, v] = self.rbm_layers[i].propdown(v)

        if toTensor:
            return v
        else:
            fn = theano.function([], theano.Out(v, borrow=True), name='dbn_proj_back')
            return fn()

    def reconstruct(self, dataX=None, toTensor=0):
        '''
        return: numpy result, theano result
        '''
        assert dataX is not None
        v = self.project_back(self.project(dataX, toTensor=1), toTensor=1)

        if toTensor:
            return v
        else:
            fn = theano.function([], theano.Out(v, borrow=True), name='dbn_reconstruct')
            return fn()

    def params_to_numpy(self):
        """
        transfer parameters to Numpy arrays to save gpu memerory
        """
        params_dict = dict()
        rbm_params = list()
        for params in self.params_dict['params']:
            single_param = list()  # W, hbias, vbias
            for param in params:
                single_param.append(param.get_value(borrow=True))
            rbm_params.append(single_param)

        params_dict['siglayer_params'] = rbm_params

        if self.n_outs is not None:
            logi_params = list()
            for param in self.logLayerParams:
                logi_params.append(param.get_value(borrow=True))

            params_dict['loglayer_params'] = logi_params
        else:
            params_dict['loglayer_params'] = None

        return params_dict

    def get_model_mean(self, toTensor=0):
        '''
        the reconstruction of empty v is the model mean
        '''
        # build empty plate
        plate = T.ones([1, self.rbm_layers[0].W.shape[0]]) * 0.5
        model_mean = self.reconstruct(plate, toTensor=1)

        if toTensor:
            return model_mean
        else:
            fn = theano.function([], theano.Out(model_mean, borrow=True),
                                   name='dbn_model_mean')
            return fn()

    def get_hidden_semantic(self, toTensor=0):
        '''
        visualize topmost layer weight.

        propdown top layer weight, e.g. if top weight is [100, 50]
         then the output should be [n_vis, 50]. each column represents
         a hidden semantic of the topmost neuron
        '''
        hid_semantic = self.project_back(self.sigmoid_layers[-1].W.T,
                                         from_layer_i=self.n_layers - 1)

        if toTensor:
            return hid_semantic
        else:
            fn = theano.function([], theano.Out(hid_semantic, borrow=True),
                                   name='dbn_hidden_semantic')
            return fn()

    def get_error_rate(self, testX, testY):
        if not isinstance(testX, theano.sandbox.cuda.var.CudaNdarraySharedVariable):
            testX = toSharedX(testX)
        if isinstance(testY, T.TensorVariable):
            testY = testY.eval()

        mat = self.get_prediction(testX)
        y_pred = numpy.argmax(mat, axis=1)

        return numpy.mean(y_pred != testY)


"""
TEST CASES
"""


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

    dbn = DBN(n_ins=n_vis, n_outs=num_of_classes,
              hidden_layers_sizes=model_structure, isPCD=isPCD)

    print '... getting the pretraining functions'
    pretrain_fns = dbn.get_pretrain_fns(train_set_x=train_x,
                                        batch_size=batch_size,
                                        k=k)

    #################################
    #     Training the RBM          #
    #################################
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)
    os.chdir(output_folder)

    start_time = time.clock()
    import PIL.Image
    from xylearn.visualizer import tile_raster_images

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
    finetune_fn = dbn.get_finetune_fn(train_x, train_y, batch_size, 'All', dropout)

    print '... finetunning the model'

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

        print 'Fine-tuning epoch %i, model cost %f' % (epoch, numpy.mean(c))

    end_time = time.clock()
    print >> sys.stderr, ('The fine tuning code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time)
                                              / 60.))

    ########################
    #   Test THE MODEL     #
    ########################
    test_score = dbn.get_error_rate(test_x, test_y)
    print(('Optimization complete with best test performance %f %%') %
          (100. - test_score * 100.))


def begin(data=None, pretrain_epoch=100, pretrain_lr=0.01,
          finetune_epoch=100, finetune_lr=0.01, dropout=0.2,
          k=1, isPCD=0, model_structure=[10], batch_size=2):
    """
    main training function
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
    train_y = T.cast(toSharedX(train_y, borrow=True), 'int32')

    Tp.Print('... building symbolic model')
    dbn = DBN(n_ins=num_of_dims, n_outs=num_of_classes,
              hidden_layers_sizes=model_structure, isPCD=isPCD)

    Tp.Print('... linking symbolic model')
    pretraining_fns = dbn.get_pretrain_fns(train_set_x=train_x,
                                           batch_size=batch_size,
                                           k=k)

    Tp.Print('... pre-training the model')
    start_time = time.clock()
    for epoch in xrange(pretrain_epoch):
        ## Pre-train layer-wise
        c = []
        for i in xrange(dbn.n_layers):
            # go through pretraining epochs
            for batch_index in xrange(n_train_batches):
                c.append(pretraining_fns[i](index=batch_index,
                                            lr=pretrain_lr))
        Tp.Print('Pre-training epoch %i, model cost %f' % (epoch, numpy.mean(c)))

    end_time = time.clock()
    Tp.Print('The pretraining code for file ' +
             os.path.split(__file__)[1] +
             ' ran for %.2fm' % ((end_time - start_time) / 60.))


    ########################
    # FINETUNING THE MODEL #
    ########################
    Tp.Print('... getting the finetuning functions')
    finetune_fn = dbn.get_finetune_fn(train_x, train_y, batch_size, 'All', dropout)

    Tp.Print('... finetunning the model')

    test_score = 0.

    start_time = time.clock()
    # go through the each batch
    for epoch in xrange(finetune_epoch):
        ## Pre-train layer-wise
        c = []
        for batch_index in xrange(n_train_batches):
            c.append(finetune_fn(index=batch_index, lr=finetune_lr))
        Tp.Print('Fine-tuning epoch %i, model cost %f' % (epoch, numpy.mean(c)))
        test_score = numpy.mean(c)

    end_time = time.clock()
    Tp.Print(('Optimization complete with best test performance %f %%') %
             (100. - test_score * 100.))
    Tp.Print('The fine tuning code for file ' +
             os.path.split(__file__)[1] +
             ' ran for %.2fm' % ((end_time - start_time)
                                 / 60.))

    Tp.Print('... testing the model')
    test_x, test_y = data['test']
    err = dbn.get_error_rate(test_x, test_y)
    return err


def kfold_train(pretrain_epoch=100, pretrain_lr=0.01,
                finetune_epoch=100, finetune_lr=0.01,
                k=1, isPCD=0, model_structure=[100, 10], dropout=0.2,
                batch_size=20, num_folds=5, normalize_idx=0):
    """
    kfold_train(): training on 5 fold cross validation
    """

    """
    1) load dataset
    """
    from sklearn.datasets import load_digits

    dataset = load_digits()

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
                              pretrain_epoch=pretrain_epoch,
                              pretrain_lr=pretrain_lr,
                              finetune_epoch=finetune_epoch,
                              finetune_lr=finetune_lr,
                              k=k,
                              dropout=dropout,
                              isPCD=isPCD,
                              model_structure=model_structure,
                              batch_size=batch_size))

    return numpy.mean(mean_err)


if __name__ == '__main__':
    test_dbn_mnist(output_folder='/Volumes/HDD750/home/TEMP/dbn_plots',
                   training_epochs=10,
                   learning_rate=0.05,
                   model_structure=[500],
                   finetune_epoch=10,
                   finetune_lr=0.1,
                   dropout=0.5,
                   plot_weight=True)
