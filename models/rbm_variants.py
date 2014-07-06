__author__ = 'eric'

import time
import os

import numpy
import theano
import theano.tensor as T
from xylearn.utils.theano_util import toSharedX

from rbm import RBM


class RBM_SPARSE_L2(RBM):
    """
    rbm with KL sparse constrain on hidden activation, plus weight decay
    """

    def _get_cost_update(self, lr=0.1, beta=0.1, gamma=0.0001, s_constrain=0.05,
                         persistent=None, k=1):

        # decide how to initialize persistent chain:
        # for CD, we use the newly generate hidden sample
        # for PCD, we initialize from the old state of the chain
        pre_sigmoid_ph, ph_mean, ph_sample = self._sample_h_given_v(self.x)

        if persistent is None:
            chain_start = ph_sample
        else:
            chain_start = persistent

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


        # sparsity constrain
        hid = ph_mean
        hidden_act = T.mean(hid, axis=1)
        sparse_constrain = T.cast(s_constrain, dtype=theano.config.floatX)
        sparsity_cost = T.sum(sparse_constrain * T.log(sparse_constrain / hidden_act) + (1 - sparse_constrain) * T.log(
            (1 - sparse_constrain) / (1 - hidden_act)))

        weight_decay = T.sum(T.sqr(self.W))

        cost = T.mean(self.free_energy(self.x)) - T.mean(self.free_energy(chain_end)) + \
               T.cast(beta, dtype=theano.config.floatX) * sparsity_cost + \
               T.cast(gamma, dtype=theano.config.floatX) * weight_decay

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

    def get_train_fn(self, dataX, batch_size=1, k=1):
        """
        dataX: theano shared data

        dataY: theano shared label
        """
        learning_rate = T.scalar('lr')
        Beta = T.scalar('beta')
        Gamma = T.scalar('gamma')
        Sparseness = T.scalar('sparseness')

        cost, updates = self._get_cost_update(lr=learning_rate,
                                              beta=Beta,
                                              gamma=Gamma,
                                              s_constrain=Sparseness,
                                              k=k)

        index = T.lscalar('index')

        fn = theano.function(inputs=[index,
                                     theano.Param(learning_rate, default=0.01),
                                     theano.Param(Beta, default=0.1),
                                     theano.Param(Gamma, default=0.0001),
                                     theano.Param(Sparseness, default=0.05)],
                             outputs=cost,
                             updates=updates,
                             givens={self.x: dataX[index * batch_size:(index + 1) * batch_size]},
                             name='train_rbm_S_L2')
        return fn


class RBM_L1(RBM):
    """
    rbm with L1 norm on W
    """

    def _get_cost_update(self, lr=0.1, beta=0.1, persistent=None, k=1):

        pre_sigmoid_ph, ph_mean, ph_sample = self._sample_h_given_v(self.x)

        if persistent is None:
            chain_start = ph_sample
        else:
            chain_start = persistent

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

        L1_cost = T.sum(abs(self.W))

        cost = T.mean(self.free_energy(self.x)) - T.mean(
            self.free_energy(chain_end)) + beta * L1_cost

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

    def get_train_fn(self, dataX, batch_size=1, k=1):
        """
        dataX: theano shared data

        dataY: theano shared label
        """
        learning_rate = T.scalar('lr')
        Beta = T.scalar('beta')

        cost, updates = self._get_cost_update(lr=learning_rate,
                                              beta=Beta,
                                              k=k)

        index = T.lscalar('index')

        fn = theano.function(inputs=[index,
                                     theano.Param(learning_rate, default=0.01),
                                     theano.Param(Beta, default=0.001)],
                             outputs=cost,
                             updates=updates,
                             givens={self.x: dataX[index * batch_size:(index + 1) * batch_size]},
                             name='train_rbm_L1')
        return fn


class RBM_Orthogonal(RBM):
    """
    rbm with orthogonal norm on W
    """

    def _get_cost_update(self, lr=0.1, beta=0.1, persistent=None, k=1):

        pre_sigmoid_ph, ph_mean, ph_sample = self._sample_h_given_v(self.x)

        if persistent is None:
            chain_start = ph_sample
        else:
            chain_start = persistent

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

        # (784, 400) x (400, 784)
        orthogonal_cost = T.sum(T.sqr(T.dot(self.W.T, self.W) - T.eye(self.n_hidden, self.n_hidden)))
        # orthogonal_cost = T.sum(T.sqr(T.dot(self.W.T, self.W)))

        cost = T.mean(self.free_energy(self.x)) - T.mean(
            self.free_energy(chain_end)) + beta * orthogonal_cost

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

    def get_train_fn(self, dataX, batch_size=1, k=1):
        """
        dataX: theano shared data

        dataY: theano shared label
        """
        learning_rate = T.scalar('lr')
        Beta = T.scalar('beta')

        cost, updates = self._get_cost_update(lr=learning_rate,
                                              beta=Beta,
                                              k=k)

        index = T.lscalar('index')

        fn = theano.function(inputs=[index,
                                     theano.Param(learning_rate, default=0.01),
                                     theano.Param(Beta, default=0.001)],
                             outputs=cost,
                             updates=updates,
                             givens={self.x: dataX[index * batch_size:(index + 1) * batch_size]},
                             name='train_rbm_orthogonal')
        return fn


class FastWeightRBM(RBM):
    '''
    fast weight pcd RBM with L2 penalty
    '''

    def __init__(self, input=None, n_visible=784, n_hidden=500,
                 W=None, hbias=None, vbias=None, isPCD=0,
                 W_fast=None, hbias_fast=None, vbias_fast=None,
                 numpy_rng=None, theano_rng=None, aFunc=T.nnet.sigmoid):
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

        super(FastWeightRBM, self).__init__(input=input, n_visible=n_visible, n_hidden=n_hidden,
                                            W=W, hbias=hbias, vbias=vbias, numpy_rng=numpy_rng, aFunc=aFunc,
                                            theano_rng=theano_rng, isPCD=isPCD)

        if W_fast is None:
            W_fast = theano.shared(value=numpy.zeros((self.n_visible, self.n_hidden),
                                                     dtype=theano.config.floatX),
                                   name='W_fast', borrow=True)

        if hbias_fast is None:
            hbias_fast = theano.shared(value=numpy.zeros(n_hidden,
                                                         dtype=theano.config.floatX),
                                       name='hbias_fast', borrow=True)

        if vbias_fast is None:
            vbias_fast = theano.shared(value=numpy.zeros(n_visible,
                                                         dtype=theano.config.floatX),
                                       name='vbias_fast', borrow=True)

        self.W_fast = W_fast
        self.hbias_fast = hbias_fast
        self.vbias_fast = vbias_fast
        # **** WARNING: It is not a good idea to put things in this list
        # other than shared variables created in this function.
        self.params = [self.W, self.hbias, self.vbias]
        self.params_fast = [self.W_fast, self.hbias_fast, self.vbias_fast]

    def free_energy(self, v_sample):
        ''' Function to compute the free energy '''
        wx_b = T.dot(v_sample, self.W + self.W_fast) + self.hbias + self.hbias_fast
        vbias_term = T.dot(v_sample, self.vbias + self.vbias_fast)
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

        pre_sigmoid_activation = T.dot(vis, self.W + self.W_fast) + self.hbias + self.hbias_fast
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
        pre_sigmoid_activation = T.dot(hid, self.W.T + self.W_fast.T) + self.vbias + self.vbias_fast
        return [pre_sigmoid_activation, self.aFunc(pre_sigmoid_activation)]

    def _get_cost_update(self, lr=0.01, persistent=None, k=1):

        pre_sigmoid_ph, ph_mean, ph_sample = self._sample_h_given_v(self.x)

        # compute positive phase

        if persistent is None:
            chain_start = ph_sample
        else:
            chain_start = persistent

        batch_size = chain_start.shape[0]

        # perform actual negative phase
        [pre_sigmoid_nvs, nv_means, nv_samples,
         pre_sigmoid_nhs, nh_means, nh_samples], updates = \
            theano.scan(self._gibbs_hvh,
                        # the None are place holders, saying that
                        # chain_start is the initial state corresponding to the
                        # 6th output
                        outputs_info=[None, None, None, None, None, chain_start],
                        n_steps=k)
        # note that we only need the sample at the end of the chain
        chain_end = nv_samples[-1]


        # positive expectation
        wx_b, poshidprobs = self.propup(self.x)
        posprods = T.dot(self.x.T, poshidprobs)
        pos_H = T.sum(poshidprobs)
        pos_V = T.sum(self.x)

        # negative gradient
        wx_b, neghidprobs = self.propup(chain_end)
        negprods = T.dot(chain_end.T, neghidprobs)
        neg_H = T.sum(neghidprobs)
        neg_V = T.sum(chain_end)


        # new gradient
        W_grad = (
            ((posprods - negprods) - 0.0001 * T.cast(numpy.sign(self.W.get_value(borrow=True)),
                                                     dtype=theano.config.floatX)) / \
            batch_size).astype(theano.config.floatX)

        hbias_grad = ((pos_H - neg_H) / batch_size).astype(theano.config.floatX)
        vbias_grad = ((pos_V - neg_V) / batch_size).astype(theano.config.floatX)

        # update rule
        updates[self.W] = self.W + T.cast(lr, dtype=theano.config.floatX) * W_grad
        updates[self.hbias] = self.hbias + T.cast(lr, dtype=theano.config.floatX) * hbias_grad
        updates[self.vbias] = self.vbias + T.cast(lr, dtype=theano.config.floatX) * vbias_grad

        updates[self.W_fast] = self.W_fast * 0.95 + 2 * T.cast(lr, dtype=theano.config.floatX) * W_grad
        updates[self.hbias_fast] = self.hbias_fast * 0.95 + 2 * T.cast(lr, dtype=theano.config.floatX) * hbias_grad
        updates[self.vbias_fast] = self.vbias_fast * 0.95 + 2 * T.cast(lr, dtype=theano.config.floatX) * vbias_grad

        if persistent:
            # Note that this works only if persistent is a shared variable
            updates[persistent] = nh_samples[-1]
            # pseudo-likelihood is a better proxy for PCD
            monitoring_cost = self._get_pseudo_likelihood_cost(updates)
        else:
            # reconstruction cross-entropy is a better proxy for CD
            monitoring_cost = self._get_reconstruction_cost(pre_sigmoid_nvs[-1])

        return monitoring_cost, updates


class PoissonRBM(RBM):
    """
        Constrained poisson RBM.  Uses a poisson distribution for modeling visible word counts
        and the standard conditional Bernoulli distribution for modeling hidden features.  As such,
        P(v_i=n|h) = Ps(v, exp(bv_i + Sum_j(h_j*w_ij)) * N/Z)
        N = Sum_i(v_i)
        Z = Sum_k(exp(bv_k + Sum_j(h_j*w_kj))

        with resulting energy,
        E(v,h) = -Sum_i(bv_i*v_i) + Sum_i(log(v_i!)) - Sum_j(bh_j*h_j) - Sum_ij(v_i*h_j*w_ij)
    """

    def _Ps(self, n, l):
        return T.exp(-l) * (l ** n) / T.gamma(n + 1)

    def free_energy(self, v_sample):
        tmp, h = self.propup(v_sample)
        return -T.dot(v_sample, self.vbias) - T.dot(h, self.hbias) + \
               T.sum((-T.dot(v_sample, self.W) * h + T.gammaln(v_sample + 1)), axis=1)

    def _gibbs_vhv(self, v0_sample):
        wx_b, h1_mean, h1_sample = self._sample_h_given_v(v0_sample)
        wTx_b, tmp = self.propdown(h1_mean)

        # un-normalized poisson rate, L
        L = T.exp(wTx_b)
        # now we normalize it wrt length of wordvector and partition function
        L = L * T.sum(v0_sample, axis=1)[:, numpy.newaxis] / \
            T.sum(L, axis=1)[:, numpy.newaxis]

        # v1_mean gives the probability of seen self.x
        v1_mean = self._Ps(v0_sample, L)
        v1_sample = self.theano_rng.binomial(size=v1_mean.shape,
                                             n=1, p=v1_mean,
                                             dtype=theano.config.floatX)

        return [wTx_b, v1_mean, v1_sample]

    def _get_cost_update(self, lr=0.1, persistent=None, k=1):
        # compute positive phase
        chain_start = self.x

        [pre_sigmoid_nvs, nv_means, nv_samples], updates = \
            theano.scan(self._gibbs_vhv,
                        outputs_info=[None, None, chain_start],
                        n_steps=k)

        # determine gradients on RBM parameters
        # note that we only need the sample at the end of the chain
        chain_end = nv_samples[-1]

        # Contrastive Loss, different from AE(cross entropy loss)
        cost = T.mean(self.free_energy(chain_start)) - T.mean(
            self.free_energy(chain_end))

        # We must not compute the gradient through the gibbs sampling
        gparams = T.grad(cost, self.params, consider_constant=[chain_end])

        # constructs the update dictionary
        for gparam, param in zip(gparams, self.params):
            # make sure that the learning rate is of the right dtype
            updates[param] = param - gparam * T.cast(lr,
                                                     dtype=theano.config.floatX)

        monitoring_cost = self._get_reconstruction_cost(pre_sigmoid_nvs[-1])

        return monitoring_cost, updates

    def reconstruct(self, testX, showSample=0):
        wx_b, v1_mean, v1_sample = self._gibbs_vhv(testX)

        if showSample:
            fn = theano.function([], theano.Out(v1_sample, borrow=True),
                                   name='prbm_recon')
        else:
            fn = theano.function([], theano.Out(v1_mean, borrow=True),
                                   name='prbm_recon')

        return fn()


class TransferRBM(RBM):
    """
    add other cost: energy distance between the model mean w.r.t input rbm

    """

    def __init__(self, input=None, n_visible=784, n_hidden=500, \
                 W=None, hbias=None, vbias=None, targetMean=None,
                 numpy_rng=None, aFunc=T.nnet.sigmoid, isPCD=0,
                 theano_rng=None):
        """
        Transfor RBM, restrict model mean to target mean
        """

        super(TransferRBM, self).__init__(input=input, n_visible=n_visible, n_hidden=n_hidden,
                                          W=W, hbias=hbias, vbias=vbias, numpy_rng=numpy_rng, aFunc=aFunc,
                                          theano_rng=theano_rng, isPCD=isPCD)

        if targetMean is not None:
            self.targetMean = targetMean
            self.OutTargetMean = self.aFunc(T.dot(targetMean, self.W) + self.hbias)


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

        if persistent is None:
            chain_start = ph_sample
        else:
            chain_start = persistent

        # perform actual negative phase
        [pre_sigmoid_nvs, nv_means, nv_samples,
         pre_sigmoid_nhs, nh_means, nh_samples], updates = \
            theano.scan(self._gibbs_hvh,
                        outputs_info=[None, None, None, None, None, chain_start],
                        n_steps=k)
        chain_end = nv_samples[-1]

        # reconstruct cost
        recon_cost = T.mean(self.free_energy(self.x)) - T.mean(
            self.free_energy(chain_end))


        # distance cost
        zero_plate = toSharedX(numpy.zeros_like(self.targetMean),
                               name='zero_plate', borrow=True)
        pre, ph_mean = self.propup(zero_plate)
        pre, pv_mean = self.propdown(ph_mean)

        # energy gap between target model and current model
        dist_cost = T.mean(self.free_energy(self.targetMean)) - T.mean(
            self.free_energy(pv_mean))

        # RMSE distance
        # dist_cost = T.sqrt(T.sum(T.square(self.targetMean - pv_mean)))

        cost = recon_cost + 100 * dist_cost + 0.001 * T.sum(self.W)
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
            monitoring_cost = self._get_reconstruction_cost(updates,
                                                            pre_sigmoid_nvs[-1])

        return monitoring_cost, updates


class GaussianRBM(RBM):
    """

    Gaussian unit Restricted Boltzmann Machine (gRBM)

    To model with/without noise in the visible layer,
    please modify 'propdown()' function

    """

    def __init__(self, input=None, n_visible=784, n_hidden=500,
                 W=None, hbias=None, vbias=None, sigma=None,
                 aFunc=T.nnet.sigmoid, isPCD=0,
                 numpy_rng=None, theano_rng=None):
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
        super(GaussianRBM, self).__init__(input=input, n_visible=n_visible, n_hidden=n_hidden,
                                          W=W, hbias=hbias, vbias=vbias, numpy_rng=numpy_rng, aFunc=aFunc,
                                          theano_rng=theano_rng, isPCD=isPCD)

        if sigma is None:
            self.sigma = theano.shared(theano._asarray(numpy.ones(n_visible), dtype=theano.config.floatX),
                                       name='sigma',
                                       borrow=False)
        else:
            self.sigma = theano.shared(theano._asarray(sigma * numpy.ones(n_visible), dtype=theano.config.floatX),
                                       name='sigma',
                                       borrow=False)

        self.params = [self.W, self.hbias, self.vbias]

    def free_energy(self, v_sample):
        ''' Function to compute the free energy '''
        wx_b, tmp = self.propup(v_sample)  # hid_input

        squared_term = ((self.vbias - v_sample) ** 2.) / (2. * self.sigma)

        return squared_term.sum(axis=1) - T.nnet.softplus(wx_b).sum(axis=1)

    def propup(self, vis):

        pre_sigmoid_activation = T.dot(vis / self.sigma, self.W) + self.hbias
        return [pre_sigmoid_activation, self.aFunc(pre_sigmoid_activation)]

    def _sample_v_given_h(self, h0_sample):

        # wx + b as mu, in other words, pre_sigmoid_v1
        pre_sigmoid_v1, v1_mean = self.propdown(h0_sample)

        v1_sample = self.theano_rng.normal(size=pre_sigmoid_v1.shape,
                                           avg=pre_sigmoid_v1,
                                           std=self.sigma,
                                           dtype=theano.config.floatX)

        return [pre_sigmoid_v1, v1_mean, v1_sample]


"""
TEST CASES
"""


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

    # from rbm import RBM
    from rbm_variants import RBM_Orthogonal as RBM
    # from rbm_variants import PoissonRBM as RBM


    train_x = toSharedX(toy_data, name="toy_data")

    n_vis = train_x.get_value(borrow=True).shape[1]

    n_samples = train_x.get_value(borrow=True).shape[0]

    if batch_size >= n_samples:
        batch_size = n_samples

    n_train_batches = n_samples / batch_size


    # construct the RBM class
    rbm = RBM(n_visible=n_vis, n_hidden=n_hidden, isPCD=isPCD)
    train_fn = rbm.get_train_fn(train_x, batch_size)

    print "... projecting"
    print rbm.project(train_x, hidSample=1)

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

        if numpy.mean(mean_cost) >= 0:
            break

        # W shape is [784 500]
        # Plot filters after each training epoch
        plotting_start = time.clock()
        # Construct image from the weight matrix
        image = PIL.Image.fromarray(tile_raster_images(
            X=rbm.W.get_value(borrow=True).T,
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
    print rbm.project(train_x, hidSample=1)

    print "... reconstructing"
    print rbm.reconstruct(train_x, showSample=1) * train_x.get_value(borrow=True)


def test_rbm_mnist(learning_rate=0.01, training_epochs=10, batch_size=20,
                   n_chains=30, n_samples=5, output_folder=None, isPCD=0,
                   n_hidden=500):
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
        test_rbm_mnist(output_folder='/home/eric/Desktop/rbm_plots')

    """

    assert output_folder is not None

    from rbm_variants import RBM_Orthogonal as RBM
    # from rbm import RBM

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

    print numpy.linalg.matrix_rank(train_x.get_value(borrow=True))

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
        # monitor projected rank
        projection = rbm.project(train_x)
        print 'rank: ' + str(numpy.linalg.matrix_rank(projection))

        # W shape is [784 500]
        # Plot filters after each training epoch
        plotting_start = time.clock()
        # Construct image from the weight matrix
        image = PIL.Image.fromarray(tile_raster_images(
            X=rbm.W.get_value(borrow=True).T,
            img_shape=(28, 28), tile_shape=(20, 20),
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

    #################################
    #     Projecting from the RBM   #
    #################################
    projection = rbm.project(train_x)

    print numpy.linalg.matrix_rank(projection)


if __name__ == '__main__':
    test_rbm_mnist(output_folder='/home/eric/Desktop/rbm_plots',
                   learning_rate=0.01, training_epochs=100, batch_size=20,
                   n_hidden=100, isPCD=0)

    # /home/eric/Desktop/rbm_plots
    # toy_test(output_folder='/Volumes/HDD750/home/TEMP/prbm_plots',
    #          training_epochs=100, learning_rate=0.01,
    #          n_hidden=10)