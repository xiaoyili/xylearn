__author__ = 'eric'

import theano
import numpy
import theano.tensor as T

"""
Theano Library related operations
"""


def toSharedX(value, name=None, borrow=True):
    """
    Transform value into a shared variable of type floatX
    """
    return theano.shared(theano._asarray(value, dtype=theano.config.floatX),
                         name=name,
                         borrow=borrow)


def toFloatX(variable):
    """
    Casts a given variable into dtype config.floatX
    numpy ndarrays will remain numpy ndarrays
    python floats will become 0-D ndarrays
    all other types will be treated as theano tensors
    """
    if isinstance(variable, float):
        return numpy.cast[theano.config.floatX](variable)

    if isinstance(variable, numpy.ndarray):
        return numpy.cast[theano.config.floatX](variable)

    return theano.tensor.cast(variable, theano.config.floatX)


def toConstantX(value):
    """
        Returns a constant of value `value` with floatX dtype
    """
    return theano.tensor.constant(numpy.asarray(value,
                                                dtype=theano.config.floatX))


def multiple_switch(*args):
    """
    .. todo::

        WRITEME properly

    Applies a cascade of ifelse. The output will be a Theano expression
    which evaluates:

    .. code-block:: none

        if args0:
            then arg1
        elif arg2:
            then arg3
        elif arg4:
            then arg5
        ....
    """
    if len(args) == 3:
        return T.switch(*args)
    else:
        return T.switch(args[0],
                        args[1],
                        multiple_switch(*args[2:]))


def shared_normal(num_rows, num_cols, scale=1, theano_rng=None, name=None):
    '''Initialize a matrix shared variable with normally distributed
elements.'''
    return theano.shared(numpy.random.normal(
        scale=scale, size=(num_rows, num_cols)).astype(theano.config.floatX),
                         name=name)


def shared_zeros(*shape):
    '''Initialize a vector shared variable with zero elements.'''
    return theano.shared(numpy.zeros(shape, dtype=theano.config.floatX))


def shared_ones(*shape):
    '''Initialize a vector shared variable with zero elements.'''
    return theano.shared(numpy.ones(shape, dtype=theano.config.floatX))


def argmax(x, axis=1):
    """
    Cannot used as Theano Graph.

        e.g. obj won't be able to be linked
    """
    idx = T.eq(T.transpose(T.eq(T.transpose(x), T.max(x, axis=axis))), 1).nonzero()[1]
    return idx


def n_argmax(x, axis=1):
    """
    numpy implementation
    """

    idx = numpy.where((x.T == x.max(axis)).T == True)[1]

    return idx