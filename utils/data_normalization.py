__author__ = 'eric'

import numpy
from numpy import dot, sqrt, diag
from numpy.linalg import eigh

"""
data manipulations. All data will be changed after the following operations
"""


def rescale(np_mat, eps=1e-8, axis=0):
    """
    feature scaling, column-wise
    scale the range in [0, 1]
    """
    f_max = numpy.max(np_mat, axis=axis)
    f_min = numpy.min(np_mat, axis=axis)
    return (np_mat - f_min) / (f_max - f_min + eps)


def rescale_global(np_mat, eps=1e-8):
    """
    Scales all values in the ndarray ndar to be between 0 and 1
    """
    np_mat = np_mat.copy()
    np_mat -= np_mat.min()
    np_mat *= 1.0 / (np_mat.max() + eps)
    return np_mat


def standardize(np_mat):
    """
    Z normalization
    X subtracting mean, divide by standard variation
    """
    mu = numpy.mean(np_mat, axis=0)
    sigma = numpy.std(np_mat, axis=0)
    return (np_mat - mu) / (sigma + 1e-8)


def whiten(X, fudge=1E-18):
    # the matrix X should be observations-by-components

    # get the covariance matrix
    Xcov = dot(X.T, X)

    # eigenvalue decomposition of the covariance matrix
    d, V = eigh(Xcov)

    # a fudge factor can be used so that eigenvectors associated with
    # small eigenvalues do not get overamplified.
    D = diag(1. / sqrt(d + fudge))

    # ensure no NaN
    D = numpy.nan_to_num(D)
    # whitening matrix, Can be return!!
    W = dot(dot(V, D), V.T)

    # multiply by the whitening matrix
    X = dot(X, W)

    return X


if __name__ == "__main__":
    test = numpy.array([[10., 0., 0., 0.],
                        [0., 1., 0., 0.],
                        [0., 0., 1., 0.],
                        [0., 0., 1., 0.],
                        [0., 0., 0., 1.]])

    print whiten(test)
