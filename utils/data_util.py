__author__ = 'eric'

import gzip
import cPickle

import numpy
import numbers

from k_fold import KFold


"""
data will not do any mathematical transformation.
"""


def shuffle_mat(mat, flag=0):
    """
    mat: input numpy matrix
    flag:   0 -- change row order
            1 -- change col order
    """
    if flag == 0:
        numpy.random.shuffle(mat)

    elif flag == 1:
        mat = numpy.transpose(mat)
        numpy.random.shuffle(mat)
        mat = numpy.transpose(mat)

    return mat


def assign_name(variable, anon="anonymous_variable"):
    """
    If variable has a name, returns that name.
    Otherwise, returns anon
    """

    if hasattr(variable, 'name') and variable.name is not None:
        return variable.name

    return anon


def has_nan(mat):
    """
    check if the input matrix contains NaN
    return the number of NaN in the matrix
    """
    return numpy.sum(numpy.array(numpy.isnan(mat), dtype=int))


def label_to_mat(y):
    """
    Transfer a vector of labels to matrix. e.g.

    input: [1,2,3,3,4]

    output:
       [[ 1.  0.  0.  0.]
        [ 0.  1.  0.  0.]
        [ 0.  0.  1.  0.]
        [ 0.  0.  1.  0.]
        [ 0.  0.  0.  1.]]

    """
    n_samples = y.shape[0]

    if len(y.shape) == 2:
        y = numpy.reshape(y, [n_samples, ])

    labelboard = numpy.zeros([n_samples, numpy.max(y)]).astype('int32')

    labelboard[xrange(n_samples), y - 1] = 1

    return labelboard


def write_to_mat(data, filename):
    """
    output to matlab .mat
    """
    import scipy.io as sio

    sio.savemat(filename, {'data': data}, oned_as='row')


def zip_and_pickle(data=None, path=None):
    assert data is not None
    assert path is not None

    fd = gzip.open(path, 'wb')
    cPickle.dump(data, fd)
    fd.close()


def _get_train_test_index(num_of_samples=0, num_of_fold=10):
    """
    helper function.
    get a list of tuple
    """
    res = list()
    kf = KFold(num_of_samples, num_of_fold)
    for train_index, test_index in kf:
        res.append((train_index, test_index))

    return res


def get_train_test(data, label=None, reshape_label=False,
                   fold_id=0, num_fold=10, shuffle=False,
                   useGPU=0):
    """
    data: numpy.ndarray, row represent individual samples
    label: numpy.asarray
    fold_id: default 10-fold, the id is the indicator of a separate share
    useGPU: transfer to theano shared matrix
    """
    num_samples = data.shape[0]

    if label is None:
        label = numpy.zeros([num_samples, ])
    elif len(label.shape) == 1 and reshape_label:
        # reshape a vector into matrix
        label = label_to_mat(label)

    if shuffle:
        from sklearn.utils import shuffle

        data, label = shuffle(data, label)

    idx_list = _get_train_test_index(num_samples, num_fold)

    dataset = dict()

    if useGPU:
        import theano.tensor as T
        from theano_util import toSharedX

        dataset['train'] = toSharedX(data[idx_list[fold_id][0]], borrow=True), \
                           T.cast(toSharedX(label[idx_list[fold_id][0]], borrow=True), 'int32')
        dataset['test'] = toSharedX(data[idx_list[fold_id][1]], borrow=True), \
                          T.cast(toSharedX(label[idx_list[fold_id][1]], borrow=True), 'int32')

    else:
        dataset['train'] = (data[idx_list[fold_id][0]], label[idx_list[fold_id][0]])
        dataset['test'] = (data[idx_list[fold_id][1]], label[idx_list[fold_id][1]])

    return dataset


if __name__ == "__main__":
    # a = numpy.random.randn(10, 2)
    # label = numpy.asarray([1, 2, 3, 4, 1, 2, 3, 4, 5, 4])
    #
    # data = get_train_test(a, label, fold_id=0, useGPU=0)
    #
    # print "----------------------"
    # print data['train'][1]

    a = numpy.asarray([1,0,1,0,1])

    print label_to_mat(a + 1)