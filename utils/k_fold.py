__author__ = 'eric'

import numbers

import numpy as np


"""
extract from sklearn

http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.KFold.html

"""


def check_random_state(seed):
    """Turn seed into a np.random.RandomState instance

    If seed is None, return the RandomState singleton used by np.random.
    If seed is an int, return a new RandomState instance seeded with seed.
    If seed is already a RandomState instance, return it.
    Otherwise raise ValueError.
    """
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, (numbers.Integral, np.integer)):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError('%r cannot be used to seed a numpy.random.RandomState'
                     ' instance' % seed)


def _validate_kfold(k, n_samples):
    if k <= 0:
        raise ValueError("Cannot have number of folds k below 1.")
    if k > n_samples:
        raise ValueError("Cannot have number of folds k=%d greater than"
                         " the number of samples: %d." % (k, n_samples))


class KFold(object):
    """K-Folds cross validation iterator.

    Provides train/test indices to split data in train test sets. Split
    dataset into k consecutive folds (without shuffling).

    Each fold is then used a validation set once while the k - 1 remaining
    fold form the training set.

    Parameters
    ----------
    n : int
        Total number of elements.

    n_folds : int, default=3
        Number of folds.

    indices : boolean, optional (default True)
        Return train/test split as arrays of indices, rather than a boolean
        mask array. Integer indices are required when dealing with sparse
        matrices, since those cannot be indexed by boolean masks.

    shuffle : boolean, optional
        Whether to shuffle the data before splitting into batches.

    random_state : int or RandomState
            Pseudo number generator state used for random sampling.

    Examples
    --------
    >>> from sklearn import cross_validation
    >>> X = np.array([[1, 2], [3, 4], [1, 2], [3, 4]])
    >>> y = np.array([1, 2, 3, 4])
    >>> kf = cross_validation.KFold(4, n_folds=2)
    >>> len(kf)
    2
    >>> print(kf)
    sklearn.cross_validation.KFold(n=4, n_folds=2)
    >>> for train_index, test_index in kf:
    ...    print("TRAIN:", train_index, "TEST:", test_index)
    ...    X_train, X_test = X[train_index], X[test_index]
    ...    y_train, y_test = y[train_index], y[test_index]
    TRAIN: [2 3] TEST: [0 1]
    TRAIN: [0 1] TEST: [2 3]

    Notes
    -----
    The first n % n_folds folds have size n // n_folds + 1, other folds have
    size n // n_folds.

    See also
    --------
    StratifiedKFold: take label information into account to avoid building
    folds with imbalanced class distributions (for binary or multiclass
    classification tasks).
    """

    def __init__(self, n, n_folds=3, indices=True, shuffle=False,
                 random_state=None):

        _validate_kfold(n_folds, n)
        random_state = check_random_state(random_state)

        if abs(n - int(n)) >= np.finfo('f').eps:
            raise ValueError("n must be an integer")
        self.n = int(n)
        if abs(n_folds - int(n_folds)) >= np.finfo('f').eps:
            raise ValueError("n_folds must be an integer")
        self.n_folds = int(n_folds)
        self.indices = indices
        self.idxs = np.arange(n)
        if shuffle:
            random_state.shuffle(self.idxs)

    def __iter__(self):
        n = self.n
        n_folds = self.n_folds
        fold_sizes = (n // n_folds) * np.ones(n_folds, dtype=np.int)
        fold_sizes[:n % n_folds] += 1
        current = 0
        if self.indices:
            ind = np.arange(n)
        for i, fold_size in enumerate(fold_sizes):
            test_index = np.zeros(n, dtype=np.bool)
            start, stop = current, current + fold_size
            test_index[self.idxs[start:stop]] = True
            train_index = np.logical_not(test_index)
            if self.indices:
                train_index = ind[train_index]
                test_index = ind[test_index]
            current = stop
            yield train_index, test_index

    def __repr__(self):
        return '%s.%s(n=%i, n_folds=%i)' % (
            self.__class__.__module__,
            self.__class__.__name__,
            self.n,
            self.n_folds,
        )

    def __len__(self):
        return self.n_folds


if __name__ == "__main__":
    import KFold

    kf = KFold(100, 10)
    for train_index, test_index in kf:
        print("TRAIN:", train_index, "TEST:", test_index)
        # X_train, X_test = X[train_index], X[test_index]
        # y_train, y_test = y[train_index], y[test_index]

        # print shuffle_mat(test, 1)