__author__ = 'eric'

"""
General data manipulation
"""
import numpy


def get_sub_dict(d, keys):
    """
    Create a sub-dictionary of d with the keys in keys.

    in other words, copy out a small dict from a large dict, given a list of keys

    """
    result = {}
    for key in keys:
        if key in d: result[key] = d[key]
    return result


def safe_update(dict_to, dict_from):
    """
    Like dict_to.update(dict_from), except don't overwrite any keys.
    """
    for key, val in dict(dict_from).iteritems():
        if key in dict_to:
            raise KeyError(key)
        dict_to[key] = val
    return dict_to


def get_union(a, b):
    """
    Does the logic of a union operation without the non-deterministic
    ordering of python sets. e.g.
        >> unorderedUnion([4,2,3,3,4], [1,2,3,3,4])
        >> [4,2,3,1]
    """
    if not isinstance(a, list):
        raise TypeError("Expected first argument to be a list, but got " + str(type(a)))
    assert isinstance(b, list)
    c = []
    for x in a + b:
        if x not in c:
            c.append(x)
    return c


def is_perfect_square(x):
    s = numpy.sqrt(numpy.array(x))
    s = s.astype(int)

    if (s * s).tolist() == x:
        return True
    else:
        return False


def set_precision(x, num_decimal=6):
    from decimal import Decimal, getcontext
    getcontext().prec = num_decimal
    # magic happens, the precision cut only
    # happens when an operation take place
    # so, Decimal(x) / Decimal(1)
    return (Decimal(x) / Decimal(1)).__float__()


def drange(start, end, num_interval):
    """
    evenly divided start~end space into num_interval pieces
    """
    assert start < end
    assert isinstance(num_interval, int)

    from decimal import Decimal, getcontext

    getcontext().prec = 6

    stepsize = Decimal(end - start) / Decimal(num_interval)

    increment = Decimal(start)
    res = list()
    for i in xrange(num_interval):
        res.append(increment.__float__())
        increment += stepsize

    return res


def drange2(start, end, stepsize, toInt=False):
    """
    increment from start to end by stepsize
    """
    assert stepsize < (abs(start - end))

    from decimal import Decimal, getcontext

    getcontext().prec = 6

    increment = Decimal(start)
    stepsize = Decimal(stepsize)
    res = list()
    while increment <= end:
        if toInt:
            res.append(int(increment.__float__()))
        else:
            res.append(increment.__float__())
        increment += stepsize

    return res


def drange3(start, end, stepsize, toInt=False):
    """
    momentum increment from start to end by stepsize
    e.g. drange3(0, 1, 0.1)
        [0.0, 0.1, 0.22, 0.413907, 0.842504]
    """
    assert stepsize < (abs(start - end))

    from decimal import Decimal, getcontext

    getcontext().prec = 6

    momentum = Decimal(2)
    increment = Decimal(start)
    stepsize = Decimal(stepsize)
    res = list()
    while increment <= end:
        if toInt:
            res.append(int(increment.__float__()))
        else:
            res.append(increment.__float__())

        increment += stepsize
        stepsize *= (Decimal(0.6) * momentum)
        momentum = Decimal(momentum + momentum.ln())

    return res


if __name__ == "__main__":
    print drange3(0, 1000, 0.1, toInt=True)

    print set_precision(1000.44321)