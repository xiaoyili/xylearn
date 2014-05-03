__author__ = 'eric'

"""
.. todo::

    WRITEME
"""
import theano
from theano import tensor
from theano.ifelse import ifelse


def linear_cg(fn, params, tol=1e-3, max_iters=1000, floatX=None):
    """
    Minimizes a POSITIVE DEFINITE quadratic function via linear conjugate
    gradient using the R operator to avoid explicitly representing the Hessian.

    If you have several variables, this is cheaper than Newton's method, which
    would need to invert the Hessian. It is also cheaper than standard linear
    conjugate gradient, which works with an explicit representation of the
    Hessian. It is also cheaper than nonlinear conjugate gradient which does a
    line search by repeatedly evaluating f.

    Parameters
    ----------
    f : theano_like
        A theano expression which is quadratic with POSITIVE DEFINITE hessian \
        in x
    x : list
        List of theano shared variables that influence f

    tol : float
        Minimization halts when the norm of the gradient is smaller than tol

    Returns
    -------
    rval : theano_like
        The solution in form of a symbolic expression (or list of \
        symbolic expressions)


    See Also
    --------
        http://en.wikipedia.org/wiki/Conjugate_gradient_method

        (This reference describes linear CG but not converting it to use
        the R operator instead of an explicit representation of the Hessian)
    """
    provided_as_list = True
    if not isinstance(params, (list, tuple)):
        params = [params]
        provided_as_list = False

    n_params = len(params)

    def loop(rsold, *args):
        ps = args[:n_params]
        rs = args[n_params:2 * n_params]
        xs = args[2 * n_params:]

        Aps = []
        for param in params:
            rval = tensor.Rop(tensor.grad(fn, param), params, ps)
            if isinstance(rval, (list, tuple)):
                Aps.append(rval[0])
            else:
                Aps.append(rval)
        alpha = rsold / sum((x * y).sum() for x, y in zip(Aps, ps))
        xs = [x - alpha * p for x, p in zip(xs, ps)]
        rs = [r - alpha * Ap for r, Ap in zip(rs, Aps)]
        rsnew = sum((r * r).sum() for r in rs)
        ps = [r + rsnew / rsold * p for r, p in zip(rs, ps)]
        return [rsnew] + ps + rs + xs, theano.scan_module.until(rsnew < tol)

    r0s = tensor.grad(fn, params)
    if not isinstance(r0s, (list, tuple)):
        r0s = [r0s]
    p0s = [x for x in r0s]
    x0s = params
    rsold = sum((r * r).sum() for r in r0s)
    outs, updates = theano.scan(loop,
                                outputs_info=[rsold] + p0s + r0s + x0s,
                                n_steps=max_iters,
                                name='linear_conjugate_gradient')
    fxs = outs[1 + 2 * n_params:]
    fxs = [ifelse(rsold < tol, x0, x[-1]) for x0, x in zip(x0s, fxs)]
    if not provided_as_list:
        return fxs[0]
    else:
        return fxs



from theano import config
import numpy
import warnings

import scipy.linalg
import time

def test_linear_cg():
    rng = numpy.random.RandomState([1,2,3])
    n = 5
    M = rng.randn(2*n,n)
    M = numpy.dot(M.T,M).astype(config.floatX)
    b = rng.randn(n).astype(config.floatX)
    c = rng.randn(1).astype(config.floatX)[0]
    x = theano.tensor.vector('x')
    f = 0.5 * tensor.dot(x,tensor.dot(M,x)) - tensor.dot(b,x) + c
    sol = linear_cg(f,[x])

    print sol
    fn_sol = theano.function([x], sol)

    start = time.time()

    data = rng.randn(n).astype(config.floatX)
    sol  = fn_sol(data)[0]

    my_lcg = time.time() -start

    eval_f = theano.function([x],f)
    cgf = eval_f(sol)
    print "conjugate gradient's value of f:", str(cgf), 'time (s)', my_lcg
    spf = eval_f( scipy.linalg.solve(M,b) )
    print "scipy.linalg.solve's value of f: "+str(spf)

    abs_diff = abs(cgf - spf)
    if not (abs_diff < 1e-5):
        raise AssertionError("Expected abs_diff < 1e-5, got abs_diff of " +
                str(abs_diff))


if __name__ == '__main__':
    test_linear_cg()