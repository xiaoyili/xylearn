import time
import sys
import argparse

import theano
import theano.tensor as T
import numpy as np


def NMFnumpy(X, r, iterations, H=None, W=None):
    rng = np.random
    n = np.size(X, 0)
    m = np.size(X, 1)
    if (H is None):
        H = rng.random((r, m)).astype(theano.config.floatX)
    if (W is None):
        W = rng.random((n, r)).astype(theano.config.floatX)

    for i in range(0, iterations):
        #print np.linalg.norm(X-np.dot(W,H))
        H = H * (np.dot(W.T, X) / np.dot(np.dot(W.T, W), H))
        W = W * (np.dot(X, H.T) / np.dot(np.dot(W, H), H.T))

    return W, H


def NMF(X, r, iterations, H=None, W=None, rng=None):
    #     rng = np.random
    n = np.size(X, 0)
    m = np.size(X, 1)
    if (H is None):
        H = rng.random((r, m)).astype(theano.config.floatX)
    if (W is None):
        W = rng.random((n, r)).astype(theano.config.floatX)

    tX = theano.shared(X.astype(theano.config.floatX), name="X")
    tH = theano.shared(H, name="H")
    tW = theano.shared(W, name="W")
    # tE = T.sqrt(((tX - T.dot(tW, tH)) ** 2).sum())

    trainH = theano.function(
        inputs=[],
        outputs=[],
        updates={tH: tH * ((T.dot(tW.T, tX)) / (T.dot(T.dot(tW.T, tW), tH)))},
        name="trainH")
    trainW = theano.function(
        inputs=[],
        outputs=[],
        updates={tW: tW * ((T.dot(tX, tH.T)) / (T.dot(tW, T.dot(tH, tH.T))))},
        name="trainW")

    for i in range(0, iterations):
        #         print np.linalg.norm(X-np.dot(tW.get_value(),tH.get_value()))
        trainH();
        trainW();

    return tW.get_value(), tH.get_value()


def NMF2(X, r, iterations, H=None, W=None, rng=None):
    #     rng = np.random
    n = np.size(X, 0)
    m = np.size(X, 1)
    if (H is None):
        H = rng.random((r, m)).astype(theano.config.floatX)
    if (W is None):
        W = rng.random((n, r)).astype(theano.config.floatX)

    tX = theano.shared(X.astype(theano.config.floatX), name="X")
    tH = theano.shared(H, name="H")
    tW = theano.shared(W, name="W")
    cost = ((tX - T.dot(tW, tH)) ** 2).sum()

    gH, gW = T.grad(cost, [tH, tW]);

    trainH = theano.function(
        inputs=[],
        outputs=[],
        updates={tH: tH - gH},
        name="trainH")

    trainW = theano.function(
        inputs=[],
        outputs=[],
        updates={tW: tW - gW},
        name="trainW")

    for i in range(0, iterations):
        #         print np.linalg.norm(X-np.dot(tW.get_value(),tH.get_value()))
        trainH();
        trainW();

    return tW.get_value(), tH.get_value()


def NMF3(X, r, iterations, H=None, W=None, rng=None):
    #     rng = np.random
    n = np.size(X, 0)
    m = np.size(X, 1)
    if (H is None):
        H = rng.random((r, m)).astype(theano.config.floatX)
    if (W is None):
        W = rng.random((n, r)).astype(theano.config.floatX)

    tX = theano.shared(X.astype(theano.config.floatX), name="X")
    tH = theano.shared(H, name="H")
    tW = theano.shared(W, name="W")
    # tE = T.sqrt(((tX - T.dot(tW, tH)) ** 2).sum())

    trainH = theano.function(
        inputs=[],
        outputs=[],
        updates={tH: tH - T.dot(tW.T, (T.dot(tW, tH) - tX))},
        name="trainH")
    trainW = theano.function(
        inputs=[],
        outputs=[],
        updates={tW: tW - T.dot((T.dot(tW, tH) - tX), tH.T)},
        name="trainW")

    for i in range(0, iterations):
        #         print np.linalg.norm(X-np.dot(tW.get_value(),tH.get_value()))
        trainH();
        trainW();

    return tW.get_value(), tH.get_value()


def NMF4(X, r, iterations, H=None, W=None, rng=None):
    #     rng = np.random
    n = np.size(X, 0)
    m = np.size(X, 1)
    if (H is None):
        H = rng.random((r, m)).astype(theano.config.floatX)
    if (W is None):
        W = rng.random((n, r)).astype(theano.config.floatX)

    tX = theano.shared(X.astype(theano.config.floatX), name="X")
    tH = theano.shared(H, name="H")
    tW = theano.shared(W, name="W")
    cost = T.sqrt(((tX - T.dot(tW, tH)) ** 2).sum()) + abs(tW).sum()

    gW, gH = T.grad(cost, [tW, tH])

    trainH = theano.function(
        inputs=[],
        outputs=[],
        updates={tH: tH - gH},
        name="trainH")
    trainW = theano.function(
        inputs=[],
        outputs=[],
        updates={tW: tW - gW},
        name="trainW")

    for i in range(0, iterations):
        #         print np.linalg.norm(X-np.dot(tW.get_value(),tH.get_value()))
        trainH();
        trainW();

    return tW.get_value(), tH.get_value()


def main(argv):
    if not len(argv):
        print 'Please check usage by: \'-h\' or \'--help\' '
        return

    parser = argparse.ArgumentParser(description='NMF training usage:')
    parser.add_argument('-S', type=int, help='the size of test matrix', nargs='+', default=[100, 100])
    parser.add_argument('-H', type=int, help='the number of hidden reason', nargs='+', default=[10])
    parser.add_argument('-I', type=int, help='the number of iteration', nargs='+', default=[100])
    args = parser.parse_args()

    it = args.I[0]
    r = args.H[0]

    print '... Generating Random Matrix'
    rng = np.random
    Hi = rng.random((r, args.S[1])).astype(theano.config.floatX)

    Wi = rng.random((args.S[0], r)).astype(theano.config.floatX)
    X = rng.random((args.S[0], args.S[1])).astype(theano.config.floatX)

    print '... Begin Calculation'
    rng = np.random
    t0 = time.time()
    W, H = NMF(X, r, it, Hi, Wi, rng)
    t1 = time.time()
    print "Time taken by Theano : ", t1 - t0
    print " --- "
    t0 = time.time()
    W, H = NMF2(X, r, it, Hi, Wi, rng)
    t1 = time.time()
    print "Time taken by Theano auto: ", t1 - t0
    print " --- "
    t0 = time.time()
    W, H = NMF3(X, r, it, Hi, Wi, rng)
    t1 = time.time()
    print "Time taken by L1 : ", t1 - t0
    print " --- "
    t0 = time.time()
    W, H = NMF4(X, r, it, Hi, Wi, rng)
    t1 = time.time()
    print "Time taken by L1 auto: ", t1 - t0


if __name__ == "__main__":
    main(sys.argv[1:])
