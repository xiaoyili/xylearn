__author__ = 'eric'

import sys
import os


def call_counter(reset=0):
    if reset:
        call_counter.counter = 0
    else:
        call_counter.counter = getattr(call_counter, 'counter', 0)
        call_counter.counter += 1

    return call_counter.counter


class TerminalPrinter(object):
    """
    control the terminal print out
    """

    def __init__(self, debug=False, verbose=False):
        self.debug = debug
        self.verbose = verbose
        self.num_iter = -1
        self.stepsize = 10
        self.residual = 0
        self.flag = True

    def set_counter(self, num_iter, stepsize=10):
        self.num_iter = num_iter
        self.flag = True
        if stepsize <= num_iter:
            self.stepsize = stepsize
            self.residual = num_iter % stepsize
        else:
            sys.exit('TerminalPrinter: -- ERROR, stepsize should smaller than the number of iterations')

    def Print(self, s=None, must_print=0):
        """
        when debug is on,
            if verbose mode is on, print everything
            else print s with "must_print" switched on
            p.s. must_print means the one that must be used when debug
        """
        if self.debug:
            if self.verbose or must_print == 1:
                sys.stdout.write(str(s) + '\n')

    def interval_print(self, s, symbol='.', must_print=0):
        if self.debug:
            if self.verbose or must_print == 1:

                if self.stepsize > 1:
                    sys.stdout.write(symbol)

                if self.flag:
                    self.num_iter -= 1
                    if self.num_iter < self.residual:
                        self.stepsize = self.residual
                        self.flag = False

                if call_counter() == self.stepsize:
                    call_counter(reset=1)
                    sys.stdout.write(str(s) + '\n')


if __name__ == "__main__":

    for j in range(10):
        tp = TerminalPrinter(debug=True, verbose=False)

        tp.set_counter(num_iter=100, stepsize=30)
        for i in range(100):
            tp.interval_print('holy', must_print=1)