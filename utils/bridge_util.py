__author__ = 'eric'

import os

import numpy

from pymatbridge import Matlab


class MatlabCaller:
    """
    bridging class with matlab
    """

    def __init__(self, port=14001, id=None):
        if id is None:
            id = numpy.random.RandomState(1234)

        # Initialise MATLAB
        self.mlab = Matlab(port=port, id=id)

    def start(self):
        # Start the server
        self.mlab.start()

    def stop(self):
        self.mlab.stop()
        os.system('pkill MATLAB')


    def call_fn(self, path=None, params=None):
        """
        path: the path to your .m function
        params: dictionary contains all parameters
        """
        assert path is not None
        assert isinstance(params, dict)

        return self.mlab.run(path, params)['result']


if __name__ == "__main__":
    pass

