__author__ = 'eric'


class BaseOptimizer(object):
    """
    Basic abstract class for computing parameter updates of a model.
    """

    def updates(self):
        """Return symbolic updates to apply."""
        raise NotImplementedError()