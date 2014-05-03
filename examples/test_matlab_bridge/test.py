from xylearn.utils.bridge_util import MatlabCaller

"""
TODO: not able to load matrix by now. showing JSON serialize problem
"""
mc = MatlabCaller(port=14001)
mc.start()

for i in range(1, 10):
    print mc.call_fn('test.m', {'a': i})

mc.stop()