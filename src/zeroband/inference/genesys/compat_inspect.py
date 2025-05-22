# monkey patch to make pyext usable with python 3.11

import inspect
from collections import namedtuple

if not hasattr(inspect, "getargspec"):
    _Full = inspect.getfullargspec
    _ArgSpec = namedtuple("ArgSpec", "args varargs keywords defaults")

    def _getargspec(func):
        fs = _Full(func)
        return _ArgSpec(fs.args, fs.varargs, fs.varkw, fs.defaults)

    inspect.getargspec = _getargspec
