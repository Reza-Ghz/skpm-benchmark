from functools import wraps
from time import time
from typing import Callable


def timing(f, with_args=False):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        if with_args:
            print('func:%r args:[%r, %r] took: %2.4f sec' % \
                  (f.__name__, args, kw, te - ts))
        else:
            print('func:%r took: %2.4f sec' % \
                  (f.__name__, te - ts))
        return result

    return wrap


def timeit(f: Callable):
    ts = time()
    f()
    te = time()
    return round(te - ts, 2)


def percentage(percent: int, total: int):
    return int(percent / 100 * total)
