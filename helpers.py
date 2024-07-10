import tracemalloc
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


def timeit(f: Callable) -> (any, int):
    ts = time()
    res = f()
    te = time()
    return res, round(te - ts, 3)


def memoryit(f: Callable) -> (any, int):
    try:
        # TODO: check pandas and polars memory functions
        # TODO: Why polars doesn't grow as pandas in linear way
        tracemalloc.start()
        res = f()
        return res, round(tracemalloc.get_traced_memory()[1] / 1_048_576, 3)
    finally:
        tracemalloc.clear_traces()
        tracemalloc.stop()


def percentage(percent: int, total: int):
    return int(percent / 100 * total)
