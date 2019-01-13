# -*- coding: utf-8 -*-
# https://ys-l.github.io/posts/2015/10/03/processifying-bulky-functions-in-python/

import os
import sys
import traceback
from functools import wraps
from multiprocessing import Process, Queue


def processify(func):
    '''Decorator to run a function as a process.
    Be sure that every argument and the return value is *pickable*.
    The created process is joined, so the code does not run in parallel.
    '''

    def process_func(q, *args, **kwargs):
        try:
            ret = func(*args, **kwargs)
        except Exception:
            ex_type, ex_value, tb = sys.exc_info()
            error = ex_type, ex_value, ''.join(traceback.format_tb(tb))
            ret = None
        else:
            error = None

        q.put((ret, error))

    # register original function with different name
    # in sys.modules so it is pickable
    process_func.__name__ = func.__name__ + 'processify_func'
    setattr(sys.modules[__name__], process_func.__name__, process_func)

    @wraps(func)
    def wrapper(*args, **kwargs):
        q = Queue()
        p = Process(target=process_func, args=[q] + list(args), kwargs=kwargs)
        p.start()
        ret, error = q.get()
        # p.join()     #Not join otherwise Blocking the main process

        if error:
            ex_type, ex_value, tb_str = error
            message = '%s (in subprocess)\n%s' % (ex_value.message, tb_str)
            raise ex_type(message)

        return ret
    return wrapper


##################################################################
@processify
def test_function():
    import time
    print "Waiting...."
    time.sleep(15)
    return "External Process is laucnched, PID", os.getpid()


@processify
def test_deadlock():
    return range(30000)


@processify
def test_exception():
    raise RuntimeError('xyz')


def test():
    print "Current PID", os.getpid()
    print test_function()
    print len(test_deadlock())
    print "Error test:"
    test_exception()

if __name__ == '__main__':
    test()



##########
'''
@processify
def work():
    """Get things done here"""
    import numpy as np
    np.random.rand(10,2) + np.random.rand(10,20)
    return np.random.rand(10,2)

if __name__ == '__main__':
    work()


'''





