from functools import wraps
import time

import matplotlib.pyplot as plt

class Timer:    
    def __enter__(self):
        self.start = time.clock()
        return self

    def __exit__(self, *args):
        self.end = time.clock()
        self.interval = self.end - self.start
        print("Executing {} took {:.2} s".format("block", self.interval))



def time_execution(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        t0 = time.time()
        result = f(*args, **kwargs)
        t1 = time.time()
        print("Executing {} took {:.2} s".format(f.__name__, t1 - t0))
        return result
    return wrapper


def show_figure(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        plt.figure()
        result = f(*args, **kwargs)
        plt.show()
        return result
    return wrapper


def save_figure(f, filename):
    @wraps(f)
    def wrapper(*args, **kwargs):
        plt.figure()
        result = f(*args, **kwargs)
        plt.savefig(filename)
        return result
    return wrapper
