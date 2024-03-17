import time
from datetime import timedelta
from typing import Callable


def measure_time(func: Callable) -> Callable:
    """
    A decorator that measures the time a function takes to run.
    """

    def wrapper(*args, **kwargs):
        print("\n------------------------------------")
        print('Started function "{}"!\n'.format(func.__name__))
        t1 = time.time()
        val = func(*args, **kwargs)
        t2 = time.time() - t1
        print('\nFunction "{}" finished!'.format(func.__name__))
        print(f"Function ran for: {timedelta(seconds=t2)}")
        print("------------------------------------\n")
        return val

    return wrapper
