from timeit import default_timer as timer
import logging

logger = logging.getLogger(__name__)
def timer_func(func):
    def wrapper(*args, **kwargs):
        t1 = timer()
        result = func(*args, **kwargs)
        t2 = timer()
        logger.info(f'{func.__name__}() executed in {(t2 - t1):.6f}s')
        return result

    return wrapper
