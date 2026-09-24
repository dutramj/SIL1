import time
from functools import wraps
import logging
import atexit
import os

logger = logging.getLogger(__name__)

# Global registry of statistics
_execution_stats = {}
_execution_stats_registered = False


def time_execution_stats(func):
    global _execution_stats_registered

    if os.environ.get('VAHSIM_PERFORMANCE_DECORATOR_STATS_ENABLE') != '1':
        return func

    # Register exit hook only once
    if not _execution_stats_registered:
        atexit.register(print_execution_stats)
        _execution_stats_registered = True

    key = f"{func.__module__}.{func.__qualname__}"

    if key not in _execution_stats:
        _execution_stats[key] = {
            "calls": 0,
            "total_time": 0.0,
        }

    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()

        result = func(*args, **kwargs)

        elapsed = time.perf_counter() - start

        stats = _execution_stats[key]
        stats["calls"] += 1
        stats["total_time"] += elapsed

        return result

    return wrapper


def print_execution_stats():
    if not _execution_stats:
        logger.debug("No execution statistics collected.")
        return

    logger.debug("Execution time statistics:")

    sorted_stats = sorted(
        _execution_stats.items(),
        key=lambda item: item[1]["total_time"],
        reverse=True
    )

    for func_name, stats in sorted_stats:
        calls = stats["calls"]
        total = stats["total_time"]
        avg = total / calls if calls else 0.0

        logger.debug(
            f"{func_name} | total: {total:.6f}s | avg: {avg:.6f}s | calls: {calls}"
        )


def time_execution(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        
        result = func(*args, **kwargs)
        
        end_time = time.perf_counter()
        elapsed = end_time - start_time

        logger.debug(f"Function '{func.__name__}' executed in {elapsed:.6f} seconds")

        return result
    return wrapper


if __name__ == '__main__':

    logging.basicConfig(level=logging.DEBUG)
    
    @time_execution
    def say_hello(name):
        time.sleep(1)  # Simulate work
        print(f"Hello, {name}!")

    @time_execution
    def add(a, b):
        time.sleep(0.5)  # Simulate work
        return a + b

    # Example usage
    say_hello("Guilherme")
    print(add(3, 5))
