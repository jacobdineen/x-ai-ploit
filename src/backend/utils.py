import functools
import time
import logging

def timer(func):
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        start_time = time.perf_counter()  # Start measuring the time
        value = func(*args, **kwargs)     # Call the actual function
        end_time = time.perf_counter()    # Stop measuring the time
        run_time = end_time - start_time  # Calculate runtime
        
        logging.info(f"Finished {func.__name__!r} in {run_time:.4f} seconds.")
        return value

    return wrapper_timer
