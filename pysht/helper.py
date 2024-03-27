import functools
import numpy as np

def shape_decorator(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        func_name = func.__name__
        class_name = func.__qualname__.split('.')[0]
        
        input_shape = [np.shape(arg) for arg in args]
        result = func(*args, **kwargs)
        output_shape = np.shape(result)

        print(f"{class_name}.{func_name}")
        print(f"  Input shape: {input_shape}")
        print(f"  Output shape: {output_shape}")

        return result

    return wrapper