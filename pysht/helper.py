import functools
import numpy as np

def shape_decorator(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        func_name = func.__name__
        class_name = func.__qualname__.split('.')[0]
        
        input_shapes = [(param_name, np.shape(arg)) for param_name, arg in zip(func.__code__.co_varnames, args)]
        result = func(*args, **kwargs)

        print(f"{class_name}.{func_name}")
        print("  Input shapes:")
        for param_name, shape in input_shapes:
            print(f"    {param_name}: {shape}")

        output_shapes = [np.shape(res) for res in result]
        output_types = [type(res) for res in result]
        print(f"  Output shape: {output_shapes}, types: {output_types}")

        return result

    return wrapper