import functools
import numpy as np
import cupy as cp

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


def timing_decorator(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        tkey = func.__name__.replace('___', ' ').replace('__', '-').replace('_', '')
        args[0].timer.reset()
        cp.cuda.runtime.deviceSynchronize()
        _ = func(*args, **kwargs)
        cp.cuda.runtime.deviceSynchronize()
        args[0].timer.add(tkey)
        print(15*"- "+"Timing {}: {:.3f} seconds".format(tkey, args[0].timer.keys[tkey]) + 15*"- "+"\n")
        return _
    return wrapper


def debug_decorator(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if not args[0].debug:
            return func(*args, **kwargs)
        res = func(*args, **kwargs)
        buff = []
        for re in res:
            print('appending item')
            if type(re) == cp.ndarray:
                buff.append(re.get())
            elif type(re) == np.ndarray:
                buff.append(re)
        args[0].ret.update({func.__name__.replace('___', ' ').replace('__', '-').replace('_', ''): np.array(buff)})
        return res
    return wrapper