import functools
import os
import cunusht
import numpy as np

from lenspyx.utils_hp import Alm, alm2cl, almxfl, alm_copy


def timing_decorator(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        tkey = func.__name__.replace('___', ' ').replace('__', '-').replace('_', '')
        t0, ti = args[0].timer.reset()
        _ = func(*args, **kwargs)
        args[0].timer.add(tkey)
        args[0].timer.set(t0, ti)
        if args[0].execmode == 'timing' or args[0].execmode == 'debug':
            print(15*"- "+"Timing {}: {:.3f} seconds".format(tkey, args[0].timer.keys[tkey]) + 15*"- "+"\n")
        return _
    return wrapper


def timing_decorator_close(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        tkey = func.__name__.replace('___', ' ').replace('__', '-').replace('_', '')
        t0, ti = args[0].timer.reset()
        args[0].timer.reset_ti()
        args[0].timer.add(tkey)
        _ = func(*args, **kwargs)
        args[0].timer.add_elapsed(tkey)
        if args[0].execmode == 'timing' or args[0].execmode == 'debug':
            print(15*"- "+"Timing {}: {:.3f} seconds".format(tkey, args[0].timer.keys[tkey]) + 15*"- "+"\n")
        args[0].timer.set(t0, ti)
        if args[0].execmode == 'timing':
            args[0].timer.close(args[0].__class__.__name__)
            dirname = os.path.dirname(cunusht.__file__)[:-7]+'/test/benchmark/timings/{}/{}/'.format(args[0].__class__.__name__, func.__name__)
            if not os.path.exists(dirname):
                os.makedirs(dirname)
            args[0].timer.dumpjson(dirname+"lmax{}_epsilon{}_run{:02d}".format(kwargs['lmax'], args[0].epsilon, args[0].runid))
            if args[0].execmode == 'timing' or args[0].execmode == 'debug':
                print(args[0].timer)
                print("::timing:: stored new timing data for lmax {}".format(kwargs['lmax']))
        return _
    return wrapper

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


def debug_decorator(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if not args[0].execmode == 'debug':
            return func(*args, **kwargs)
        res = func(*args, **kwargs)
        buff = res
        args[0].ret.update({func.__name__.replace('___', ' ').replace('__', '-').replace('_', ''): np.array(buff)})
        return res
    return wrapper


def get_spin_raise(s, lmax):
    r"""Response coefficient of spin-s spherical harmonic to spin raising operator.

        :math:`\sqrt{ (l - s) (l + s + 1) }` for abs(s) <= l <= lmax

    """
    ret = np.zeros(lmax + 1, dtype=float)
    ret[abs(s):] = np.sqrt(np.arange(abs(s) -s, lmax - s + 1) * np.arange(abs(s) + s + 1, lmax + s + 2))
    return ret

def get_spin_lower(s, lmax):
    r"""Response coefficient of spin-s spherical harmonic to spin lowering operator.

        :math:`-\sqrt{ (l + s) (l - s + 1) }` for abs(s) <= l <= lmax

    """
    ret = np.zeros(lmax + 1, dtype=float)
    ret[abs(s):] = -np.sqrt(np.arange(s + abs(s), lmax + s + 1) * np.arange(abs(s) - s + 1, lmax - s + 2))
    return ret


def dlm2A(dlm, lmax_dlm, mmax_dlm, geom, nthreads, dclm=None):
    """Returns determinant of magnification matrix corresponding to input deflection field

        Returns:
            determinant of magnification matrix. Array of size input pixelization geometry

    """
    #FIXME splits in band with new offsets
    geom, lmax, mmax = geom, lmax_dlm, mmax_dlm
    dgclm = np.empty((2, dlm.size), dtype=dlm.dtype)
    dgclm[0] = dlm
    dgclm[1] = np.zeros_like(dlm) if dclm is None else dclm
    d2k = -0.5 * get_spin_lower(1, lmax_dlm)  # For k = 12 \eth^{-1} d, g = 1/2\eth 1d
    d2g = -0.5 * get_spin_raise(1, lmax_dlm) #TODO: check the sign of this one
    glms = np.empty((2, dlm.size), dtype=dlm.dtype) # Shear
    glms[0] = almxfl(dgclm[0], d2g, mmax_dlm, False)
    glms[1] = almxfl(dgclm[1], d2g, mmax_dlm, False)
    klm = almxfl(dgclm[0], d2k, mmax, False)
    k = geom.synthesis(klm, 0, lmax, mmax, nthreads)
    g1, g2 = geom.synthesis(glms, 2, lmax, mmax, nthreads)
    d1, d2 = geom.synthesis(dgclm, 1, lmax, mmax, nthreads)
    if np.any(dgclm[1]):
        wlm = almxfl(dgclm[1], d2k, mmax, False)
        w = geom.synthesis(wlm, 0, lmax, mmax, nthreads)
    else:
        wlm, w = 0., 0.
    del dgclm, glms, klm, wlm
    d = np.sqrt(d1 * d1 + d2 * d2)
    max_d = np.max(d)
    if max_d > 0:
        f0 = np.sin(d) / d
        di = d
    else:
        from scipy.special import spherical_jn as jn
        f0 = jn(0, d)
        di = np.where(d > 0, d, 1.) # Something I can take the inverse of
    f1 = np.cos(d) - f0
    try: #FIXME
        import numexpr
        HAS_NUMEXPR = True
    except:
        HAS_NUMEXPR = False
    if HAS_NUMEXPR:
        A = numexpr.evaluate('f0 * ((1. - k) ** 2 - g1 * g1 - g2 * g2 + w * w)')
        A+= numexpr.evaluate('f1 * (1. - k - ( (d1 * d1 - d2 * d2)  * g1 + (2 * d1 * d2) * g2) / (di * di))')
    else:
        A  = f0 * ((1. - k) ** 2 - g1 * g1 - g2 * g2 + w * w)
        A += f1 * (1. - k - ( (d1 * d1 - d2 * d2)  * g1 + (2 * d1 * d2) * g2) / (di * di))
        #                 -      (   cos 2b * g1 + sin 2b * g2 )
    return A.squeeze()