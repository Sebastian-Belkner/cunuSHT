from __future__ import annotations
import numpy as np
from lenspyx import utils_hp
from ducc0.sht.experimental import (
    synthesis as ducc_synthesis,
    adjoint_synthesis as ducc_adjoint_synthesis
)
from ducc0.sht.experimental import (
    adjoint_synthesis_general as ducc_adjoint_synthesis_general,
    synthesis_general as ducc_synthesis_general
)
from ducc0.misc import GL_thetas, GL_weights
from ducc0.fft import good_size
import shtns

VAR = 'GPU'

def set_backend(var):
    global VAR
    VAR = var

def get_backend(var):
    global VAR
    return VAR

def get_transformer(lmax, mmax):
    global sh_gpu
    sh_gpu = shtns.sht(lmax, mmax)
    sh_gpu.set_grid(flags=shtns.SHT_ALLOW_GPU + shtns.SHT_THETA_CONTIGUOUS)
    return sh_gpu


def __shtns_synthesis(*args, **kwargs):
    global sh_gpu
    return np.atleast_2d(sh_gpu.synth(kwargs['alm']))

def __shtns_adjoint_synthesis(*args, **kwargs):
    global sh_gpu
    return np.atleast_2d(sh_gpu.analys(kwargs['map']))

def __shtns_synthesis_general(*args, **kwargs):
    assert 0, "Implement if needed"
    global sh_gpu
    return sh_gpu.analys(**kwargs)

def __shtns_adjoint_synthesis_general(*args, **kwargs):
    assert 0, "Implement if needed"
    global sh_gpu
    return sh_gpu.analys(**kwargs)


def synthesis(*args, **kwargs):
    if get_backend(VAR) == 'CPU':
        # signature: (alm, theta, lmax, mmax, nphi, spin, phi0, nthreads, ringstart, map, **kwargs)
        return ducc_synthesis(*args, **kwargs)
    elif get_backend(VAR) == 'GPU':
        return __shtns_synthesis(*args, **kwargs)


def synthesis_general(*args, **kwargs):
    if get_backend(VAR) == 'CPU':
        # signature: (lmax, mmax, alm, loc, spin, epsilon, nthreads, mode, verbose)
        return ducc_synthesis_general(*args, **kwargs)
    elif get_backend(VAR) == 'GPU':
        assert 0, "implement"
        return __shtns_synthesis_general(*args, **kwargs)


def adjoint_synthesis(*args, **kwargs):
    if get_backend(VAR) == 'CPU':
        # signature: map, theta, lmax, mmax, nphi, spin, phi0, nthreads, ringstart, alm,  **kwargs)
        return ducc_adjoint_synthesis(*args, **kwargs)
    elif get_backend(VAR) == 'GPU':
        
        return __shtns_adjoint_synthesis(*args, **kwargs)


def adjoint_synthesis_general(*args, **kwargs):
    if get_backend(VAR) == 'CPU':
        # signature: (lmax, mmax, map, loc, spin, epsilon, nthreads, mode, alm, verbose)
        return ducc_adjoint_synthesis_general(*args, **kwargs)
    elif get_backend(VAR) == 'GPU':
        assert 0, "implement"
        return __shtns_adjoint_synthesis_general(*args, **kwargs)