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
geom = None

from pysht.transformer import Geom

def get_geom(geometry: tuple[str, dict]=('healpix', {'nside':2048}), backend='CPU'):
    r"""Returns sphere pixelization geometry instance from name and arguments

        Note:
            Custom geometries can be defined following lenspyx.remapping.utils_geom.Geom

    """
    geo = getattr(Geom, '_'.join(['get', geometry[0], 'geometry']), None)
    if geo is None:
        assert 0, 'Geometry %s not found, available geometries: '%geometry[0] + Geom.get_supported_geometries()
    return geo(**geometry[1], backend=backend)

def set_backend(var):
    global VAR
    VAR = var

def get_backend(var):
    global VAR
    return VAR

def get_transformer(lmax, mmax):
    global sh_gpu
    sh_gpu = shtns.sht(lmax, mmax)
    sh_gpu.set_grid(flags=shtns.SHT_ALLOW_GPU + shtns.SHT_THETA_CONTIGUOUS) #  + shtns.SHT_THETA_CONTIGUOUS
    return sh_gpu

def set_geom(g):
    global geom
    geom = g


def __shtns_synthesis(alm, **kwargs):
    global sh_gpu
    return np.atleast_2d(sh_gpu.synth(alm))

def __shtns_adjoint_synthesis(*args, **kwargs):
    global sh_gpu
    return np.atleast_2d(sh_gpu.analys(kwargs['map'])) # analys includes the weights

def __shtns_synthesis_general(*args, **kwargs):
    assert 0, "Implement if needed"
    global sh_gpu
    return sh_gpu.analys(**kwargs)

def __shtns_adjoint_synthesis_general(*args, **kwargs):
    assert 0, "Implement if needed"
    global sh_gpu
    return sh_gpu.analys(**kwargs)


def synthesis(alm, **kwargs):
    if get_backend(VAR) == 'CPU':
        global geom
        # signature: (alm, theta, lmax, mmax, nphi, spin, phi0, nthreads, ringstart, map, **kwargs)
        gclm = np.atleast_2d(alm)
        return ducc_synthesis(alm=gclm, theta=geom.theta, nphi=geom.nph, phi0=geom.phi0, ringstart=geom.ofs, **kwargs)
    elif get_backend(VAR) == 'GPU':
        return __shtns_synthesis(alm, **kwargs)


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