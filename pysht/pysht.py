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

def __pysht_synthesis_general(*args, **kwargs):
    global sh_gpu
    def _get_ptg(self):
        # TODO improve this and fwd angles, e.g. this is computed twice for gamma if no cacher
        self._build_angles() if not self._cis else self._build_angleseig()
        return self.cacher.load('ptg')
    
    # def gclm2lenmap(self, gclm:np.ndarray, mmax:int or None, spin, backwards:bool, polrot=True, ptg=None):
    """Produces deflected spin-weighted map from alm array and instance pointing information

        Args:
            gclm: input alm array, shape (ncomp, nalm), where ncomp can be 1 (gradient-only) or 2 (gradient or curl)
            mmax: mmax parameter of alm array layout, if different from lmax
            spin: spin (>=0) of the transform
            backwards: forward or backward (adjoint) operation
    """
    gclm = np.atleast_2d(gclm)
    lmax_unl = Alm.getlmax(gclm[0].size, mmax)
    if mmax is None:
        mmax = lmax_unl
    if self.single_prec and gclm.dtype != np.complex64:
        gclm = gclm.astype(np.complex64)
        self.tim.add('type conversion')
    if spin == 0 and self._totalconvolves0: # this will probably just disappear
        # The code below would work just as well for spin-0 but seems slightly slower
        # For the moment this seems faster
        blm_T = blm_gauss(0, lmax_unl, 0)
        self.tim.add('blm_gauss')
        if ptg is None:
            ptg = self._build_angles()
        self.tim.add('ptg')
        assert mmax == lmax_unl
        # FIXME: this might only accept doubple prec input
        inter_I = ducc0.totalconvolve.Interpolator(gclm, blm_T, separate=False, lmax=lmax_unl,
                                                    kmax=0,
                                                    epsilon=self.epsilon, ofactor=self.ofactor,
                                                    nthreads=self.sht_tr)
        ret = inter_I.interpol(ptg).squeeze()
        return ret
    # transform slm to Clenshaw-Curtis map
    ntheta = ducc0.fft.good_size(lmax_unl + 2)
    nphihalf = ducc0.fft.good_size(lmax_unl + 1)
    nphi = 2 * nphihalf
    # Is this any different to scarf wraps ? NB: type of map, map_df, and FFTs will follow that of input gclm
    mode = ducc_sht_mode(gclm, spin)
    map = ducc0.sht.experimental.synthesis_2d(alm=gclm, ntheta=ntheta, nphi=nphi,
                            spin=spin, lmax=lmax_unl, mmax=mmax, geometry="CC", nthreads=self.sht_tr, mode=mode)
    # extend map to double Fourier sphere map
    map_dfs = np.empty((2 * ntheta - 2, nphi), dtype=map.dtype if spin == 0 else ctype[map.dtype])
    if spin == 0:
        map_dfs[:ntheta, :] = map[0]
    else:
        map_dfs[:ntheta, :].real = map[0]
        map_dfs[:ntheta, :].imag = map[1]
    del map

    map_dfs[ntheta:, :nphihalf] = map_dfs[ntheta - 2:0:-1, nphihalf:]
    map_dfs[ntheta:, nphihalf:] = map_dfs[ntheta - 2:0:-1, :nphihalf]
    if (spin % 2) != 0:
        map_dfs[ntheta:, :] *= -1

    # go to Fourier space
    if spin == 0:
        tmp = np.empty(map_dfs.shape, dtype=ctype[map_dfs.dtype])
        map_dfs = ducc0.fft.c2c(map_dfs, axes=(0, 1), inorm=2, nthreads=self.sht_tr, out=tmp)
        del tmp
    else:
        map_dfs = ducc0.fft.c2c(map_dfs, axes=(0, 1), inorm=2, nthreads=self.sht_tr, out=map_dfs)

    if self.planned: # planned nufft
        assert ptg is None
        plan = self.make_plan(lmax_unl, spin)
        values = plan.u2nu(grid=map_dfs, forward=False, verbosity=self.verbosity)
        self.tim.add('planned u2nu')
    else:
        if ptg is None:
            ptg = _get_ptg()
        self.tim.add('get ptg')
        values = ducc0.nufft.u2nu(grid=map_dfs, coord=ptg, forward=False,
                                    epsilon=self.epsilon, nthreads=self.sht_tr,
                                    verbosity=self.verbosity, periodicity=2 * np.pi, fft_order=True)
        self.tim.add('u2nu')

    if polrot * spin:
        if self._cis:
            cis = self._get_cischi()
            for i in range(polrot * abs(spin)):
                values *= cis
            self.tim.add('polrot (cis)')
        else:
            if HAS_DUCCROTATE:
                lensing_rotate(values, self._get_gamma(), spin, self.sht_tr)
                self.tim.add('polrot (ducc)')
            else:
                func = fremap.apply_inplace if values.dtype == np.complex128 else fremap.apply_inplacef
                func(values, self._get_gamma(), spin, self.sht_tr)
                self.tim.add('polrot (fortran)')
    # Return real array of shape (2, npix) for spin > 0
    return values.real if spin == 0 else values.view(rtype[values.dtype]).reshape((values.size, 2)).T

def __pysht_adjoint_synthesis_general(*args, **kwargs):

    """
        Note:
            points mst be already quadrature-weigthed

        Note:
            For inverse-lensing, need to feed in lensed maps times unlensed forward magnification matrix.

    """
    self.tim.start('lenmap2gclm')
    self.tim.reset()
    if spin == 0 and not np.iscomplexobj(points):
        points = points.astype(ctype[points.dtype]).squeeze()
    if spin > 0 and not np.iscomplexobj(points):
        points = (points[0] + 1j * points[1]).squeeze()
    ptg = self._get_ptg()


    ntheta = ducc0.fft.good_size(lmax + 2)
    nphihalf = ducc0.fft.good_size(lmax + 1)
    nphi = 2 * nphihalf
    map_dfs = np.empty((2 * ntheta - 2, nphi), dtype=points.dtype)
    if self.planned:
        plan = self.make_plan(lmax, spin)
        map_dfs = plan.nu2u(points=points, out=map_dfs, forward=True, verbosity=self.verbosity)
        self.tim.add('planned nu2u')

    else:
        # perform NUFFT
        map_dfs = ducc0.nufft.nu2u(points=points, coord=ptg, out=map_dfs, forward=True,
                                    epsilon=self.epsilon, nthreads=self.sht_tr, verbosity=self.verbosity,
                                    periodicity=2 * np.pi, fft_order=True)
        self.tim.add('nu2u')
    # go to position space
    map_dfs = ducc0.fft.c2c(map_dfs, axes=(0, 1), forward=False, inorm=2, nthreads=self.sht_tr, out=map_dfs)
    self.tim.add('c2c FFT')

    # go from double Fourier sphere to Clenshaw-Curtis grid
    if (spin % 2) != 0:
        map_dfs[1:ntheta - 1, :nphihalf] -= map_dfs[-1:ntheta - 1:-1, nphihalf:]
        map_dfs[1:ntheta - 1, nphihalf:] -= map_dfs[-1:ntheta - 1:-1, :nphihalf]
    else:
        map_dfs[1:ntheta - 1, :nphihalf] += map_dfs[-1:ntheta - 1:-1, nphihalf:]
        map_dfs[1:ntheta - 1, nphihalf:] += map_dfs[-1:ntheta - 1:-1, :nphihalf]
    map_dfs = map_dfs[:ntheta, :]
    map = np.empty((1 if spin == 0 else 2, ntheta, nphi), dtype=rtype[points.dtype])
    map[0] = map_dfs.real
    if spin > 0:
        map[1] = map_dfs.imag
    del map_dfs
    self.tim.add('Double Fourier')

    # adjoint SHT synthesis
    slm = ducc0.sht.experimental.adjoint_synthesis_2d(map=map, spin=spin,
                        lmax=lmax, mmax=mmax, geometry="CC", nthreads=self.sht_tr, mode=sht_mode, alm=gclm_out)
    self.tim.add('adjoint_synthesis_2d (%s)'%sht_mode)
    self.tim.close('lenmap2gclm')
    return slm.squeeze()


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
        return __pysht_synthesis_general(*args, **kwargs)


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
        return __pysht_adjoint_synthesis_general(*args, **kwargs)