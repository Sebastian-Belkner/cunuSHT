import numpy as np

from lenspyx.utils_hp import Alm
from lenspyx.utils_hp import Alm, alm2cl, almxfl, alm_copy
from lenspyx.utils import timer, blm_gauss
from lenspyx.remapping.utils_angles import d2ang
from lenspyx import cachers

import ducc0

import pysht.geometry as geometry
from pysht.sht_transformer import CPU_DUCC_transformer

ctype = {np.dtype(np.float32): np.complex64,
         np.dtype(np.float64): np.complex128,
         np.dtype(np.longfloat): np.longcomplex,
         np.float32: np.complex64,
         np.float64: np.complex128,
         np.longfloat: np.longcomplex}
rtype = {np.dtype(np.complex64): np.float32,
         np.dtype(np.complex128): np.float64,
         np.dtype(np.longcomplex): np.longfloat,
         np.complex64: np.float32,
         np.complex128: np.float64,
         np.longcomplex: np.longfloat}

HAS_DUCCPOINTING = 'get_deflected_angles' in ducc0.misc.__dict__
HAS_DUCCROTATE = 'lensing_rotate' in ducc0.misc.__dict__
HAS_DUCCGRADONLY = 'mode:' in ducc0.sht.experimental.synthesis.__doc__

from ducc0.sht.experimental import (
    adjoint_synthesis_general as ducc_adjoint_synthesis_general,
    synthesis_general as ducc_synthesis_general
)

if HAS_DUCCPOINTING:
    from ducc0.misc import get_deflected_angles
if HAS_DUCCROTATE:
    from ducc0.misc import lensing_rotate
try:
    from lenspyx.fortran.remapping import remapping as fremap
    HAS_FORTRAN = True
except:
    HAS_FORTRAN = False

@staticmethod
def ducc_sht_mode(gclm, spin):
    gclm_ = np.atleast_2d(gclm)
    return 'GRAD_ONLY' if ((gclm_[0].size == gclm_.size) * (abs(spin) > 0)) else 'STANDARD'


class base:
    def __init__(self):
        self.backend = 'CPU'
        self.single_prec = False
        self.verbosity = 1
        self.tim = timer(verbose=self.verbosity)
        self._totalconvolves0 = False
        self.sht_tr = 4
        self.planned = False
        self._cis = False
        self.cacher = cachers.cacher_mem()
        self.sht_transformer = CPU_DUCC_transformer()
        self.epsilon = 1e-7


    def set_geometry(self, geom_desc):
        self.geom = geometry.get_geom(geom_desc)
        self.sht_transformer.set_geometry(self.geom)


    def _build_d1(self, dlm, lmax_dlm, mmax_dlm, dclm=None):
        if dclm is None:
            # undo p2d to use
            d1 = self.sht_transformer.synthesis(dlm, spin=1, lmax=lmax_dlm, mmax=mmax_dlm, nthreads=self.sht_tr, mode='GRAD_ONLY')
            print('build angles <- synthesis (GRAD_ONLY)')
        else:
            # FIXME: want to do that only once
            dgclm = np.empty((2, dlm.size), dtype=dlm.dtype)
            dgclm[0] = dlm
            dgclm[1] = dclm
            d1 = self.sht_transformer.synthesis(dgclm, spin=1, lmax=lmax_dlm, mmax=mmax_dlm, nthreads=self.sht_tr)
            print('build angles <- synthesis (STANDARD)')
        return d1


    def _build_angles(self, dlm, lmax_dlm, mmax_dlm, fortran=True, calc_rotation=True):
        """Builds deflected positions and angles

            Returns (npix, 3) array with new tht, phi and -gamma

        """
        fns = ['ptg'] + calc_rotation * ['gamma']
        if not np.all([self.cacher.is_cached(fn) for fn in fns]) :
            d1 = self._build_d1(dlm, lmax_dlm, mmax_dlm)
            # Probably want to keep red, imd double precision for the calc?
            if HAS_DUCCPOINTING:
                tht, phi0, nph, ofs = self.geom.theta, self.geom.phi0, self.geom.nph, self.geom.ofs
                print("theta=", tht, "phi0=", phi0, "nphi=", nph, "ringstart=", ofs, "deflect=", d1.T, "calc_rotation=", calc_rotation, "nthreads=", self.sht_tr)
                tht_phip_gamma = get_deflected_angles(theta=tht, phi0=phi0, nphi=nph, ringstart=ofs, deflect=d1.T,
                                                        calc_rotation=calc_rotation, nthreads=self.sht_tr)
                if calc_rotation:
                    self.cacher.cache(fns[0], tht_phip_gamma[:, 0:2])
                    self.cacher.cache(fns[1], tht_phip_gamma[:, 2] if not self.single_prec else tht_phip_gamma[:, 2].astype(np.float32))
                else:
                    self.cacher.cache(fns[0], tht_phip_gamma)
                print('build_angles')
                return
            npix = self.npix
            if fortran and HAS_FORTRAN:
                red, imd = d1
                tht, phi0, nph, ofs = self.geom.theta, self.geom.phi0, self.geom.nph, self.geom.ofs
                if self.single_prec_ptg:
                    thp_phip_gamma = fremap.fpointing(red, imd, tht, phi0, nph, ofs, self.sht_tr)
                else:
                    thp_phip_gamma = fremap.pointing(red, imd, tht, phi0, nph, ofs, self.sht_tr)
                print('build angles <- th-phi-gm (ftn)')
                # I think this just trivially turns the F-array into a C-contiguous array:
                self.cacher.cache(fns[0], thp_phip_gamma.transpose()[:, 0:2])
                if calc_rotation:
                    self.cacher.cache(fns[1], thp_phip_gamma.transpose()[:, 2] if not self.single_prec else thp_phip_gamma.transpose()[:, 2].astype(np.float32))
                return
            elif fortran and not HAS_FORTRAN:
                print('Cant use fortran pointing building since import failed. Falling back on python impl.')
            thp_phip_gamma = np.empty((3, npix), dtype=float)  # (-1) gamma in last arguement
            startpix = 0
            assert np.all(self.geom.theta > 0.) and np.all(self.geom.theta < np.pi), 'fix this (cotangent below)'
            red, imd = d1
            for ir in np.argsort(self.geom.ofs): # We must follow the ordering of scarf position-space map
                pixs = Geom.rings2pix(self, [ir])
                if pixs.size > 0:
                    t_red = red[pixs]
                    i_imd = imd[pixs]
                    phis = Geom.phis(self, ir)[pixs - self.geom.ofs[ir]]
                    assert phis.size == pixs.size, (phis.size, pixs.size)
                    thts = self.geom.theta[ir] * np.ones(pixs.size)
                    thtp_, phip_ = d2ang(t_red, i_imd, thts , phis, int(np.round(np.cos(self.geom.theta[ir]))))
                    sli = slice(startpix, startpix + len(pixs))
                    thp_phip_gamma[0, sli] = thtp_
                    thp_phip_gamma[1, sli] = phip_
                    cot = np.cos(self.geom.theta[ir]) / np.sin(self.geom.theta[ir])
                    d = np.sqrt(t_red ** 2 + i_imd ** 2)
                    thp_phip_gamma[2, sli] = np.arctan2(i_imd, t_red ) - np.arctan2(i_imd, d * np.sin(d) * cot + t_red * np.cos(d))
                    startpix += len(pixs)
            self.cacher.cache(fns[0], thp_phip_gamma.T[:, 0:2])
            if calc_rotation:
                self.cacher.cache(fns[1], thp_phip_gamma.T[:, 2] if not self.single_prec else thp_phip_gamma.T[:, 2].astype(np.float32) )
            assert startpix == npix, (startpix, npix)
            return


    def _get_ptg(self, dlm, mmax):
        # TODO improve this and fwd angles, e.g. this is computed twice for gamma if no cacher
        self._build_angles(dlm, mmax, mmax) if not self._cis else self._build_angleseig()
        ptg = self.cacher.load('ptg')
        return ptg


    def synthesis_general(self, gclm:np.ndarray, dlm, mmax:int or None, spin:int, backwards:bool, polrot=True, ptg=None, epsilon=1e-8, single_prec=True, dclm=None):
        """Produces deflected spin-weighted map from alm array and instance pointing information
            Args:
                gclm: input alm array, shape (ncomp, nalm), where ncomp can be 1 (gradient-only) or 2 (gradient or curl)
                mmax: mmax parameter of alm array layout, if different from lmax
                spin: spin (>=0) of the transform
                backwards: forward or backward (adjoint) operation
        """
        # init deflection via Geom init
        # gclm2lenmap
        if dclm is not None:
            assert 0, "Implement if needed"
            # s2_d += np.sum(alm2cl(dclm, dclm, lmax, mmax_dlm, lmax) * (2 * np.arange(lmax + 1) + 1)) / (4 * np.pi)
            # s2_d /= np.sqrt(2.)
        def _get_ptg():
            # TODO improve this and fwd angles, e.g. this is computed twice for gamma if no cacher
            self._build_angles(dlm, mmax, mmax) if not self._cis else self._build_angleseig()
            ptg = self.cacher.load('ptg')
            print(ptg)
            return self.cacher.load('ptg')

        self.single_prec = single_prec * (epsilon > 1e-6)
        assert not backwards, 'backward 2lenmap not implemented at this moment'
        if self.single_prec and gclm.dtype != np.complex64:
            print('** gclm2lenmap: inconsistent input dtype !')
            gclm = gclm.astype(np.complex64)
        gclm = np.atleast_2d(gclm)
        sht_mode = ducc_sht_mode(gclm, spin)
        lmax_unl = Alm.getlmax(gclm[0].size, mmax)
        if mmax is None:
            mmax = lmax_unl
        if ptg is None:
            ptg = _get_ptg()
        assert ptg.shape[-1] == 2, ptg.shape
        assert ptg.dtype == np.float64, 'synthesis_general only accepts float here'
        if spin == 0:
            print('pysht ', lmax_unl, mmax, gclm.shape, ptg.shape, spin, epsilon, self.sht_tr, sht_mode, self.verbosity)
            print('pysht ', gclm, ptg)
            values = ducc_synthesis_general(lmax=lmax_unl, mmax=mmax, alm=gclm, loc=ptg, spin=spin, epsilon=epsilon,
                                    nthreads=self.sht_tr, mode=sht_mode, verbose=self.verbosity)
            return values


    def synthesis(self, dlm, spin, lmax, mmax, nthreads, mode='STANDARD'):
        return self.sht_transformer.synthesis(dlm, spin=spin, lmax=lmax, mmax=mmax, nthreads=nthreads, mode=mode)