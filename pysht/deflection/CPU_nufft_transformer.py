import os
import sys
import numpy as np
import healpy as hp

import line_profiler

import finufft

from lenspyx.lensing import get_geom as lenspyx_get_geom
from lenspyx.remapping import deflection as lenspyx_deflection
from lenspyx.utils_hp import Alm, alm2cl, almxfl, alm_copy
from lenspyx.remapping.utils_angles import d2ang

import ducc0
from ducc0.sht.experimental import adjoint_synthesis_general, synthesis_general

import pysht
from pysht import cacher
from pysht.utils import timer
import pysht.geometry as geometry
from pysht.geometry import Geom
from pysht.helper import shape_decorator, timing_decorator, debug_decorator
from pysht.sht.CPU_sht_transformer import CPU_SHT_DUCC_transformer, CPU_SHT_SHTns_transformer

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

class deflection:
    def __init__(self, shttransformer_desc, dlm, mmax_dlm:int, geominfo, dclm:np.ndarray=None, epsilon=1e-5, verbosity=0, nthreads=10, single_prec=True, timer=None):
        self.backend = 'CPU'
        self.shttransformer_desc = shttransformer_desc
        if shttransformer_desc == 'ducc':
            self.BaseClass = type('CPU_SHT_DUCC_transformer()', (CPU_SHT_DUCC_transformer,), {})
            self.instance = self.BaseClass(geominfo)
        elif shttransformer_desc == 'shtns':
            self.BaseClass = type('CPU_SHT_SHTns_transformer()', (CPU_SHT_SHTns_transformer,), {})
            self.instance = self.BaseClass(geominfo)
        else:
            raise ValueError('shttransformer_desc must be either "ducc" or "shtns"')
        
        if timer is not None:
            self.timer = timer
        self.single_prec = single_prec
        self.verbosity = verbosity
        self.planned = False
        self.cacher = cacher.cacher_mem()
        self.epsilon = epsilon
        self.nthreads = nthreads
        
        dlm = np.atleast_2d(dlm)
        self.dlm = dlm
        
        self.lmax_dlm = Alm.getlmax(dlm[0].size, mmax_dlm)
        self.mmax_dlm = mmax_dlm
        
        self.geom = geometry.get_geom(geominfo)
        
        s2_d = np.sum(alm2cl(dlm[0], dlm[0], self.lmax_dlm, mmax_dlm, self.lmax_dlm) * (2 * np.arange(self.lmax_dlm + 1) + 1)) / (4 * np.pi)
        if dlm.shape[0]>1:
            s2_d += np.sum(alm2cl(dlm[1], dlm[1], self.lmax_dlm, mmax_dlm, self.lmax_dlm) * (2 * np.arange(self.lmax_dlm + 1) + 1)) / (4 * np.pi)
            s2_d /= np.sqrt(2.)
        sig_d = np.sqrt(s2_d / self.geom.fsky())
        sig_d_amin = sig_d / np.pi * 180 * 60
        if sig_d >= 0.01:
            print('deflection std is %.2e amin: this is really too high a value for something sensible'%sig_d_amin)
        elif self.verbosity:
            print('deflection std is %.2e amin' % sig_d_amin)
            
    def __getattr__(self, name):
        return getattr(self.instance, name)           
        
    def flip_tpg_2d(self, m):
        buff = np.array([_.reshape(self.lmax_dlm+1,-1).T.flatten() for _ in m])
        return buff

    def _build_d1(self, dlm, lmax_dlm, mmax_dlm, dclm=None):
        '''
        This depends on the backend. If SHTns, we can use the synthesis_der1 method. If not, we use a spin-1 SHT
        # FIXME this is a bit of a mess, this function should not distinguish between different SHT backends.
        # Instead, there should be a _build_d1() for each, and they should sit in the repsective transformer modules.
        '''
        ll = np.arange(0,lmax_dlm+1,1)
        if self.shttransformer_desc == 'shtns':
            if dclm is None:
                synth_spin1_map = self.synthesis_der1(hp.almxfl(dlm, np.nan_to_num(np.sqrt(1/(ll*(ll+1))))), nthreads=self.nthreads)  
            else:
                assert 0, "implement if needed, not sure if this is possible with SHTns"
                dgclm = np.empty((2, dlm.size), dtype=dlm.dtype)
                dgclm[0] = dlm
                dgclm[1] = dclm
                synth_spin1_map = self.synthesis_der1(hp.almxfl(dlm, np.nan_to_num(np.sqrt(1/(ll*(ll+1))))), nthreads=self.nthreads)
            return self.flip_tpg_2d(synth_spin1_map)
        elif self.shttransformer_desc == 'ducc':
            if dclm is None:
                d1 = self.synthesis(dlm, spin=1, lmax=lmax_dlm, mmax=mmax_dlm, nthreads=self.nthreads, mode='GRAD_ONLY')
            else:
                dgclm = np.empty((2, dlm.size), dtype=dlm.dtype)
                dgclm[0] = dlm
                dgclm[1] = dclm
                d1 = self.synthesis(dgclm, spin=1, lmax=lmax_dlm, mmax=mmax_dlm, nthreads=self.nthreads)
            return d1
        elif self.shttransformer_desc == 'pysht':
            assert 0, "implement if needed"
        else:
            assert 0, "Not sure what to do with {}".format(self.shttransformer_desc)

    # @profile
    def _build_angles(self, dlm, lmax_dlm, mmax_dlm, fortran=True, calc_rotation=True):
        """Builds deflected positions and angles

            Returns (npix, 3) array with new tht, phi and -gamma

        """
        fns = ['ptg'] + calc_rotation * ['gamma']
        if not np.all([self.cacher.is_cached(fn) for fn in fns]):

            d1 = self._build_d1(dlm, lmax_dlm, mmax_dlm)
            # self.timer.add('spin-1 maps')
            # Probably want to keep red, imd double precision for the calc?
            if HAS_DUCCPOINTING:
                tht, phi0, nph, ofs = self.geom.theta, self.geom.phi0, self.geom.nph, self.geom.ofs
                tht_phip_gamma = get_deflected_angles(theta=tht, phi0=phi0, nphi=nph, ringstart=ofs, deflect=d1.T,
                                                        calc_rotation=calc_rotation, nthreads=self.nthreads)
                if calc_rotation:
                    self.cacher.cache(fns[0], tht_phip_gamma[:, 0:2])
                    self.cacher.cache(fns[1], tht_phip_gamma[:, 2] if not self.single_prec else tht_phip_gamma[:, 2].astype(np.float32))
                else:
                    self.cacher.cache(fns[0], tht_phip_gamma)
                return
            npix = self.geom.npix()
            thp_phip_gamma = np.empty((3, npix), dtype=float)  # (-1) gamma in last arguement
            startpix = 0
            assert np.all(self.geom.theta > 0.) and np.all(self.geom.theta < np.pi), 'fix this (cotangent below)'
            red, imd = d1
            for ir in np.argsort(self.geom.ofs): # We must follow the ordering of scarf position-space map
                pixs = Geom.rings2pix(self.geom, [ir])
                if pixs.size > 0:
                    t_red = red[pixs]
                    i_imd = imd[pixs]
                    phis = Geom.phis(self.geom, ir)[pixs - self.geom.ofs[ir]]
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
        self._build_angles(dlm, mmax, mmax)
        return self.cacher.load('ptg')
    
    @timing_decorator
    # @debug_decorator
    def dlm2pointing(self, dlm):
        pointing_theta, pointing_phi =  self._get_ptg(dlm, self.mmax_dlm).T
        return tuple([pointing_theta, pointing_phi])


class CPU_DUCCnufft_transformer:
    def __init__(self, shttransformer_desc, geominfo, single_prec, epsilon, nthreads, verbosity, planned, deflection_kwargs):
        self.backend = 'CPU'
        self.shttransformer_desc = shttransformer_desc
        self.single_prec = single_prec
        self.epsilon = epsilon
        self.nthreads = nthreads
        self.verbosity = verbosity
        self.planned = planned
        
        if shttransformer_desc == 'ducc':
            self.BaseClass = type('CPU_SHT_DUCC_transformer()', (CPU_SHT_DUCC_transformer,), {})
            self.instance = self.BaseClass(geominfo)
        elif shttransformer_desc == 'shtns':
            self.BaseClass = type('CPU_SHT_SHTns_transformer()', (CPU_SHT_SHTns_transformer,), {})
            self.instance = self.BaseClass(geominfo)
        else:
            raise ValueError('shttransformer_desc must be either "ducc" or "shtns"')
        
        self.geominfo = geominfo
        self.set_geometry(self.geominfo)
        
        self.ntheta = (ducc0.fft.good_size(geominfo[1]['lmax'] + 2) + 3) // 4 * 4
        self.nphihalf = ducc0.fft.good_size(geominfo[1]['lmax'] + 1)
        self.nphi = 2 * self.nphihalf
            
        self.timer = timer(1, prefix=self.backend)
        self.deflectionlib = deflection(shttransformer_desc, **deflection_kwargs, timer=self.timer)

    def __getattr__(self, name):
        return getattr(self.instance, name)

    def gclm2lenmap(self, gclm, dlm, lmax, mmax, spin, nthreads, polrot=True, pointing_theta=None, pointing_phi=None, execmode=0):
        """CPU algorithm for spin-n remapping using duccnufft
            Args:
                gclm: input alm array, shape (ncomp, nalm), where ncomp can be 1 (gradient-only) or 2 (gradient or curl)
                mmax: mmax parameter of alm array layout, if different from lmax
                spin: spin (>=0) of the transform
                backwards: forward or backward (adjoint) operation
        """ 
        
        self.timer.start('gclm2lenmap()')
        self.ret = {}
            
        @timing_decorator
        def _setup(self, gclm, execmode, nthreads):
            assert execmode in ['normal', 'debug', 'timing']
            print('Running in {} execution mode')
            nthreads = self.nthreads if nthreads is None else nthreads
            gclm = np.atleast_2d(gclm)
            if self.single_prec and gclm.dtype != np.complex64:
                gclm = gclm.astype(np.complex64)
            self.execmode = execmode
            self.deflectionlib.execmode = self.execmode
            return gclm
         
        @debug_decorator
        @timing_decorator
        @shape_decorator
        def _synthesis(self, gclm, out):
            out = ducc0.sht.experimental.synthesis_2d(alm=gclm, ntheta=self.ntheta, nphi=self.nphi, spin=spin, lmax=lmax, mmax=mmax, geometry="CC", nthreads=nthreads, mode=ducc_sht_mode(gclm, spin))
            return tuple([out])
        
        @debug_decorator
        @timing_decorator
        @shape_decorator
        def _doubling(self, map, ntheta, nphi, out):
            map_dfs = np.empty((2 * ntheta - 2, nphi), dtype=map.dtype if spin == 0 else ctype[map.dtype])
            if spin == 0:
                map_dfs[:ntheta, :] = map[0]
            else:
                map_dfs[:ntheta, :].real = map[0]
                map_dfs[:ntheta, :].imag = map[1]
            del map
            map_dfs[ntheta:, :self.nphihalf] = map_dfs[ntheta - 2:0:-1, self.nphihalf:]
            map_dfs[ntheta:, self.nphihalf:] = map_dfs[ntheta - 2:0:-1, :self.nphihalf]
            if (spin % 2) != 0:
                map_dfs[ntheta:, :] *= -1
            return tuple([map_dfs])
        
        @debug_decorator
        @timing_decorator
        @shape_decorator
        def _C2C(self, map_dfs, spin, out):
            if spin == 0:
                tmp = np.empty(map_dfs.shape, dtype=ctype[map_dfs.dtype])
                map_dfs = ducc0.fft.c2c(map_dfs.copy(), axes=(0, 1), inorm=2, nthreads=nthreads, out=tmp, forward=True)
                del tmp
            else:
                map_dfs = ducc0.fft.c2c(map_dfs, axes=(0, 1), inorm=2, nthreads=nthreads, out=map_dfs)
            return tuple([map_dfs])
        
        @debug_decorator
        @timing_decorator
        @shape_decorator
        def _nuFFT(self, map_dfs, theta, phi, out):
            out = ducc0.nufft.u2nu(grid=map_dfs.T, coord=np.array([phi,theta]).T, forward=False, epsilon=self.epsilon, nthreads=self.nthreads, verbosity=self.verbosity, periodicity=2*np.pi, fft_order=True)
            return tuple([out])
        
        @debug_decorator
        @timing_decorator
        @shape_decorator
        def _rotate(self, lenmap):
            if polrot * spin:
                if self._cis:
                    cis = self._get_cischi()
                    for i in range(polrot * abs(spin)):
                        lenmap *= cis
                else:
                    if HAS_DUCCROTATE:
                        lensing_rotate(lenmap, self._get_gamma(), spin, nthreads)
                    else:
                        func = fremap.apply_inplace if lenmap.dtype == np.complex128 else fremap.apply_inplacef
                        func(lenmap, self._get_gamma(), spin, nthreads)
            return tuple([lenmap])
             
        
        self.timing, self.debug = None, None
        gclm = _setup(self, gclm, execmode, nthreads)
        
        out = None
        map = _synthesis(self, gclm, out)[0]
        
        @debug_decorator
        @timing_decorator
        @shape_decorator        
        def dlm2pointing(self, pointing_theta, pointing_phi):
            # if pointing_theta is None or pointing_phi is None:
            pointing_theta, pointing_phi = self.deflectionlib.dlm2pointing(dlm)
            return tuple([pointing_theta, pointing_phi])
        
        pointing_theta, pointing_phi = None, None
        pointing_theta, pointing_phi = dlm2pointing(self, pointing_theta, pointing_phi)
        out = None
        map_dfs = _doubling(self, map, self.ntheta, self.nphi, out)[0]
        out = None
        map_dfs = _C2C(self, map_dfs, spin, out)[0]
        lenmap = _nuFFT(self, map_dfs, pointing_theta, pointing_phi, out)[0]
        lenmap = _rotate(self, lenmap)[0]
        
        if self.execmode == 'timing':
            print(self.timer)
            self.timer.dumpjson(os.path.dirname(pysht.__file__)[:-5]+'/test/benchmark/timings/CPU_duccnufft_{}_e{}'.format(lmax, self.epsilon))
        if self.execmode == 'debug':
            return self.ret
        else:
            return lenmap.real if spin == 0 else lenmap.view(rtype[lenmap.dtype]).reshape((lenmap.size, 2)).T


    def lenmap2gclm(self, lenmap:np.ndarray[complex or float], dlm, spin:int, lmax:int, mmax:int, nthreads:int, gclm_out=None, sht_mode='STANDARD', ptg=None):

        def setup(self, lenmap, nthreads):
            assert lenmap.ndim == 2, lenmap.ndim
            assert not np.iscomplexobj(lenmap), (spin, lenmap.ndim, lenmap.dtype)
            lenmap = np.atleast_2d(lenmap)
            if self.single_prec and lenmap.dtype != np.complex64:
                lenmap = lenmap.astype(np.complex64)
            lenmap = np.array(lenmap, dtype=np.complex)
            return lenmap
        if spin == 0 and not np.iscomplexobj(lenmap):
            lenmap = lenmap.astype(ctype[lenmap.dtype]).squeeze()
        if spin > 0 and not np.iscomplexobj(lenmap):
            lenmap = (lenmap[0] + 1j * lenmap[1]).squeeze()
            
        def nuFFT(self):
            map_dfs = np.empty((2 * ntheta - 2, nphi), dtype=lenmap.dtype)
            if self.planned:
                plan = self.make_plan(lmax, spin)
                map_dfs = plan.nu2u(lenmap=lenmap, out=map_dfs, forward=True, verbosity=self.verbosity)
            else:
                map_dfs = ducc0.nufft.nu2u(lenmap=lenmap, coord=ptg, out=map_dfs, forward=True, epsilon=self.epsilon, nthreads=nthreads, verbosity=self.verbosity, periodicity=2 * np.pi, fft_order=True)
            return map_dfs
        
        def C2C(self):
            # map_dfs = ducc0.fft.c2c(map_dfs, axes=(0, 1), forward=False, inorm=2, nthreads=nthreads, out=map_dfs)
            map_dfs = np.empty((2 * ntheta - 2, nphi), dtype=lenmap.dtype)
            map_dfs = scipy.fft.ifft2(lenmap, norm='backward')
            return map_dfs

        def adjoint_doubling(self):
            # go from double Fourier sphere to Clenshaw-Curtis grid
            if (spin % 2) != 0:
                map_dfs[1:ntheta - 1, :nphihalf] -= map_dfs[-1:ntheta - 1:-1, nphihalf:]
                map_dfs[1:ntheta - 1, nphihalf:] -= map_dfs[-1:ntheta - 1:-1, :nphihalf]
            else:
                map_dfs[1:ntheta - 1, :nphihalf] += map_dfs[-1:ntheta - 1:-1, nphihalf:]
                map_dfs[1:ntheta - 1, nphihalf:] += map_dfs[-1:ntheta - 1:-1, :nphihalf]
            map_dfs = map_dfs[:ntheta, :]
            map = np.empty((1 if spin == 0 else 2, ntheta, nphi), dtype=rtype[lenmap.dtype])
            map[0] = map_dfs.real
            if spin > 0:
                map[1] = map_dfs.imag
            del map_dfs
            return map

        def adjoing_synthesis(self):
            gclm = ducc0.sht.experimental.adjoint_synthesis_2d(map=map, spin=spin, lmax=lmax, mmax=mmax, geometry="CC", nthreads=self.nthreads, mode=sht_mode, alm=gclm_out)
            return gclm

        lenmap = setup(self, lenmap, nthreads)
        ptg = self._get_ptg(dlm, mmax)
        map_dfs = nuFFT(self)
        map_dfs = C2C(self)
        gcmap = adjoint_doubling(self)
        gclm = adjoing_synthesis(self)
        return gclm.squeeze()


    def synthesis_general(self, lmax, mmax, map, loc, spin, epsilon, nthreads, sht_mode, alm, verbose):
        assert 0, "implement if needed"
        return synthesis_general(lmax=lmax, mmax=mmax, alm=alm, loc=loc, spin=spin, epsilon=self.epsilon, nthreads=self.sht_tr, mode=sht_mode, verbose=self.verbosity)

  
    def adjoint_synthesis_general(self, lmax, mmax, map, loc, spin, epsilon, nthreads, sht_mode, alm, verbose):
        assert 0, "implement if needed"
        return adjoint_synthesis_general(lmax=lmax, mmax=mmax, map=map, loc=loc, spin=spin, epsilon=self.epsilon, nthreads=self.sht_tr, mode=sht_mode, alm=alm, verbose=self.verbosity)

class CPU_Lenspyx_transformer:
    def __init__(self, shttransformer_desc, geominfo, single_prec, epsilon, nthreads, verbosity, planned, deflection_kwargs):
        self.shttransformer_desc = shttransformer_desc
        self.deflectionlib = lenspyx_deflection(lens_geom=lenspyx_get_geom(deflection_kwargs['geominfo']), dglm=deflection_kwargs['dlm'], mmax_dlm=deflection_kwargs['geominfo'][1]['lmax'], numthreads=deflection_kwargs['nthreads'], verbosity=deflection_kwargs['verbosity'], epsilon=deflection_kwargs['epsilon'], single_prec=deflection_kwargs['single_prec'])
        lens_geom = lenspyx_get_geom(deflection_kwargs['geominfo'])
        self.backend = 'CPU'
        self.single_prec = single_prec
        self.epsilon = epsilon
        self.nthreads = nthreads
        self.verbosity = verbosity
        self.planned = planned


    def gclm2lenmap(self, gclm:np.ndarray, dlm, lmax, mmax:int or None, spin:int, nthreads, backwards:bool=False, polrot=True, ptg=None, dclm=None, execmode=0):
        self.ret = {}
        
        @timing_decorator
        def _setup(self, gclm, execmode, nthreads):
            assert execmode in ['normal','debug', 'timing']
            print('Running in {} execution mode')
            nthreads = self.nthreads if nthreads is None else nthreads
            gclm = np.atleast_2d(gclm)
            if self.single_prec and gclm.dtype != np.complex64:
                gclm = gclm.astype(np.complex64)
            self.execmode = execmode
            self.deflectionlib.execmode = self.execmode
            return gclm
        
        @timing_decorator
        @shape_decorator
        @debug_decorator      
        def _pointing(self):
            ptg = self.get_ptg()
            return tuple([ptg])
            
        @timing_decorator
        @shape_decorator
        @debug_decorator
        def synthesis_general(self, ptg=None):
            res = self.synthesis_general(lmax=lmax, mmax=mmax, alm=gclm, spin=spin, ptg=ptg, epsilon=self.epsilon, nthreads=self.sht_tr, mode=ducc_sht_mode(gclm, spin), verbose=self.verbosity)
            return tuple([res])
            
        self.timer = timer(1, prefix=self.backend)
        self.timer.start('lenspyx')
        gclm = _setup(self, gclm, execmode, nthreads)
        ptg = _pointing(self)[0]
        lenmap = synthesis_general(self, ptg)[0]
        if self.execmode == 'timing':
            self.timer.dumpjson(os.path.dirname(pysht.__file__)[:-5]+'/test/benchmark/timings/CPU_lenspyx_{}_e{}'.format(lmax, self.epsilon))
            print(self.timer)
            print("::timing:: stored new timing data")
        if self.execmode == 'debug':
            print("::debug:: returned component results")
            return self.ret
        return lenmap


    def lenmap2gclm(self, lenmap:np.ndarray[complex or float], dlm:np.ndarray, spin:int, lmax:int, mmax:int, nthreads:int, gclm_out=None, sht_mode='STANDARD', ptg=None):
        @timing_decorator
        @shape_decorator
        @debug_decorator      
        def _pointing(self):
            ptg = self.get_ptg()
            return tuple([ptg])
            
        @timing_decorator
        @shape_decorator
        @debug_decorator
        def adjoint_synthesis_general(self, ptg=None):
            # res = self.deflectionlib.lenmap2gclm(gclm=gclm, mmax=mmax, spin=spin, ptg=ptg)
            res = adjoint_synthesis_general(lmax=lmax, mmax=mmax, map=lenmap, loc=ptg, spin=spin, epsilon=self.epsilon, nthreads=self.sht_tr, mode=ducc_sht_mode(gclm, spin), alm=gclm_out, verbose=self.verbosity)
            return tuple([res])
        
        if ptg is None:
            ptg = _pointing(self)[0]
        gclm = adjoint_synthesis_general(points=np.atleast_2d(lenmap), spin=spin, lmax=lmax, mmax=mmax, gclm_out=gclm_out, sht_mode=sht_mode, ptg=ptg)
        if self.execmode == 'timing':
            self.timer.dumpjson(os.path.dirname(pysht.__file__)[:-5]+'/test/benchmark/timings/lenmap2gclm/CPU_lenspyx_{}_e{}'.format(lmax, self.epsilon))
            print(self.timer)
            print("::timing:: stored new timing data")
        if self.execmode == 'debug':
            print("::debug:: Returned component results")
            return self.ret
        return gclm


    def gclm2lenpixs(self, gclm:np.ndarray, mmax:int or None, spin:int, pixs:np.ndarray[int], polrot=True, ptg=None):
        """Produces the remapped field on the required lensing geometry pixels 'exactly', by brute-force calculation
            Note:
                The number of pixels must be small here, otherwise way too slow
            Note:
                If the remapping angles etc were not calculated previously, it will build the full map, so may take some time.
        """
        assert spin >= 0, spin
        gclm = np.atleast_2d(gclm)
        sth_mode = ducc_sht_mode(gclm, spin)
        ptg = self._get_ptg()
        thts, phis = ptg[pixs, 0], ptg[pixs, 1]
        nph = 2 * np.ones(thts.size, dtype=np.uint64)  # I believe at least 2 points per ring
        ofs = 2 * np.arange(thts.size, dtype=np.uint64)
        wt = np.ones(thts.size, dtype=float)
        geom = Geom(thts.copy(), phis.copy(), nph, ofs, wt)
        gclm = np.atleast_2d(gclm)
        lmax = Alm.getlmax(gclm[0].size, mmax)
        if mmax is None: mmax = lmax
        m = geom.synthesis(gclm, spin, lmax, mmax, self.sht_tr, mode=sth_mode)[:, 0::2]
        # could do: complex view trick etc
        if spin and polrot:
            gamma = self._get_gamma()[pixs]
            m = np.exp(1j * spin * gamma) * (m[0] + 1j * m[1])
            return m.real, m.imag
        return m.squeeze()


    def synthesis_general(self, lmax, mmax, loc, spin, sht_mode, alm):
        assert 0, "implement if needed"
        return synthesis_general(lmax=lmax, mmax=mmax, alm=alm, loc=loc, spin=spin, epsilon=self.epsilon, nthreads=self.sht_tr, mode=sht_mode, verbose=self.verbosity)
    
    def adjoint_synthesis_general(self, lmax, mmax, map, loc, spin, epsilon, nthreads, sht_mode, alm, verbose):
        assert 0, "implement if needed"
        return adjoint_synthesis_general(lmax=lmax, mmax=mmax, map=map, loc=loc, spin=spin, epsilon=self.epsilon, nthreads=self.sht_tr, mode=sht_mode, alm=alm, verbose=self.verbosity)

    def get_ptg(self):
        return self.deflectionlib._get_ptg()    


class CPU_finufft_transformer:
    def __init__(self, shttransformer_desc, geominfo, deflection_kwargs):
        self.backend = 'CPU'
        self.shttransformer_desc = shttransformer_desc
        if shttransformer_desc == 'ducc':
            self.BaseClass = type('CPU_SHT_DUCC_transformer()', (CPU_SHT_DUCC_transformer,), {})
            self.instance = self.BaseClass(geominfo)
        elif shttransformer_desc == 'shtns':
            self.BaseClass = type('CPU_SHT_SHTns_transformer()', (CPU_SHT_SHTns_transformer,), {})
            self.instance = self.BaseClass(geominfo)
        else:
            raise ValueError('shttransformer_desc must be either "ducc" or "shtns"')

        self.geominfo = geominfo
        self.set_geometry(geominfo)


    def __getattr__(self, name):
        return getattr(self.instance, name)


    # @profile
    def gclm2lenmap(self, gclm, dlm, lmax, mmax, spin, nthreads, polrot=True, cc_transformer=None, HAS_DUCCPOINTING=True, mode=0):
        """CPU algorithm for spin-n remapping using finufft
            Args:
                gclm: input alm array, shape (ncomp, nalm), where ncomp can be 1 (gradient-only) or 2 (gradient or curl)
                mmax: mmax parameter of alm array layout, if different from lmax
                spin: spin (>=0) of the transform
                backwards: forward or backward (adjoint) operation
        """ 
        if mode == 0:
            print('Running in normal mode')
            timing = False
            debug = False
        if mode == 1:
            print('Running in timing mode')
            timing = True
            debug = False
        if mode == 2:
            print("Running in debug mode")
            timing = False
            debug = True
        ret = []
        
        self.timer = timer(1, prefix=self.backend)
        self.timer.start('gclm2lenmap()')
        gclm = np.atleast_2d(gclm)
        lmax_unl = Alm.getlmax(gclm[0].size, mmax)
        if mmax is None:
            mmax = lmax_unl
        if self.single_prec and gclm.dtype != np.complex64:
            gclm = gclm.astype(np.complex64)
        # self.timer.add('setup')

        # transform slm to Clenshaw-Curtis map
        if not debug:
            ntheta = (ducc0.fft.good_size(lmax_unl + 2) + 3) // 4 * 4
            # ntheta = ducc0.fft.good_size(lmax_unl + 2)
            nphihalf = ducc0.fft.good_size(lmax_unl + 1)
            nphi = 2 * nphihalf
        else:
            ntheta = lmax+1
            nphihalf = lmax+1
            nphi = 2 * nphihalf
        
        ### SYNTHESIS CC GEOMETRY ###
        mode = ducc_sht_mode(gclm, spin)
        map = ducc0.sht.experimental.synthesis_2d(alm=gclm, ntheta=ntheta, nphi=nphi, spin=spin, lmax=lmax_unl, mmax=mmax, geometry="CC", nthreads=nthreads, mode=mode)
        self.timer.add('synthesis')
        if debug:
            ret.append(np.copy(map))
        
        map_dfs = np.empty((2 * ntheta - 2, nphi), dtype=np.complex128 if spin == 0 else ctype[map.dtype])
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
        self.timer.add('doubling')
        if debug:
            ret.append(np.copy(map_dfs))


        # go to Fourier space
        if spin == 0:
            tmp = np.empty(map_dfs.shape, dtype=np.complex128)
            map_dfs = ducc0.fft.c2c(map_dfs, axes=(0, 1), inorm=2, nthreads=nthreads, out=tmp)
            del tmp
        else:
            map_dfs = ducc0.fft.c2c(map_dfs, axes=(0, 1), inorm=2, nthreads=nthreads, out=map_dfs)
        self.timer.add('c2c')
        if debug:
            ret.append(np.copy(map_dfs))
        
        if self.planned: # planned nufft
            assert ptg is None
            plan = self.make_plan(lmax_unl, spin)
            values = plan.u2nu(grid=map_dfs, forward=False, verbosity=self.verbosity)
        else:
            ptg = None
            if ptg is None:
                ptg = self._get_ptg(dlm, mmax)
            self.timer.add('get ptg')
            if debug:
                ret.append(np.copy(ptg))
                
            map_shifted = np.fft.fftshift(map_dfs, axes=(0,1))
            x_ = np.array(ptg[:,0], order="C")
            y_ = np.array(ptg[:,1], order="C")
            f_ = np.array(map_shifted, dtype=np.complex128, order="C")
            v_ = finufft.nufft2d2(x=x_, y=y_, f=f_, isign=1)
            self.timer.add('nuFFT')
            values = np.roll(np.real(v_))
            
        if debug:
            ret.append(np.copy(values))   

        if polrot * spin:
            if self._cis:
                cis = self._get_cischi()
                for i in range(polrot * abs(spin)):
                    values *= cis
            else:
                if HAS_DUCCROTATE:
                    lensing_rotate(values, self._get_gamma(), spin, nthreads)
                else:
                    func = fremap.apply_inplace if values.dtype == np.complex128 else fremap.apply_inplacef
                    func(values, self._get_gamma(), spin, nthreads)
        if debug:
            ret.append(np.copy(values)) 
        
        if timing:
            self.timer.dumpjson('/mnt/home/sbelkner/git/pySHT/test/benchmark/timings/CPU_finufft_{}'.format(lmax))
        if debug:
            return ret
        else:
            return values.real.flatten() if spin == 0 else values.view(rtype[values.dtype]).reshape((values.size, 2)).T
        # np.atleast_2d(values.real.flatten())


    def lenmap2gclm(self, points:np.ndarray[complex or float], dlm, spin:int, lmax:int, mmax:int, nthreads:int, gclm_out=None, sht_mode='STANDARD'):
        """
            Note:
                points mst be already quadrature-weigthed
                For inverse-lensing, need to feed in lensed maps times unlensed forward magnification matrix.
        """
        if spin == 0 and not np.iscomplexobj(points):
            points = points.astype(ctype[points.dtype]).squeeze()
        if spin > 0 and not np.iscomplexobj(points):
            points = (points[0] + 1j * points[1]).squeeze()
        # FIXME stop passing synthesis function as _get_d1 needs it..
        ptg = self._get_ptg(dlm, mmax)


        ntheta = ducc0.fft.good_size(lmax + 2)
        nphihalf = ducc0.fft.good_size(lmax + 1)
        nphi = 2 * nphihalf
        map_dfs = np.empty((2 * ntheta - 2, nphi), dtype=points.dtype)
        if self.planned:
            plan = self.make_plan(lmax, spin)
            map_dfs = plan.nu2u(points=points, out=map_dfs, forward=True, verbosity=self.verbosity)

        else:
            # perform NUFFT
        
            map_dfs = ducc0.nufft.nu2u(points=points, coord=ptg, out=map_dfs, forward=True,
                                       epsilon=self.epsilon, nthreads=nthreads, verbosity=self.verbosity,
                                       periodicity=2 * np.pi, fft_order=True)
        # go to position space
        map_dfs = ducc0.fft.c2c(map_dfs, axes=(0, 1), forward=False, inorm=2, nthreads=nthreads, out=map_dfs)

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

        # adjoint SHT synthesis
        slm = ducc0.sht.experimental.adjoint_synthesis_2d(map=map, spin=spin, lmax=lmax, mmax=mmax, geometry="CC", nthreads=nthreads, mode=sht_mode, alm=gclm_out)
        return slm.squeeze()
    

    def lensgclm(self, gclm:np.ndarray, dlm:np.array, spin:int, lmax_out:int, nthreads:int, mmax:int=None, mmax_out:int=None,gclm_out:np.ndarray=None, polrot=True, out_sht_mode='STANDARD'):
        """Adjoint remapping operation from lensed alm space to unlensed alm space

            Args:
                gclm: input gradient and possibly curl mode ((1 or 2, nalm)-shaped complex numpy.ndarray)
                mmax: set this for non-standard mmax != lmax in input array
                spin: spin-weight of the fields (larger or equal 0)
                lmax_out: desired output array lmax
                mmax_out: desired output array mmax (defaults to lmax_out if None)
                gclm_out(optional): output array (can be same as gclm provided it is large enough)
                polrot(optional): includes small rotation of spin-weighted fields (defaults to True)
                out_sht_mode(optional): e.g. 'GRAD_ONLY' if only the output gradient mode is desired
            Note:
                 nomagn=True is a backward comptability thing to ask for inverse lensing
        """
        stri = 'lengclm ' +  'fwd' 
        input_sht_mode = ducc_sht_mode(gclm, spin)
        if mmax_out is None:
            mmax_out = lmax_out
        m = self.gclm2lenmap(gclm, dlm=dlm, lmax=lmax_out, mmax=lmax_out, spin=spin, nthreads=nthreads, polrot=polrot)
        if gclm_out is not None:
            assert gclm_out.dtype == ctype[m.dtype], 'type precision must match'
        gclm_out = self.adjoint_synthesis(m, spin=spin, lmax=lmax_out, mmax=mmax_out, nthreads=nthreads, alm=gclm_out, mode=out_sht_mode)
        return gclm_out.squeeze()

    
    def synthesis_general(self, lmax, mmax, map, loc, spin, epsilon, nthreads, sht_mode, alm, verbose):
        assert 0, "implement if needed"
        return synthesis_general(lmax=lmax, mmax=mmax, alm=alm, loc=loc, spin=spin, epsilon=self.epsilon, nthreads=self.sht_tr, mode=sht_mode, verbose=self.verbosity)
    
    def adjoint_synthesis_general(self, lmax, mmax, map, loc, spin, epsilon, nthreads, sht_mode, alm, verbose):
        assert 0, "implement if needed"
        return adjoint_synthesis_general(lmax=lmax, mmax=mmax, map=map, loc=loc, spin=spin, epsilon=self.epsilon, nthreads=self.sht_tr, mode=sht_mode, alm=alm, verbose=self.verbosity)

