import os
import sys
import numpy as np
import healpy as hp

import line_profiler

import finufft

import lenspyx
from lenspyx.lensing import get_geom as get_lenspyxgeom
from lenspyx.remapping import deflection as lenspyx_deflection
from lenspyx.utils_hp import Alm, alm2cl, almxfl, alm_copy
from lenspyx.remapping.utils_angles import d2ang

import ducc0
from ducc0.sht.experimental import adjoint_synthesis_general, synthesis_general

import cunusht
from cunusht import cacher
from cunusht.utils import timer as tim
import cunusht.geometry as geometry
from cunusht.geometry import Geom
from cunusht.helper import shape_decorator, timing_decorator, debug_decorator, timing_decorator_close
from cunusht.sht.CPU_sht_transformer import CPU_SHT_DUCC_transformer, CPU_SHT_SHTns_transformer

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
timer = tim(1, prefix='GPU')
class deflection:
    def __init__(self, geominfo, shttransformer_desc='ducc', timer_instance=None):
        if timer_instance is None:
            self.timer = timer
        else:
            self.timer = timer_instance
        self.cacher = cacher.cacher_mem()
        
        self.geominfo = geominfo
        self.geom = geometry.get_geom(geominfo)
        
        self.shttransformer_desc = shttransformer_desc
        if shttransformer_desc == 'shtns':
            self.BaseClass = type('GPU_SHTns_transformer', (CPU_SHT_SHTns_transformer,), {})
        elif shttransformer_desc == 'ducc':
            self.BaseClass = type('CPU_SHT_DUCC_transformer()', (CPU_SHT_DUCC_transformer,), {})
        else:
            raise ValueError('shttransformer_desc must be either "ducc" or "shtns"')
        self.instance = self.BaseClass(geominfo=geominfo)


    def __getattr__(self, name):
        return getattr(self.instance, name)           
        
    def flip_tpg_2d(self, m):
        buff = np.array([_.reshape(self.lmax_dlm+1,-1).T.flatten() for _ in m])
        return buff

    @timing_decorator
    def _spin__1___synth(self, dlm, lmax_dlm, mmax_dlm, nthreads, dclm=None):
        '''
        This depends on the backend. If SHTns, we can use the synthesis_der1 method. If not, we use a spin-1 SHT
        # FIXME this is a bit of a mess, this function should not distinguish between different SHT backends.
        # Instead, there should be a _spin__1___synth() for each, and they should sit in the repsective transformer modules.
        '''
        ll = np.arange(0,lmax_dlm+1,1)
        if self.shttransformer_desc == 'shtns':
            if dclm is None:
                synth_spin1_map = self.synthesis_der1(hp.almxfl(dlm, np.nan_to_num(np.sqrt(1/(ll*(ll+1))))), nthreads=nthreads)  
            else:
                assert 0, "implement if needed, not sure if this is possible with SHTns"
                dgclm = np.empty((2, dlm.size), dtype=dlm.dtype)
                dgclm[0] = dlm
                dgclm[1] = dclm
                synth_spin1_map = self.synthesis_der1(hp.almxfl(dlm, np.nan_to_num(np.sqrt(1/(ll*(ll+1))))), nthreads=nthreads)
            return self.flip_tpg_2d(synth_spin1_map)
        elif self.shttransformer_desc == 'ducc':
            if dclm is None:
                d1 = self.synthesis(dlm, spin=1, lmax=lmax_dlm, mmax=mmax_dlm, nthreads=nthreads, mode='GRAD_ONLY')
            else:
                dgclm = np.empty((2, dlm.size), dtype=dlm.dtype)
                dgclm[0] = dlm
                dgclm[1] = dclm
                d1 = self.synthesis(dgclm, spin=1, lmax=lmax_dlm, mmax=mmax_dlm, nthreads=nthreads)
            return d1
        elif self.shttransformer_desc == 'cunusht':
            assert 0, "implement if needed"
        else:
            assert 0, "Not sure what to do with {}".format(self.shttransformer_desc)

    # @profile
    def _build_angles(self, dlm, lmax_dlm, mmax_dlm, nthreads, fortran=True, calc_rotation=True):
        """Builds deflected positions and angles

            Returns (npix, 3) array with new tht, phi and -gamma

        """
        fns = ['ptg'] + calc_rotation * ['gamma']
        if not np.all([self.cacher.is_cached(fn) for fn in fns]):

            d1 = self._spin__1___synth(dlm, lmax_dlm, mmax_dlm, nthreads)
            # self.timer.add('spin-1 maps')
            # Probably want to keep red, imd double precision for the calc?
            if HAS_DUCCPOINTING:
                tht, phi0, nph, ofs = self.geom.theta, self.geom.phi0, self.geom.nph, self.geom.ofs
                tht_phip_gamma = get_deflected_angles(theta=tht, phi0=phi0, nphi=nph, ringstart=ofs, deflect=d1.T,
                                                        calc_rotation=calc_rotation, nthreads=nthreads)
                self.timer.add('pointing')
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


    def _get_ptg(self, dlm, mmax, nthreads):
        self._build_angles(dlm, mmax, mmax, nthreads)
        return self.cacher.load('ptg')
    
    @timing_decorator
    # @debug_decorator
    def dlm2pointing(self, dlm, mmax_dlm, single_prec, nthreads):
        self.single_prec = single_prec
        pointing_theta, pointing_phi =  self._get_ptg(dlm, mmax_dlm, nthreads).T
        return np.array([pointing_theta, pointing_phi])


class CPU_DUCCnufft_transformer:
    def __init__(self, geominfo_deflection, shttransformer_desc='ducc', nuFFTtype=None, epsilon=None):
        """This is not the fastest way of performing synthesis_general, but serves as a compoarison to GPU code

        Args:
            geominfo_deflection (_type_): _description_
            shttransformer_desc (str, optional): _description_. Defaults to 'shtns'.
            planned (bool, optional): _description_. Defaults to True.
            epsilon (_type_, optional): _description_. Defaults to None.

        Raises:
            ValueError: _description_
        """
        self.timer = timer
        self.timer.reset_ti()
        self.timer.start(self.__class__.__name__)
        
        if nuFFTtype:
            assert False, "nuFFTtype mode not supported"
        self.backend = 'GPU'
        self.shttransformer_desc = shttransformer_desc
        self.nuFFTtype = nuFFTtype
        self.execmode = None
        self.timer = timer
        self.ret = {} # This is for execmode='debug'
        
        # Take ducc good_size, but adapt for good size needed by GPU SHTns (nlat must be multiple of 4)
        self.ntheta_CAR = (ducc0.fft.good_size(geominfo_deflection[1]['lmax'] + 2) + 3) // 4 * 4
        self.nphihalf_CAR = ducc0.fft.good_size(geominfo_deflection[1]['lmax'] + 1)
        self.nphi_CAR = 2 * self.nphihalf_CAR
            
        self.ntheta_dCAR, self.nphi_dCAR = int(2*self.ntheta_CAR-2), int(self.nphi_CAR)
        self.CARmap = np.empty((self.ntheta_CAR*self.nphi_CAR), dtype=np.double)
        self.CARdmap = np.empty((self.ntheta_dCAR*self.nphi_dCAR), dtype=np.double)
        
        self.deflectionlib = deflection(geominfo=geominfo_deflection, shttransformer_desc=shttransformer_desc, timer_instance=self.timer)
        self.timer.add("init")


    def _ensure_dtype_nuFFT(self, item):
        if self.single_prec:
            item = item.astype(np.complex64)
        else:
            item = item.astype(np.complex128)
        return item
    
    def _ensure_shape(self, item):
        return item.squeeze()
    
    def _ensure_batchedshape(self, item):
        return np.atleast_2d(item)
    
    def _ensure_dtype(self, item):
        if self.single_prec:
            item = item.astype(np.float32)
        else:
            item = item.astype(np.double)
        return item
        
    def _assert_shape(self, lenmap, gclm_out, ndim):
        """ no-batch shape check.
        """
        assert len(lenmap.shape) == ndim, len(lenmap.shape)
        assert len(gclm_out.shape) == ndim, len(gclm_out.shape)
        
    def _assert_precision(self, lenmap, gclm_out, epsilon):
        # assert lenmap.dtype == gclm_out.dtype, "lenmap and gclm_out must have same precision, but are {} and {}".format(lenmap.dtype, gclm_out.dtype)
        if self.single_prec:
            assert lenmap.dtype in [np.float32], "lenmap must be single precision"
            assert gclm_out.dtype in [np.complex64], "gclm_out must be single precision"
            assert epsilon>1e-6, "epsilon must be > 1e-6 for single precision"
        else:
            assert lenmap.dtype in [np.float64], "lenmap must be double precision"
            assert gclm_out.dtype in [np.complex128], "gclm_out must be double precision"
            assert epsilon<=1e-6, "epsilon must be > 1e-6 for double precision"

    def __getattr__(self, name):
        return getattr(self.instance, name)


    @debug_decorator
    @timing_decorator
    # @shape_decorator
    def C2C(self, dmap, spin, nthreads, out):
        if spin == 0:
            tmp = np.empty(dmap.shape, dtype=ctype[dmap.dtype])
            dmap = ducc0.fft.c2c(dmap, axes=(0, 1), inorm=2, nthreads=nthreads, out=tmp, forward=True)
            del tmp
        else:
            dmap = ducc0.fft.c2c(dmap, axes=(0, 1), inorm=2, nthreads=nthreads, out=dmap)
        return dmap
    
    @debug_decorator
    @timing_decorator
    # @shape_decorator       
    def iC2C(self, fc, dmap, nthreads):
        return ducc0.fft.c2c(fc, axes=(0, 1), inorm=2, nthreads=nthreads, out=dmap, forward=False)

    @debug_decorator
    @timing_decorator
    # @shape_decorator
    def doubling(self, map, ntheta, nphi, spin, out):
        dmap = np.empty((2 * ntheta - 2, nphi), dtype=map.dtype if spin == 0 else ctype[map.dtype])
        if spin == 0:
            dmap[:ntheta, :] = map[0]
        else:
            dmap[:ntheta, :].real = map[0]
            dmap[:ntheta, :].imag = map[1]
        del map
        dmap[ntheta:, :self.nphihalf_CAR] = dmap[ntheta - 2:0:-1, self.nphihalf_CAR:]
        dmap[ntheta:, self.nphihalf_CAR:] = dmap[ntheta - 2:0:-1, :self.nphihalf_CAR]
        if (spin % 2) != 0:
            dmap[ntheta:, :] *= -1
        return dmap
    
    @debug_decorator
    @timing_decorator
    # @shape_decorator
    def adjoint_doubling(self, dmap, gcmap, spin):
        # go from double Fourier sphere to Clenshaw-Curtis grid
        if (spin % 2) != 0:
            dmap[1:self.ntheta_CAR - 1, :self.nphihalf_CAR] -= dmap[-1:self.ntheta_CAR - 1:-1, self.nphihalf_CAR:]
            dmap[1:self.ntheta_CAR - 1, self.nphihalf_CAR:] -= dmap[-1:self.ntheta_CAR - 1:-1, :self.nphihalf_CAR]
        else:
            dmap[1:self.ntheta_CAR - 1, :self.nphihalf_CAR] += dmap[-1:self.ntheta_CAR - 1:-1, self.nphihalf_CAR:]
            dmap[1:self.ntheta_CAR - 1, self.nphihalf_CAR:] += dmap[-1:self.ntheta_CAR - 1:-1, :self.nphihalf_CAR]
        dmap = dmap[:self.ntheta_CAR, :]
        if self.single_prec:
            gcmap = np.empty((1 if spin == 0 else 2, self.ntheta_CAR, self.nphi_CAR), dtype=np.complex64)
        else:
            gcmap = np.empty((1 if spin == 0 else 2, self.ntheta_CAR, self.nphi_CAR), dtype=np.complex128)
        gcmap[0] = dmap.real
        if spin > 0:
            gcmap[1] = dmap.imag
        del dmap 
        return gcmap

    @debug_decorator
    @timing_decorator
    # @shape_decorator
    def synthesis(self, gclm, spin, lmax, mmax, nthreads, mode, out):
        out = ducc0.sht.experimental.synthesis_2d(alm=gclm, ntheta=self.ntheta_CAR, nphi=self.nphi_CAR, spin=spin, lmax=lmax, mmax=mmax, geometry="CC", nthreads=nthreads, mode=mode)
        return out

    @debug_decorator
    @timing_decorator
    # @shape_decorator
    def adjoint_synthesis(self, dmap, gclm, spin, lmax, mmax, mode='STANDARD'):
        return ducc0.sht.experimental.adjoint_synthesis_2d(map=dmap.real, spin=spin, lmax=lmax, mmax=mmax, geometry="CC", nthreads=self.nthreads, mode=mode, alm=gclm)

    @debug_decorator
    @timing_decorator
    # @shape_decorator        
    def nuFFT2d1(self, lenmap, ptg, nthreads, epsilon, verbosity, fc):
        """assert_shape: batching not supported
        assert dtype: lenmap, fc must be complex
        assert shape: lenmap must be flat (2 * ntheta - 2) * nphi, ptg must be ((2 * ntheta - 2) * nphi, 2), fc must be (2 * ntheta - 2, nphi)
        """
        return ducc0.nufft.nu2u(
            points=lenmap, coord=ptg, forward=True,
            epsilon=epsilon, nthreads=nthreads,
            verbosity=verbosity, periodicity=2*np.pi, out=fc, fft_order=True)
    
    @debug_decorator
    @timing_decorator
    # @shape_decorator
    def nuFFT2d2(self, grid, x, y, epsilon, nthreads, verbosity, out):
        out = ducc0.nufft.u2nu(grid=grid.T, coord=np.array([y,x]).T, forward=False, epsilon=epsilon, nthreads=nthreads, verbosity=verbosity, periodicity=2*np.pi, fft_order=True)
        return out
    
    @debug_decorator
    @timing_decorator
    # @shape_decorator
    def rotate(self, pointmap, polrot, spin, nthreads):
        if polrot * spin:
            if self._cis:
                cis = self._get_cischi()
                for i in range(polrot * abs(spin)):
                    pointmap *= cis
            else:
                if HAS_DUCCROTATE:
                    lensing_rotate(pointmap, self._get_gamma(), spin, nthreads)
                else:
                    func = fremap.apply_inplace if pointmap.dtype == np.complex128 else fremap.apply_inplacef
                    func(pointmap, self._get_gamma(), spin, nthreads)
        return pointmap
    
    @debug_decorator
    @timing_decorator
    # @shape_decorator  
    def synthesis_general(self, pointmap, lmax, mmax, alm, loc, spin, epsilon, nthreads, mode, verbose):
        pointing_theta, pointing_phi = loc[:,0], loc[:,1]
        
        pointmap = self._ensure_batchedshape(pointmap)
        alm = self._ensure_batchedshape(alm)
        
        out = None
        CARmap = self.synthesis(alm, spin=spin, lmax=lmax, mmax=mmax, mode=mode, nthreads=nthreads, out=out)
        out = None
        dmap = self.doubling(CARmap, ntheta=self.ntheta_CAR, nphi=self.nphi_CAR, spin=spin, out=out)
        out = None
        dmap = self.C2C(dmap, spin, nthreads, out)
        pointmap = self.nuFFT2d2(grid=dmap, x=pointing_theta, y=pointing_phi, epsilon=epsilon, nthreads=nthreads, verbosity=verbose, out=out)
        return pointmap

    @debug_decorator
    @timing_decorator
    # @shape_decorator  
    def adjoint_synthesis_general(self, lmax, mmax, pointmap, loc, spin, epsilon, nthreads, mode, alm, verbose):
        nalm = ((mmax+1)*(mmax+2))//2 + (mmax+1)*(lmax-mmax)
        pointing_theta, pointing_phi = loc[:,0], loc[:,1]
        
        fc = np.empty((2 * self.ntheta_CAR - 2, self.nphi_CAR), dtype=pointmap.dtype)
        fc = self._ensure_dtype_nuFFT(fc)
        fc = self.nuFFT2d1(pointmap, np.array([pointing_theta, pointing_phi]).T, nthreads, epsilon, verbose, fc)
        
        dmap = np.empty((2 * self.ntheta_CAR - 2, self.nphi_CAR), dtype=pointmap.dtype)
        dmap = self.iC2C(fc, dmap, nthreads)
        gcmap = np.empty((self.ntheta_CAR, self.nphi_CAR), dtype=pointmap.real.dtype)
        gcmap = self.adjoint_doubling(dmap, gcmap, spin)
        alm = np.empty((1, nalm), dtype=pointmap.dtype)
        alm = self.adjoint_synthesis(gcmap, alm, spin=spin, lmax=lmax, mmax=mmax, mode=mode)
        
        return alm

    @timing_decorator
    def gclm2lenmap(self, gclm, dlm, lmax, mmax, spin, epsilon, nthreads, polrot=True, ptg=None, lenmap=None, verbosity=1, execmode=0):
        """CPU algorithm for spin-n remapping using duccnufft
            Args:
                gclm: input alm array, shape (ncomp, nalm), where ncomp can be 1 (gradient-only) or 2 (gradient or curl)
                mmax: mmax parameter of alm array layout, if different from lmax
                spin: spin (>=0) of the transform
                backwards: forward or backward (adjoint) operation
        """
        
        self.single_prec = True if epsilon>1e-6 else False
        dlm = np.atleast_2d(dlm)
        s2_d = np.sum(alm2cl(dlm[0], dlm[0], lmax, mmax, lmax) * (2 * np.arange(lmax + 1) + 1)) / (4 * np.pi)
        if dlm.shape[0]>1:
            s2_d += np.sum(alm2cl(dlm[1], dlm[1], lmax, mmax, lmax) * (2 * np.arange(lmax + 1) + 1)) / (4 * np.pi)
            s2_d /= np.sqrt(2.)
        sig_d = np.sqrt(s2_d / self.deflectionlib.geom.fsky())
        sig_d_amin = sig_d / np.pi * 180 * 60
        if sig_d >= 0.01:
            print('deflection std is %.2e amin: this is really too high a value for something sensible'%sig_d_amin)
        elif verbosity:
            print('deflection std is %.2e amin' % sig_d_amin)
        if dlm.shape[0]==1:
            # FIXME this is to align shape of dlm for the next steps
            dlm = dlm[0]

        def setup(self, gclm, nthreads):
            assert execmode in ['normal', 'debug', 'timing']
            print('Running in {} execution mode')
            nthreads = self.nthreads if nthreads is None else nthreads
            if self.single_prec and gclm.dtype != np.complex64:
                gclm = gclm.astype(np.complex64)
            self.execmode = execmode
            self.deflectionlib.execmode = self.execmode
         
        setup(self, gclm, nthreads)
        if lenmap is not None:
            lenmap = self._ensure_dtype(lenmap)
            self._assert_precision(lenmap, gclm, epsilon)
   
        if ptg is None:
            pointing_theta, pointing_phi = self.deflectionlib.dlm2pointing(dlm, mmax_dlm=lmax, single_prec=self.single_prec, nthreads=nthreads)
        else:
            pointing_theta, pointing_phi = ptg[:,0], ptg[:,1]
        if self.execmode == 'debug':
            self.ret.update({'dlm2pointing': np.array([pointing_theta, pointing_phi])})
        pointing_theta = self._ensure_dtype(pointing_theta)
        pointing_phi = self._ensure_dtype(pointing_phi)
        
        lenmap = self.synthesis_general(lmax=lmax, mmax=mmax, pointmap=lenmap, loc=np.array([pointing_theta, pointing_phi]).T, spin=spin, epsilon=epsilon, nthreads=nthreads, mode=ducc_sht_mode(gclm, spin), alm=gclm, verbose=verbosity)
        lenmap = self.rotate(lenmap, polrot, spin, nthreads)
        
        if self.execmode == 'debug':
            return self.ret
        else:
            return lenmap.real if spin == 0 else lenmap.view(rtype[lenmap.dtype]).reshape((lenmap.size, 2)).T

    @timing_decorator
    def lenmap2gclm(self, lenmap:np.ndarray[complex or float], dlm, gclm_out, spin:int, lmax:int, mmax:int, nthreads:int, epsilon=None, ptg=None, verbosity=1, execmode='normal'):
        self.timer.start('lenmap2gclm()')
        
        self.single_prec = True if epsilon>1e-6 else False
        self._assert_shape(lenmap, gclm_out, ndim=2)
        self._assert_precision(lenmap, gclm_out, epsilon)
        
        lenmap = self._ensure_dtype_nuFFT(lenmap)
        lenmap = self._ensure_shape(lenmap)
        gclm_out = self._ensure_shape(gclm_out)
        dlm = self._ensure_shape(dlm)
        
        def setup(self, nthreads):
            assert execmode in ['normal', 'debug', 'timing']
            print('Running in {} execution mode'.format(execmode))
            self.execmode = execmode
            self.deflectionlib.execmode = self.execmode
            self.nthreads = self.nthreads if nthreads is None else nthreads
        
        setup(self, nthreads)
        
        if ptg is None:
            pointing_theta, pointing_phi = self.deflectionlib.dlm2pointing(dlm, mmax_dlm=lmax, single_prec=self.single_prec, nthreads=nthreads)
        else:
            pointing_theta, pointing_phi = ptg[:,0], ptg[:,1]
        if self.execmode == 'debug':
            self.ret.update({'dlm2pointing': np.array([pointing_theta, pointing_phi])})
        pointing_theta = self._ensure_dtype(pointing_theta)
        pointing_phi = self._ensure_dtype(pointing_phi)
        
        gclm = self.adjoint_synthesis_general(lmax=lmax, mmax=mmax, pointmap=lenmap, loc=np.array([pointing_theta, pointing_phi]).T, spin=spin, epsilon=epsilon, nthreads=nthreads, mode=ducc_sht_mode(gclm_out, spin), alm=gclm_out, verbose=verbosity)
        
        if self.execmode == 'timing':
            self.timer.close('lenmap2gclm()')
            self.timer.dumpjson(os.path.dirname(cunusht.__file__)[:-5]+'/test/benchmark/timings/lenmap2gclm/CPU_duccnufft_{}_e{}'.format(lmax, epsilon))
            print(self.timer)
            print("::timing:: stored new timing data for lmax {}".format(lmax))
        if self.execmode == 'debug':
            print("::debug:: Returned component results")
            return self.ret
        del lenmap, pointing_theta, pointing_phi
        return gclm
    
    def gclm2lenpixs(self, gclm:np.ndarray, mmax:int or None, spin:int, pixs:np.ndarray[int], polrot=True, ptg=None, nthreads=10):
        """Produces the remapped field on the required lensing geometry pixels 'exactly', by brute-force calculation
            Note:
                The number of pixels must be small here, otherwise way too slow
            Note:
                If the remapping angles etc were not calculated previously, it will build the full map, so may take some time.
        """
        assert spin >= 0, spin
        gclm = np.atleast_2d(gclm)
        sth_mode = ducc_sht_mode(gclm, spin)
        if ptg is None:
            ptg = self._get_ptg()
        thts, phis = ptg[0, pixs], ptg[1, pixs]
        nph = 2 * np.ones(thts.size, dtype=np.uint64)  # I believe at least 2 points per ring
        ofs = 2 * np.arange(thts.size, dtype=np.uint64)
        wt = np.ones(thts.size, dtype=float)
        from lenspyx.remapping.utils_geom import Geom as LG
        geom = LG(thts.copy(), phis.copy(), nph, ofs, wt)
        gclm = np.atleast_2d(gclm)
        lmax = Alm.getlmax(gclm[0].size, mmax)
        if mmax is None: mmax = lmax
        m = geom.synthesis(gclm, spin, lmax, mmax, nthreads, mode=sth_mode)[:, 0::2]
        # could do: complex view trick etc
        if spin and polrot:
            gamma = self._get_gamma()[pixs]
            m = np.exp(1j * spin * gamma) * (m[0] + 1j * m[1])
            return m.real, m.imag
        return m.squeeze()

class CPU_Lenspyx_transformer:
    def __init__(self, geominfo_deflection, dglm, mmax_dlm, nthreads, verbosity, epsilon, single_prec):
        self.timer = timer
        self.timer.reset_ti()
        self.timer.start(self.__class__.__name__)
        self.epsilon = epsilon
        self.lenspyx_geom = get_lenspyxgeom(geominfo_deflection)
        self.deflectionlib = lenspyx_deflection(
            lens_geom=self.lenspyx_geom,
            dglm=dglm,
            mmax_dlm=mmax_dlm,
            numthreads=nthreads,
            verbosity=verbosity,
            epsilon=epsilon,
            single_prec=single_prec)
        
        self.backend = 'CPU'
        self.execmode = None
        
        self.ret = {}
        self.timer.add('init')
    
    def _ensure_dtype(self, item):
        if self.deflectionlib.single_prec:
            item = item.astype(np.float32)
        else:
            item = item.astype(np.double)
        return item
    
    def _ensure_shape(self, item):
        return np.atleast_2d(item)
    
    def _ensure_complexdtype(self, item):
        if self.deflectionlib.single_prec:
            item = item.astype(np.complex64)
        else:
            item = item.astype(np.complex128)
        return item
        
    def _assert_shape(self, pointmap, gclm, dlm, ndim):
        """ no-batch shape check.
        """
        assert len(pointmap.shape) == ndim, len(pointmap.shape)
        assert len(gclm.shape) == ndim, len(gclm.shape)
        assert len(dlm.shape) == ndim, len(dlm.shape)
        
    def _assert_precision(self, lenmap, gclm_out):
        if self.single_prec:
            assert lenmap.dtype in [np.float32], "lenmap must be single precision"
            assert gclm_out.dtype in [np.complex64], "gclm_out must be single precision"
            assert self.deflectionlib.epsilon>1e-6, "epsilon must be > 1e-6 for single precision"
        else:
            assert lenmap.dtype in [np.float64], "lenmap must be double precision"
            assert gclm_out.dtype in [np.complex128], "gclm_out must be double precision"
            assert self.deflectionlib.epsilon<=1e-6, "epsilon must be > 1e-6 for double precision"
            
    @timing_decorator
    # @shape_decorator
    @debug_decorator
    def synthesis_general(self, lmax, mmax, alm, loc, spin, epsilon, nthreads, mode, verbose):
        return synthesis_general(lmax=lmax, mmax=mmax, alm=alm, loc=loc, spin=spin, epsilon=epsilon, nthreads=nthreads, mode=mode, verbose=verbose)
  
    @timing_decorator
    # @shape_decorator
    @debug_decorator
    def adjoint_synthesis_general(self, lmax, mmax, pointmap, loc, spin, epsilon, mode, nthreads, alm, verbose):
        return adjoint_synthesis_general(lmax=lmax, mmax=mmax, map=pointmap, loc=loc, spin=spin, epsilon=epsilon, mode=mode, nthreads=nthreads, alm=alm, verbose=verbose)

    @timing_decorator
    # @shape_decorator
    @debug_decorator      
    def dlm2pointing(self):
        return self.deflectionlib._get_ptg()           
    
    @timing_decorator_close
    def gclm2lenmap(self, gclm:np.ndarray, dlm, lmax, mmax:int or None, spin:int, nthreads, backwards:bool=False, polrot=True, ptg=None, dclm=None, lenmap=None, execmode='normal'):

        def setup(self, nthreads):
            assert execmode in ['normal','debug', 'timing']
            print('Running in {} execution mode'.format(execmode))
            self.execmode = execmode
            self.deflectionlib.execmode = self.execmode
            self.deflectionlib.nthreads = self.nthreads if nthreads is None else nthreads
        
        gclm = self._ensure_complexdtype(gclm)
        gclm = self._ensure_shape(gclm)
        if lenmap is not None:
            lenmap = self._ensure_dtype(lenmap)
            self._assert_precision(lenmap, gclm)
        setup(self, nthreads)
        
        if ptg is None:
            ptg = self.dlm2pointing()
            ptg = np.array(ptg, dtype=np.float64) if not self.deflectionlib.single_prec else np.array(ptg, dtype=np.float32)
        lenmap = self.synthesis_general(lmax, mmax, gclm, ptg, spin, self.deflectionlib.epsilon, nthreads, ducc_sht_mode(gclm, spin), self.deflectionlib.verbosity)
        if self.execmode == 'debug':
            print("::debug:: returned component results")
            return self.ret
        return lenmap

    @timing_decorator_close
    def lenmap2gclm(self, lenmap:np.ndarray[complex or float], dlm:np.ndarray, spin:int, lmax:int, mmax:int, nthreads:int, gclm_out=None, ptg=None, execmode='normal'):

        def setup(self, nthreads):
            assert execmode in ['normal','debug', 'timing']
            print('Running in {} execution mode'.format(execmode))
            self.execmode = execmode
            self.deflectionlib.execmode = self.execmode
            self.nthreads = self.nthreads if nthreads is None else nthreads
        
        lenmap = self._ensure_dtype(lenmap)
        gclm_out = self._ensure_complexdtype(gclm_out)
        self._assert_precision(lenmap, gclm_out)
        
        setup(self, nthreads)
        
        if ptg is None:
            ptg = self.dlm2pointing()
            ptg = np.array(ptg, dtype=np.float64) if not self.single_prec else np.array(ptg, dtype=np.float32)
        gclm_out = self.adjoint_synthesis_general(lmax=lmax, mmax=mmax, pointmap=lenmap, loc=ptg, mode=ducc_sht_mode(dlm, spin), alm=gclm_out, spin=spin, epsilon=self.epsilon, nthreads=nthreads, verbose=self.verbosity)
       
        if self.execmode == 'debug':
            print("::debug:: Returned component results")
            return self.ret
        return gclm_out


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
        if ptg is None:
            ptg = self._get_ptg()
        thts, phis = ptg[pixs, 0], ptg[pixs, 1]
        nph = 2 * np.ones(thts.size, dtype=np.uint64)  # I believe at least 2 points per ring
        ofs = 2 * np.arange(thts.size, dtype=np.uint64)
        wt = np.ones(thts.size, dtype=float)
        geom = Geom(thts.copy(), phis.copy(), nph, ofs, wt)
        gclm = np.atleast_2d(gclm)
        lmax = Alm.getlmax(gclm[0].size, mmax)
        if mmax is None: mmax = lmax
        m = geom.synthesis(gclm, spin, lmax, mmax, self.nthreads, mode=sth_mode)[:, 0::2]
        # could do: complex view trick etc
        if spin and polrot:
            gamma = self._get_gamma()[pixs]
            m = np.exp(1j * spin * gamma) * (m[0] + 1j * m[1])
            return m.real, m.imag
        return m.squeeze()

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
        ret = {}
        
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
        
        dmap = np.empty((2 * ntheta - 2, nphi), dtype=np.complex128 if spin == 0 else ctype[map.dtype])
        if spin == 0:
            dmap[:ntheta, :] = map[0]
        else:
            dmap[:ntheta, :].real = map[0]
            dmap[:ntheta, :].imag = map[1]
        del map
        dmap[ntheta:, :nphihalf] = dmap[ntheta - 2:0:-1, nphihalf:]
        dmap[ntheta:, nphihalf:] = dmap[ntheta - 2:0:-1, :nphihalf]
        if (spin % 2) != 0:
            dmap[ntheta:, :] *= -1
        self.timer.add('doubling')
        if debug:
            ret.append(np.copy(dmap))


        # go to Fourier space
        if spin == 0:
            tmp = np.empty(dmap.shape, dtype=np.complex128)
            dmap = ducc0.fft.c2c(dmap, axes=(0, 1), inorm=2, nthreads=nthreads, out=tmp)
            del tmp
        else:
            dmap = ducc0.fft.c2c(dmap, axes=(0, 1), inorm=2, nthreads=nthreads, out=dmap)
        self.timer.add('c2c')
        if debug:
            ret.append(np.copy(dmap))
        
        if self.planned: # planned nufft
            assert ptg is None
            plan = self.make_plan(lmax_unl, spin)
            values = plan.u2nu(grid=dmap, forward=False, verbosity=self.verbosity)
        else:
            ptg = None
            if ptg is None:
                ptg = self._get_ptg(dlm, mmax)
            self.timer.add('get ptg')
            if debug:
                ret.append(np.copy(ptg))
                
            map_shifted = np.fft.fftshift(dmap, axes=(0,1))
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
            self.timer.dumpjson('/mnt/home/sbelkner/git/cunusht/test/benchmark/timings/CPU_finufft_{}'.format(lmax))
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
        dmap = np.empty((2 * ntheta - 2, nphi), dtype=points.dtype)
        if self.planned:
            plan = self.make_plan(lmax, spin)
            dmap = plan.nu2u(points=points, out=dmap, forward=True, verbosity=self.verbosity)

        else:
            # perform NUFFT
        
            dmap = ducc0.nufft.nu2u(points=points, coord=ptg, out=dmap, forward=True,
                                       epsilon=self.epsilon, nthreads=nthreads, verbosity=self.verbosity,
                                       periodicity=2 * np.pi, fft_order=True)
        # go to position space
        dmap = ducc0.fft.c2c(dmap, axes=(0, 1), forward=False, inorm=2, nthreads=nthreads, out=dmap)

        # go from double Fourier sphere to Clenshaw-Curtis grid
        if (spin % 2) != 0:
            dmap[1:ntheta - 1, :nphihalf] -= dmap[-1:ntheta - 1:-1, nphihalf:]
            dmap[1:ntheta - 1, nphihalf:] -= dmap[-1:ntheta - 1:-1, :nphihalf]
        else:
            dmap[1:ntheta - 1, :nphihalf] += dmap[-1:ntheta - 1:-1, nphihalf:]
            dmap[1:ntheta - 1, nphihalf:] += dmap[-1:ntheta - 1:-1, :nphihalf]
        dmap = dmap[:ntheta, :]
        map = np.empty((1 if spin == 0 else 2, ntheta, nphi), dtype=rtype[points.dtype])
        map[0] = dmap.real
        if spin > 0:
            map[1] = dmap.imag
        del dmap

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
        return synthesis_general(lmax=lmax, mmax=mmax, alm=alm, loc=loc, spin=spin, epsilon=self.epsilon, nthreads=self.nthreads, mode=sht_mode, verbose=self.verbosity)
    
    def adjoint_synthesis_general(self, lmax, mmax, map, loc, spin, epsilon, nthreads, sht_mode, alm, verbose):
        assert 0, "implement if needed"
        return adjoint_synthesis_general(lmax=lmax, mmax=mmax, map=map, loc=loc, spin=spin, epsilon=self.epsilon, nthreads=self.nthreads, mode=sht_mode, alm=alm, verbose=self.verbosity)

