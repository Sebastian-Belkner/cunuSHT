import numpy as np
import os, sys

from lenspyx.utils_hp import Alm, alm2cl, almxfl, alm_copy
from lenspyx.remapping.utils_angles import d2ang

import functools
import cupy as cp
import time
import healpy as hp
import ctypes
import line_profiler
import inspect

import cupyx.scipy.fft as cufft
import scipy.fft
scipy.fft.set_global_backend(cufft)

import ducc0
import cufinufft

import pysht
from pysht import cacher
import pysht.c.podo_interface as podo
import pysht.geometry as geometry
from pysht.geometry import Geom
from pysht.utils import timer
from pysht.helper import shape_decorator, debug_decorator, timing_decorator
from pysht.sht.GPU_sht_transformer import GPU_SHT_pySHT_transformer, GPU_SHTns_transformer
from pysht.sht.CPU_sht_transformer import CPU_SHT_DUCC_transformer

ctype = {np.dtype(np.float32): np.complex64,
         np.dtype(np.float64): np.complex128,
         np.dtype(np.longfloat): np.longcomplex,
         np.float32: np.complex64,
         np.float64: np.complex128,
         np.longfloat: np.longcomplex}

class deflection:
    def __init__(self, shttransformer_desc, dlm, mmax_dlm:int, geominfo, dclm:np.ndarray=None, epsilon=1e-10, verbosity=0, nthreads=10, single_prec=False, timer=None):
        if timer is not None:
            self.timer = timer
            
        sht_kwargs = {
            'geominfo': geominfo,
            'verbosity': verbosity,
            'single_prec': single_prec,
            'nthreads': nthreads
        } 
        if shttransformer_desc == 'shtns':
            self.BaseClass = type('GPU_SHTns_transformer', (GPU_SHTns_transformer,), {})
            self.instance = self.BaseClass(**sht_kwargs)
        elif shttransformer_desc == 'ducc':
            self.BaseClass = type('CPU_SHT_DUCC_transformer()', (CPU_SHT_DUCC_transformer,), {})
            self.instance = self.BaseClass(**sht_kwargs)
        elif shttransformer_desc == 'pysht':
            assert 0, "implement if needed"
            self.BaseClass = type('GPU_SHT_pySHT_transformer', (GPU_SHT_pySHT_transformer,), {})
            self.instance = self.BaseClass(**sht_kwargs)
        else:
            raise ValueError('shttransformer_desc must be either "ducc" or "shtns" or "pysht"')
        
        self.single_prec = single_prec
        self.verbosity = verbosity
        self.nthreads = nthreads
        self.cacher = cacher.cacher_mem()
        self.epsilon = epsilon
        
        dlm = np.atleast_2d(dlm)        
        self.lmax_dlm = Alm.getlmax(dlm[0].size, mmax_dlm)
        self.mmax_dlm = mmax_dlm
        
        self.geominfo = geominfo
        self.geom = geometry.get_geom(geominfo)
        
        s2_d = np.sum(alm2cl(dlm[0], dlm[0], self.lmax_dlm, mmax_dlm, self.lmax_dlm) * (2 * np.arange(self.lmax_dlm + 1) + 1)) / (4 * np.pi)
        if dlm.shape[0]>1:
            s2_d += np.sum(alm2cl(dlm[1], dlm[1], self.lmax_dlm, mmax_dlm, self.lmax_dlm) * (2 * np.arange(self.lmax_dlm + 1) + 1)) / (4 * np.pi)
            s2_d /= np.sqrt(2.)
        sig_d = np.sqrt(s2_d / self.geom.fsky())
        sig_d_amin = sig_d / np.pi * 180 * 60
        if sig_d >= 0.01:
            print('deflection std is %.2e amin: this is really too high for something sensible'%sig_d_amin)
        elif self.verbosity:
            print('deflection std is %.2e amin' % sig_d_amin)
    
    def __getattr__(self, name):
        return getattr(self.instance, name)


    # @debug_decorator
    @timing_decorator
    # @shape_decorator
    def dlm2pointing(self, dlm_scaled, pointing_theta, pointing_phi):
        
        # @debug_decorator
        @timing_decorator
        # @shape_decorator
        def _spin__1___synth(self, dlm_scaled, out_theta, out_phi):
            self.synthesis_der1_cupy(dlm_scaled, out_theta, out_phi, nthreads=self.nthreads)
            return tuple([out_theta, out_phi])
          
        # @debug_decorator
        @timing_decorator
        @shape_decorator
        def _pointing(self, spin1_theta, spin1_phi, cpt, cpphi0, cpnph, cpofs, pointing_theta, pointing_phi):
            podo.Cpointing_1Dto1D(cpt, cpphi0, cpnph, cpofs, spin1_theta, spin1_phi, pointing_theta, pointing_phi)
            return tuple([pointing_theta, pointing_phi])
        
        dlm_scaled = cp.array(dlm_scaled, dtype=np.complex)
        cpt = cp.array(self.geom.theta.astype(np.double), dtype=cp.double)
        cpphi0 = cp.array(self.geom.phi0, dtype=cp.double)
        cpnph = cp.array(self.geom.nph, dtype=cp.uint64)
        cpofs = cp.array(self.geom.ofs, dtype=cp.uint64)
        spin1_theta = cp.zeros(self.constructor.spat_shape, dtype=cp.double)
        spin1_phi = cp.zeros(self.constructor.spat_shape, dtype=cp.double)
        self.timer.add('dlm2pointing - allocation')

        _spin__1___synth(self, dlm_scaled, spin1_theta, spin1_phi)
        _pointing(self, spin1_theta.T.flatten(), spin1_phi.T.flatten(), cpt, cpphi0, cpnph, cpofs, pointing_theta, pointing_phi)
        
        del spin1_theta, spin1_phi, cpt, cpphi0, cpnph, cpofs, dlm_scaled
        return tuple([pointing_theta, pointing_phi])


class GPU_cufinufft_transformer:
    def __init__(self, shttransformer_desc, geominfo, single_prec, epsilon, nthreads, verbosity, planned, deflection_kwargs):
        self.backend = 'GPU'
        self.shttransformer_desc = shttransformer_desc
        self.single_prec = single_prec
        self.epsilon = epsilon
        self.nthreads = nthreads
        self.verbosity = verbosity
        self.planned = planned
        
        self.execmode = None
        
        sht_kwargs = {
            'geominfo': geominfo,
            'verbosity': verbosity,
            'single_prec': single_prec,
            'nthreads': nthreads
        } 
        if shttransformer_desc == 'shtns':
            self.BaseClass = type('GPU_SHTns_transformer', (GPU_SHTns_transformer,), {})
            self.instance = self.BaseClass(**sht_kwargs)
        elif shttransformer_desc == 'ducc':
            self.BaseClass = type('CPU_SHT_DUCC_transformer()', (CPU_SHT_DUCC_transformer,), {})
            self.instance = self.BaseClass(**sht_kwargs)
        elif shttransformer_desc == 'pysht':
            assert 0, "implement if needed"
            self.BaseClass = type('GPU_SHT_pySHT_transformer', (GPU_SHT_pySHT_transformer,), {})
            self.instance = self.BaseClass(**sht_kwargs)
        else:
            raise ValueError('shttransformer_desc must be either "ducc" or "shtns" or "pysht"')
            
        self.geominfo = geominfo
        self.set_geometry(geominfo)
        
        self.timer = timer(1, prefix=self.backend)
        self.deflectionlib = deflection(shttransformer_desc, **deflection_kwargs, timer=self.timer)
        
        # Take ducc good_size, but adapt for good size needed by GPU SHTns (nlat must be multiple of 4)
        self.ntheta_CAR = (ducc0.fft.good_size(geominfo[1]['lmax'] + 2) + 3) // 4 * 4
        self.nphihalf_CAR = ducc0.fft.good_size(geominfo[1]['lmax'] + 1)
        self.nphi_CAR = 2 * self.nphihalf_CAR
        self.geominfo_CAR = ('cc',{'lmax': geominfo[1]['lmax'], 'mmax':geominfo[1]['lmax'], 'ntheta':self.ntheta_CAR, 'nphi':self.nphi_CAR})
        self.cc_transformer = pysht.get_transformer('shtns', 'SHT', 'GPU')(self.geominfo_CAR)


    def _assert_type(self, lenmap, gclm_out, dlm):
        assert isinstance(lenmap, cp.ndarray), "only accepting cupy arrays here, {} is {}".format("lenmap", type(lenmap))
        assert isinstance(gclm_out, cp.ndarray), "only accepting cupy arrays here, {} is {}".format("gclm_out", type(gclm_out))
        assert isinstance(dlm, cp.ndarray), "only accepting cupy arrays here, {} is {}".format("dlm", type(dlm))

    def _assert_dtype(self, lenmap, gclm_out, dlm):
        assert lenmap.dtype in [cp.float32, cp.float64, cp.complex64, cp.complex128], "{} is {}".format("lenmap", lenmap.dtype)
        assert gclm_out.dtype in [cp.float32, cp.float64, cp.complex64, cp.complex128], "{} is {}".format("gclm_out", gclm_out.dtype)
        assert dlm.dtype in [cp.complex64, cp.complex128], "{} is {}".format("dlm", dlm.dtype)
        if self.single_prec:
            assert lenmap.dtype in [cp.float32, cp.complex64], "{} is {}".format(lenmap, lenmap.dtype)
            assert gclm_out.dtype in [cp.float32, cp.complex64], "{} is {}".format(gclm_out, gclm_out.dtype)
        if self.deflectionlib.single_prec:
            assert dlm.dtype in [cp.complex64], "{} is {}".format(dlm, dlm.dtype)

    def _assert_shape(self, lenmap, gclm_out, dlm, ndim, nbatch):
        """ batched shape check, in leading dimension
        """
        assert len(lenmap.shape) == ndim, len(lenmap.shape)
        assert lenmap.shape[0] == nbatch, lenmap.shape[0]
        
        assert len(gclm_out.shape) == ndim, len(gclm_out.shape)
        assert gclm_out.shape[0] == nbatch, gclm_out.shape[0]
        
        # assert len(gclm_out.shape) == len(lenmap.shape), "lenmap and gclm_out must have same number of dimensions"
        # assert lenmap.shape == gclm_out.shape, "lenmap and gclm_out must have same shape"
        
        assert len(dlm.shape) == ndim, len(dlm.shape)
        assert dlm.shape[0] == nbatch, dlm.shape[0]
        
    def _ensure_dtype(self, item, single_prec, isreal):
        if single_prec:
            if isreal:
                return item.astype(np.float32)
            else:
                return item.astype(np.complex64)
        else:
            if isreal:
                return item.astype(np.float64)
            else:
                return item.astype(np.complex128)
  
    def _assert_precision(self, lenmap, gclm_out):
        assert lenmap.dtype == gclm_out.dtype, "lenmap and gclm_out must have same precision"
        if self.single_prec:
            assert lenmap.dtype in [cp.float32, cp.complex64], "lenmap must be single precision"
            assert gclm_out.dtype in [cp.complex64], "gclm_out must be single precision"
            assert self.epsilon>1e-6, "epsilon must be > 1e-6 for single precision"

    def __getattr__(self, name):
        return getattr(self.instance, name)
        

    def gclm2lenmap(self, gclm, dlm, lmax, mmax, spin, pointing_theta:cp.array = None, pointing_phi:cp.array = None, nthreads=None, polrot=True, execmode='normal'):
        """
        Same as gclm2lenmap, but using cupy allocated intermediate results (synth, doubling, c2c, nuFFt),
        so no transfers needed to CPU between them.
        
        gclm and dlm are assumed to be on host, will be transfered in _setup().
        Can provide pointing_theta and pointing_phi to avoid dlm2pointing() call.
        """
        self.ret = {}
            
        # @timing_decorator
        def _setup(self, gclm, execmode, nthreads):
            assert execmode in ['normal', 'debug', 'timing']
            print('Running in {} execution mode'.format(execmode))
            nthreads = self.nthreads if nthreads is None else nthreads
            gclm = np.atleast_2d(gclm)
            if self.single_prec and gclm.dtype != np.complex64:
                gclm = gclm.astype(np.complex64)
            gclm = cp.array(gclm, dtype=np.complex)
            self.execmode = execmode
            self.deflectionlib.execmode = self.execmode
            return gclm
        
        @debug_decorator
        @timing_decorator
        @shape_decorator
        def _synthesis(self, gclm, out):
            self.cc_transformer.synthesis_cupy(gclm, out, spin=0, lmax=lmax, mmax=mmax, nthreads=nthreads)
            return tuple([out])

        @debug_decorator
        @timing_decorator
        @shape_decorator
        def _doubling(self, map_in, ntheta, nphi, out):
            podo.Cdoubling_1D(map_in, ntheta, nphi, out)
            return tuple([out])
        
        @debug_decorator
        @timing_decorator
        @shape_decorator
        def _C2C(self, map_in, out):
            out = scipy.fft.fft2(map_in, norm='forward')
            return tuple([out])
        
        @debug_decorator
        @timing_decorator
        @shape_decorator
        def _nuFFT(self, fc, ptg_theta, ptg_phi, result):
            result = cufinufft.nufft2d2(data=fc, x=ptg_theta, y=ptg_phi, isign=1, eps=self.epsilon)
            return tuple([result])

        self.timer = timer(1, prefix=self.backend)
        self.timer.start('gclm2lenmap_cupy()')
        self.timing, self.debug = None, None
        gclm = _setup(self, gclm, execmode, nthreads)
        ntheta_dCAR, nphi_dCAR = 2*self.ntheta_CAR-2, self.nphi_CAR
        CARmap = cp.empty((self.ntheta_CAR*self.nphi_CAR), dtype=np.double)
        CARdmap = cp.zeros((2*self.cc_transformer.constructor.nlat-2)*self.cc_transformer.constructor.nphi, dtype=np.double)
        self.timer.add('Transfers ->')
        fc, lenmap = None, None #TODO decide if these preallocated or not
        
        @debug_decorator
        @timing_decorator
        @shape_decorator
        def dlm2pointing(self, dlm, pointing_theta, pointing_phi):
            self.deflectionlib.dlm2pointing(dlm, pointing_theta, pointing_phi)
            return tuple([pointing_theta, pointing_phi])
            
        if pointing_theta is None or pointing_phi is None:
            pointing_theta = cp.zeros((self.deflectionlib.geom.npix()), dtype=cp.double)
            pointing_phi = cp.zeros((self.deflectionlib.geom.npix()), dtype=cp.double)
            dlm2pointing(self, dlm, pointing_theta, pointing_phi)

        _synthesis(self, gclm, CARmap)
        del gclm
        _doubling(self, CARmap.reshape(self.nphi_CAR,-1).T.flatten(), int(ntheta_dCAR), int(nphi_dCAR), CARdmap)
        del CARmap
        fc = _C2C(self, CARdmap.reshape(ntheta_dCAR,-1).T, fc)[0]
        del CARdmap
        fc = cufft.fftshift(fc, axes=(0,1))
        self.timer.add('FFTshift')
        print('dtypes before ensuring; ', pointing_theta.dtype, pointing_phi.dtype)
        pointing_theta = self._ensure_dtype(pointing_theta, self.single_prec, isreal=True)
        pointing_phi = self._ensure_dtype(pointing_phi, self.single_prec, isreal=True)
        print('dtypes after ensuring: ', pointing_theta.dtype, pointing_phi.dtype)
        result = cufinufft.nufft2d2(x=pointing_phi, y=pointing_theta, data=fc, isign=1)
        self.timer.add("nuFFT init")
        pointing_theta = self._ensure_dtype(pointing_theta, self.single_prec, isreal=True)
        pointing_phi = self._ensure_dtype(pointing_phi, self.single_prec, isreal=True)
        lenmap = _nuFFT(self, fc, pointing_phi, pointing_theta, lenmap)
        self.timer.add('gclm2lenmap')
        result = lenmap[0].get()
        self.timer.add('Transfer <-')
        
        if self.execmode == 'timing':
            self.timer.dumpjson(os.path.dirname(pysht.__file__)[:-5]+'/test/benchmark/timings/gclm2lenmap/GPU_cufinufft_{}_e{}'.format(lmax, self.epsilon))
            print(self.timer)
            print("::timing:: stored new timing data for lmax {}".format(lmax))
        if self.execmode == 'debug':
            print("::debug:: Returned component results")
            return self.ret
        del fc, lenmap, pointing_theta, pointing_phi
        return result


    @debug_decorator
    @timing_decorator
    # @shape_decorator
    def dlm2pointing(self, dlm_scaled):
        pointing_theta = cp.zeros((self.deflectionlib.geom.npix()), dtype=cp.float32) if self.deflectionlib.single_prec else cp.zeros((self.deflectionlib.geom.npix()), dtype=cp.double)
        pointing_phi = cp.zeros((self.deflectionlib.geom.npix()), dtype=cp.float32) if self.deflectionlib.single_prec else cp.zeros((self.deflectionlib.geom.npix()), dtype=cp.double)
        self.deflectionlib.dlm2pointing(dlm_scaled, pointing_theta, pointing_phi)
        return tuple([pointing_theta, pointing_phi])
        
    @debug_decorator
    @timing_decorator
    # @shape_decorator
    def nuFFT2d1(self, lenmap, ptg_theta, ptg_phi): #.reshape(self.geom.nph[0],-1).T.flatten()
        fc = cufinufft.nufft2d1(data=lenmap, x=ptg_theta, y=ptg_phi, n_modes=(2*self.ntheta_CAR-2,self.nphi_CAR), isign=-1, eps=self.epsilon)
        return fc
    
    @debug_decorator
    @timing_decorator
    # @shape_decorator
    def iC2C(self, fc):
        map_out = scipy.fft.ifft2(fc[0], norm='backward')
        return map_out
    
    @debug_decorator
    @timing_decorator
    # @shape_decorator        
    def adjoint_doubling(self, CARdmap):
        CARmap = cp.empty(shape=(self.ntheta_CAR*self.nphi_CAR), dtype=np.float32) if self.single_prec else cp.empty(shape=(self.ntheta_CAR*self.nphi_CAR), dtype=np.double)
        podo.Cadjoint_doubling_1D(CARdmap, int(self.ntheta_CAR), int(self.nphi_CAR), CARmap)
        return CARmap

    @debug_decorator
    @timing_decorator
    # @shape_decorator
    def adjoint_synthesis(self, synthmap, lmax, mmax, out):
        out = self.cc_transformer.adjoint_synthesis_cupy(synthmap, gclm=out, spin=0, lmax=lmax, mmax=mmax, nthreads=self.nthreads)
        return out
    

    def lenmap2gclm(self, lenmap:cp.ndarray, dlm_scaled:cp.ndarray, gclm_out:cp.ndarray, spin:int, lmax:int, mmax:int, nthreads:int, ptg=None, execmode='normal'):
        """
            Note:
                For inverse-lensing, need to feed in lensed maps times unlensed forward magnification matrix.
                lenmap must be theta contiguous
        """
        self.ret = {}
        
        self.timer = timer(1, prefix=self.backend)
        self.timer.start('lenmap2gclm()')
        
        self._assert_shape(lenmap, gclm_out, dlm_scaled, ndim=2, nbatch=1)
        self._assert_type(lenmap, gclm_out, dlm_scaled)
        self._assert_dtype(lenmap, gclm_out, dlm_scaled)
        self._assert_precision(lenmap, gclm_out)
        
        def setup(self):
            assert execmode in ['normal', 'debug', 'timing']
            print('Running in {} execution mode'.format(execmode))
            self.execmode = execmode
            self.deflectionlib.execmode = self.execmode
  
        setup(self)
        
        if ptg is None:
            pointing_theta, pointing_phi = self.dlm2pointing(dlm_scaled)
        else:
            pointing_theta, pointing_phi = ptg.T
        pointing_theta = self._ensure_dtype(pointing_theta, self.single_prec, isreal=True)
        pointing_phi = self._ensure_dtype(pointing_phi, self.single_prec, isreal=True)
        
        ptg = cp.array([pointing_theta, pointing_phi])
        gclm = self.adjoint_synthesis_general(lmax, mmax, lenmap, ptg, spin, self.epsilon, nthreads, gclm_out, self.verbosity)
        
        if self.execmode == 'timing':
            self.timer.dumpjson(os.path.dirname(pysht.__file__)[:-5]+'/test/benchmark/timings/lenmap2gclm/GPU_cufinufft_{}_e{}'.format(lmax, self.epsilon))
            print(self.timer)
            print("::timing:: stored new timing data for lmax {}".format(lmax))
        if self.execmode == 'debug':
            print("::debug:: Returned component results")
            return self.ret
        del lenmap, pointing_theta, pointing_phi
        return gclm


    def synthesis_general(self, lmax, mmax, map, loc, spin, epsilon, nthreads, sht_mode, alm, verbose):
        assert 0, "implement if needed"
        # return self.gclm2lenmap_cupy(lmax=lmax, mmax=mmax, alm=alm, pointing_theta=loc[0], pointing_phi=loc[1], spin=spin, epsilon=self.epsilon, nthreads=self.nthreads, mode=sht_mode, verbose=self.verbosity)
    
    def adjoint_synthesis_general(self, lmax, mmax, pointmap, loc, spin, epsilon, nthreads, alm, verbose):
        # TODO use nthreads, spin, verbose
        pointing_theta, pointing_phi = loc
        cufinufft.nufft2d1(data=pointmap, x=pointing_phi, y=pointing_theta, n_modes=(self.nphi_CAR, 2*self.ntheta_CAR-2), isign=-1, eps=epsilon)
        self.timer.add("nuFFT init")
        
        fc = self.nuFFT2d1(pointmap, pointing_theta, pointing_phi)
        CARdmap = self.iC2C(cufft.fftshift(fc, axes=(1,2)))
        synthmap = self.adjoint_doubling(CARdmap.real.flatten())
        synthmap = synthmap.reshape(-1,self.nphi_CAR).T.flatten()
        alm = self.adjoint_synthesis(synthmap=synthmap, lmax=lmax, mmax=mmax, out=alm)
        alm = alm[0].get()
        self.timer.add('Transfer <-')
        return alm


    def flip_tpg_2d(self, m):
        # FIXME this should probably be lmax, not lmax_dlm
        # dim of m supposedly (2, -1)
        buff = np.array([_.reshape(2*(self.lmax_dlm+1),-1).T.flatten() for _ in m])
        return buff

            
    def hashdict():
        '''
        Compatibility with delensalot
        '''
        return "GPU_cufinufft_transformer"

