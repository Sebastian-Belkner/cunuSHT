import numpy as np
import os, sys

from lenspyx.utils_hp import Alm, alm2cl, almxfl, alm_copy
from lenspyx.remapping.utils_angles import d2ang

import cupy as cp
import time
import healpy as hp
import line_profiler
from cupyx.scipy.fft import get_fft_plan
import cupyx
import cupyx.scipy.fft as cufft


from cufinufft import Plan, _compat 
import ducc0
import cufinufft

import pysht
import pysht.c.podo_interface as podo
import pysht.geometry as geometry
from pysht.geometry import Geom
from pysht.utils import timer as tim

from pysht.helper import shape_decorator, debug_decorator, timing_decorator, timing_decorator_close
from pysht.sht.GPU_sht_transformer import GPU_SHT_pySHT_transformer, GPU_SHTns_transformer
from pysht.sht.CPU_sht_transformer import CPU_SHT_DUCC_transformer

ctype = {True: np.complex64, False: np.complex128}
rtype = {True: np.float32, False: np.float64}

timer = tim(1, prefix='GPU')

class deflection:
    def __init__(self, geominfo, shttransformer_desc='shtns', timer_instance=None, SHTbaseclass=None):
        if timer_instance is None:
            self.timer = timer
        else:
            self.timer = timer_instance
        
        self.geominfo = geominfo
        self.geom = geometry.get_geom(geominfo)

        self.instance = SHTbaseclass(geominfo=geominfo)
        
        # assert np.multiply(*self.constructor.spat_shape) == self.geom.npix(), "{} => {} =/= {}, ({},{}) <= {}".format(self.constructor.spat_shape, np.multiply(*self.constructor.spat_shape), self.geom.npix(), self.geom.nph[0], len(self.geom.ofs), geominfo) # just a check to see if shtns geometry behaves as pysht geometry


    def __getattr__(self, name):
        return getattr(self.instance, name)


    # @debug_decorator
    @timing_decorator
    # @shape_decorator
    def dlm2pointing(self, dlm_scaled, mmax_dlm, nthreads, verbosity, pointing_theta, pointing_phi):
        if False:
            # FIXME this is only true for dlm, not dlm_scaled
            dlm_scaled = np.atleast_2d(dlm_scaled)
            self.lmax_dlm = Alm.getlmax(dlm_scaled[0].size, mmax_dlm)
            self.mmax_dlm = mmax_dlm
            s2_d = np.sum(alm2cl(dlm_scaled[0], dlm_scaled[0], self.lmax_dlm, self.mmax_dlm, self.lmax_dlm) * (2 * np.arange(self.lmax_dlm + 1) + 1)) / (4 * np.pi)
            if dlm_scaled.shape[0]>1:
                s2_d += np.sum(alm2cl(dlm_scaled[1], dlm_scaled[1], self.lmax_dlm, self.mmax_dlm, self.lmax_dlm) * (2 * np.arange(self.lmax_dlm + 1) + 1)) / (4 * np.pi)
                s2_d /= np.sqrt(2.)
            sig_d = np.sqrt(s2_d / self.geom.fsky())
            sig_d_amin = sig_d / np.pi * 180 * 60
            if sig_d >= 0.01:
                print('deflection std is %.2e amin: this is really too high for something sensible'%sig_d_amin)
            elif verbosity:
                print('deflection std is %.2e amin' % sig_d_amin)
        
        # @debug_decorator
        @timing_decorator
        def _spin__1___synth(self, dlm_scaled, out_theta, out_phi):
            self.synthesis_der1_cupy(dlm_scaled, out_theta, out_phi, nthreads=nthreads)
            return cp.array([out_theta, out_phi])
          
        # @debug_decorator
        @timing_decorator
        def _pointing(self, spin1_theta, spin1_phi, cpt, cpphi0, cpnph, cpofs, pointing_theta, pointing_phi):
            return podo.Cpointing_1Dto1D(cpt, cpphi0, cpnph, cpofs, spin1_theta, spin1_phi, pointing_theta, pointing_phi)
        
        dlm_scaled = cp.array(dlm_scaled, dtype=np.complex)
        cpt = cp.array(self.geom.theta.astype(np.double), dtype=cp.double)
        cpphi0 = cp.array(self.geom.phi0, dtype=cp.double)
        cpnph = cp.array(self.geom.nph, dtype=cp.uint64)
        cpofs = cp.array(self.geom.ofs, dtype=cp.uint64)
        spin1_theta = cp.zeros(self.constructor.spat_shape, dtype=cp.double)
        spin1_phi = cp.zeros(self.constructor.spat_shape, dtype=cp.double)

        _spin__1___synth(self, dlm_scaled, spin1_theta, spin1_phi)
        _pointing(self, spin1_theta.T.flatten(), spin1_phi.T.flatten(), cpt, cpphi0, cpnph, cpofs, pointing_theta, pointing_phi)
        
        del spin1_theta, spin1_phi, cpt, cpphi0, cpnph, cpofs, dlm_scaled
        return cp.array([pointing_theta, pointing_phi])


class GPU_cufinufft_transformer:
    def __init__(self, geominfo_deflection, shttransformer_desc='shtns', nuFFTtype=None, epsilon=None):
        """GPU transformer using cufiNUFFT

        Args:
            geominfo_deflection (_type_): If pointing provided, geominfo_deflection is only needed to have consistent lmax and mmax parameters for CAR geometry.
            shttransformer_desc (str, optional): SHT solver choice. Defaults to 'shtns'.
            nuFFTtype (int, optional): Plan FFT and nuFFT before execution, nuFFTtype is either 1 or 2.  Defaults to None, in which case planning is not done

        Raises:
            ValueError: _description_
        """
        self.timer = timer
        self.timer.reset_ti()
        self.timer.start(self.__class__.__name__)
        if nuFFTtype:
            assert epsilon is not None, "epsilon must be provided if planned is True"
            # assert dtype_nuFFT is not None, "dtype_nuFFT must be provided if planned is True"
        self.backend = 'GPU'
        self.shttransformer_desc = shttransformer_desc
        self.nuFFTtype = nuFFTtype
        self.execmode = None
        self.ret = {} # This is for execmode='debug'
        
        # Take ducc good_size, but adapt for good size needed by GPU SHTns (nlat must be multiple of 4)
        self.ntheta_CAR = (ducc0.fft.good_size(geominfo_deflection[1]['lmax'] + 2) + 3) // 4 * 4
        self.nphihalf_CAR = ducc0.fft.good_size(geominfo_deflection[1]['lmax'] + 1)
        self.nphi_CAR = 2 * self.nphihalf_CAR
        self.geominfo_CAR = ('cc',{'lmax': geominfo_deflection[1]['lmax'], 'mmax':geominfo_deflection[1]['lmax'], 'ntheta':self.ntheta_CAR, 'nphi':self.nphi_CAR})
        if shttransformer_desc == 'shtns':
            self.BaseClass = type('GPU_SHTns_transformer', (GPU_SHTns_transformer,), {})
        elif shttransformer_desc == 'ducc':
            self.BaseClass = type('CPU_SHT_DUCC_transformer()', (CPU_SHT_DUCC_transformer,), {})
        elif shttransformer_desc == 'pysht':
            assert 0, "implement if needed"
            self.BaseClass = type('GPU_SHT_pySHT_transformer', (GPU_SHT_pySHT_transformer,), {})
        else:
            raise ValueError('shttransformer_desc must be either "ducc" or "shtns" or "pysht"')
        self.instance = self.BaseClass(geominfo=self.geominfo_CAR)
        self.timer.add('init - SHTlib')
        
        w = self.constructor.gauss_wts()
        w = np.hstack((w, np.flip(w)))
        self.iw = cp.array(1/w)
        self.timer.add('init - weights')
        
        self.ntheta_dCAR, self.nphi_dCAR = int(2*self.ntheta_CAR-2), int(self.nphi_CAR)
        self.CARmap = cp.empty((self.ntheta_CAR*self.nphi_CAR), dtype=np.double)
        self.CARdmap = cp.empty((self.ntheta_dCAR*self.nphi_dCAR), dtype=np.double)
        self.timer.add('init - allocation')
        
        self.deflectionlib = deflection(geominfo=geominfo_deflection, shttransformer_desc=shttransformer_desc, timer_instance=self.timer, SHTbaseclass=self.BaseClass)
        self.timer.add('init - deflectionlib')
        if self.nuFFTtype:
            self.epsilon = epsilon
            self.single_prec = True if epsilon > 1e-6 else False
            self.nuFFTshape = (self.nphi_dCAR, self.ntheta_dCAR)
            self.plan(epsilon=epsilon)
            self.timer.add('nuFFT and C2C - plan')
        self.timer.add_elapsed('init')
                     
    def __getattr__(self, name):
        return getattr(self.instance, name)

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

    # @timing_decorator
    def plan(self, epsilon, nuFFTtype=1):
        "nuFFT dtype real"
        
        FFT_dtype = cp.complex64 if epsilon>1e-6 else cp.complex128
        _C = cp.empty(self.nuFFTshape, dtype=FFT_dtype)
        self.FFTplan = get_fft_plan(_C, axes=(0, 1), value_type='C2C')
        cupyx.scipy.fft.fft2(_C, axes=(0, 1), norm='forward', plan=self.FFTplan)
        
        nuFFT_dtype = cp.complex64 if epsilon>1e-6 else cp.complex128
        self.nuFFTplan = Plan(nuFFTtype, self.nuFFTshape[-2:], 1, epsilon, 1, nuFFT_dtype)
    
    @debug_decorator
    # @timing_decorator
    # @shape_decorator
    def dlm2pointing(self, dlm_scaled, mmax_dlm, verbosity, nthreads):
        # TODO let's keep this double precision for now, and check later
        # pointing_theta = cp.zeros((self.deflectionlib.geom.npix()), dtype=cp.float32) if self.deflectionlib.single_prec else cp.zeros((self.deflectionlib.geom.npix()), dtype=cp.double)
        # pointing_phi = cp.zeros((self.deflectionlib.geom.npix()), dtype=cp.float32) if self.deflectionlib.single_prec else cp.zeros((self.deflectionlib.geom.npix()), dtype=cp.double)
        pointing_theta = cp.empty((self.deflectionlib.geom.npix()), dtype=cp.double)
        pointing_phi = cp.empty((self.deflectionlib.geom.npix()), dtype=cp.double)
        
        self.deflectionlib.dlm2pointing(dlm_scaled, mmax_dlm=mmax_dlm, nthreads=nthreads, verbosity=verbosity, pointing_theta=pointing_theta, pointing_phi=pointing_phi)
        return cp.array([pointing_theta, pointing_phi])       
    
    @debug_decorator
    @timing_decorator
    # @shape_decorator
    def synthesis(self, alm, out, lmax, mmax, nthreads):
        self.synthesis_cupy(alm, out, lmax=lmax, mmax=mmax, nthreads=nthreads)
        return out
    
    @debug_decorator
    @timing_decorator
    # @shape_decorator
    def adjoint_synthesis(self, synthmap, lmax, mmax, nthreads, out):
        out = self.adjoint_synthesis_cupy(synthmap, gclm=out, lmax=lmax, mmax=mmax, nthreads=nthreads)
        return out

    @debug_decorator
    @timing_decorator
    # @shape_decorator
    def C2C(self, map_in, norm='forward', fc_out=None):
        return cupyx.scipy.fft.fft2(map_in, axes=(0, 1), norm=norm, plan=self.FFTplan)
    
    @debug_decorator
    @timing_decorator
    # @shape_decorator
    def iC2C(self, fc, norm='backward', map_out=None):
        return cp.fft.fft2(fc[0], axes=(0,1), norm=norm)
    
    @debug_decorator
    @timing_decorator
    # @shape_decorator
    def nuFFT2d2(self, fc, x, y, epsilon, map_out=None):
        if self.planned:
            self.nuFFTplan.setpts(x, y, None)
            self.nuFFTplan.execute(fc, out=map_out)
        else:
            map_out = cufinufft.nufft2d2(data=fc, x=x, y=y, isign=1, eps=epsilon, out=map_out)
        return map_out
    
    @debug_decorator
    @timing_decorator
    # @shape_decorator
    def nuFFT2d1(self, pointmap, nmodes, x, y, epsilon, fc_out=None): #.reshape(self.geom.nph[0],-1).T.flatten()
        return cufinufft.nufft2d1(data=pointmap, x=x, y=y, n_modes=nmodes, isign=-1, eps=epsilon)
    
    @debug_decorator
    @timing_decorator
    # @shape_decorator
    def doubling(self, CARmap, ntheta_dCAR, nphi_dCAR, CARdmap):
        podo.Cdoubling_1D(CARmap, ntheta_dCAR, nphi_dCAR, CARdmap)
        return CARdmap   
    
    @debug_decorator
    @timing_decorator
    # @shape_decorator        
    def adjoint_doubling(self, CARdmap, ntheta_CAR, nphi_CAR, CARmap):
        podo.Cadjoint_doubling_1D(CARdmap, ntheta_CAR, nphi_CAR, CARmap)
        return CARmap

    @timing_decorator
    def synthesis_general(self, lmax, mmax, pointmap, loc, epsilon, nthreads, alm, verbosity):
        pointing_theta, pointing_phi = loc[0], loc[1]        
        self.synthesis(alm, self.CARmap, lmax=lmax, mmax=mmax, nthreads=nthreads)
        del alm
        self.doubling(self.CARmap.reshape(self.nphi_CAR,-1).T.flatten(), self.ntheta_dCAR, self.nphi_dCAR, self.CARdmap)
        del self.CARmap
        
        t0, ti = self.timer.reset()
        FFT_dtype = cp.complex64 if epsilon>1e-6 else cp.complex128
        _C = cp.ascontiguousarray(self.CARdmap.reshape(self.ntheta_dCAR,-1).T.astype(FFT_dtype))
        self.timer.add('ascontiguousarray')
        self.timer.set(t0, ti)
        t0, ti = self.timer.reset()
        fc = cupyx.scipy.fft.fft2(_C, axes=(0, 1), norm='forward', plan=self.FFTplan)
        self.timer.add('C2C')
        self.timer.set(t0, ti)
        del self.CARdmap

        t0, ti = self.timer.reset()
        nuFFT_dtype = cp.complex64 if epsilon>1e-6 else cp.complex128
        _fc = cp.ascontiguousarray(cufft.fftshift(fc, axes=(0,1)), dtype=nuFFT_dtype)#.reshape(1, *fc.shape)
        self.timer.add('ascontiguousarray')
        self.timer.set(t0, ti)
        
        del fc, _C
        
        if self.planned:
            t0, ti = self.timer.reset()
            self.nuFFTplan.setpts(pointing_phi, pointing_theta, None)
            self.timer.add('nuFFT - set points')
            pointmap = self.nuFFTplan.execute(_fc)
            self.timer.add('nuFFT - exec')
            self.timer.set(t0, ti)
        return pointmap
    
    @timing_decorator
    def adjoint_synthesis_general(self, lmax, mmax, pointmap, loc, epsilon, nthreads, alm, verbosity):
        pointing_theta, pointing_phi = loc[0], loc[1]
       
        if self.planned:
            # _ = cufinufft.nufft2d1(data=_d, x=pointing_theta, y=pointing_phi, n_modes=(2*self.ntheta_CAR-2,self.nphi_CAR), isign=-1, eps=epsilon)
            # fc = self.nuFFT2d1(_d, nmodes=(2*self.ntheta_CAR-2,self.nphi_CAR), x=_x, y=_y, epsilon=epsilon)
            _d = pointmap.astype(np.complex64)
            t0, ti = self.timer.reset()
            self.nuFFTplan.setpts(pointing_theta, pointing_phi, None)
            self.timer.add('nuFFT - set points')
            pointmap = self.nuFFTplan.execute(_d)
            self.timer.add('nuFFT - exec')
            self.timer.set(t0, ti)
        
        CARdmap = self.iC2C(cufft.fftshift(_d, axes=(1,2))).astype(np.complex128)
        
        CARmap = cp.empty(shape=(self.ntheta_CAR*self.nphi_CAR), dtype=np.float32) if self.single_prec else cp.empty(shape=(self.ntheta_CAR*self.nphi_CAR), dtype=np.double)
        synthmap = self.adjoint_doubling(CARdmap.real.flatten(), int(self.ntheta_CAR), int(self.nphi_CAR), CARmap)
        synthmap = synthmap.reshape(-1,self.nphi_CAR)
        synthmap = synthmap * self.iw[:,None] # TODO replace with shtns_no_weights-flag once it exists
        synthmap = synthmap.T.flatten()

        alm = self.adjoint_synthesis(synthmap=synthmap, lmax=lmax, mmax=mmax, nthreads=nthreads, out=alm)
        return alm
    
    @timing_decorator_close
    def gclm2lenmap(self, gclm, lmax, mmax, ptg=None, dlm_scaled=None, nthreads=None, epsilon=None, polrot=True, lenmap=None, verbosity=1, execmode='normal'):
        """
        Same as gclm2lenmap, but using cupy allocated intermediate results (synth, doubling, c2c, nuFFt),
        No h2d needed between them.
        
        gclm and dlm are assumed to be on host, will be transfered in _setup().
        Can provide pointing_theta and pointing_phi (ptg) to avoid dlm2pointing() call.
        """
        self.timer.start('gclm2lenmap()')
        if self.planned:
            assert epsilon == self.epsilon if epsilon is not None else True==True, "epsilon must be the same as in the plan"
            epsilon = self.epsilon
        else:
            assert epsilon is not None, "epsilon must be provided if not planned"
        self.deflectionlib.epsilon = epsilon
        self.single_prec = True if epsilon > 1e-6 else False
        
        
        # @timing_decorator
        def setup(self, nthreads):
            assert execmode in ['normal', 'debug', 'timing']
            print('Running in {} execution mode'.format(execmode))
            self.nthreads = self.nthreads if nthreads is None else nthreads
            self.execmode = execmode
            self.deflectionlib.execmode = self.execmode

        setup(self, nthreads)
        
        if ptg is None:
            assert dlm_scaled is not None, "Need to provide dlm_scaled if ptg is None"
            pointing_theta, pointing_phi = self.dlm2pointing(dlm_scaled, lmax, verbosity, nthreads)
        else:
            pointing_theta, pointing_phi = ptg.T
        pointing_theta = self._ensure_dtype(pointing_theta, self.single_prec, isreal=True)
        pointing_phi = self._ensure_dtype(pointing_phi, self.single_prec, isreal=True)
        pointing_dtype = cp.float32 if self.single_prec else cp.float64
        pointing_phi = cp.ascontiguousarray(pointing_phi, dtype=pointing_dtype)
        t0, ti = self.timer.reset()
        pointing_theta = cp.ascontiguousarray(pointing_theta, dtype=pointing_dtype)
        self.timer.add('ascontiguousarray')
        self.timer.set(t0, ti)
        lenmap = self.synthesis_general(lmax, mmax, alm=gclm, loc=(pointing_theta, pointing_phi), epsilon=self.epsilon, nthreads=nthreads, pointmap=lenmap, verbosity=verbosity)
        
        result = lenmap[0].get()
        # self.timer.add('Transfer <-')
        
        if self.execmode == 'debug':
            print("::debug:: Returned component results")
            return self.ret
        del lenmap, pointing_theta, pointing_phi
        return result

    @timing_decorator_close
    def lenmap2gclm(self, lenmap:cp.ndarray, dlm_scaled:cp.ndarray, gclm_out:cp.ndarray, lmax:int, mmax:int, epsilon=None, nthreads:int=None, ptg=None, verbosity=1, execmode='normal'):
        """
            Note:
                For inverse-lensing, need to feed in lensed maps times unlensed forward magnification matrix.
                lenmap must be theta contiguous
        """
        
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
            assert dlm_scaled is not None, "Need to provide dlm_scaled if ptg is None"
            pointing_theta, pointing_phi = self.dlm2pointing(dlm_scaled, lmax, verbosity, nthreads)
        else:
            pointing_theta, pointing_phi = ptg.T
        pointing_theta = self._ensure_dtype(pointing_theta, self.single_prec, isreal=True)
        pointing_phi = self._ensure_dtype(pointing_phi, self.single_prec, isreal=True)
        pointing_dtype = cp.float32 if self.single_prec else cp.float64
        pointing_phi = cp.ascontiguousarray(pointing_phi, dtype=pointing_dtype)
        t0, ti = self.timer.reset()
        pointing_theta = cp.ascontiguousarray(pointing_theta, dtype=pointing_dtype)
        self.timer.add('ascontiguousarray')
        self.timer.set(t0, ti)
        
        ptg = cp.array([pointing_theta, pointing_phi])
        gclm = self.adjoint_synthesis_general(lmax, mmax, lenmap, ptg, self.epsilon, nthreads, gclm_out, verbosity)

        if self.execmode == 'debug':
            print("::debug:: Returned component results")
            return self.ret
        del lenmap, pointing_theta, pointing_phi
        return gclm
    
    def synthesis_general_notiming(self, lmax, mmax, pointmap, loc, epsilon, nthreads, alm, verbosity):
        pointing_theta, pointing_phi = loc[0], loc[1]        
        self.synthesis(alm, self.CARmap, lmax=lmax, mmax=mmax, nthreads=nthreads)
        del alm
        
        self.doubling(self.CARmap.reshape(self.nphi_CAR,-1).T.flatten(), self.ntheta_dCAR, self.nphi_dCAR, self.CARdmap)
        del self.CARmap
        
        FFT_dtype = cp.complex64 if epsilon>1e-6 else cp.complex128
        _C = cp.ascontiguousarray(self.CARdmap.reshape(self.ntheta_dCAR,-1).T.astype(FFT_dtype))
        fc = cupyx.scipy.fft.fft2(_C, axes=(0, 1), norm='forward', plan=self.FFTplan)
        del self.CARdmap

        nuFFT_dtype = cp.complex64 if epsilon>1e-6 else cp.complex128
        _fc = cp.ascontiguousarray(cufft.fftshift(fc, axes=(0,1)), dtype=nuFFT_dtype)#.reshape(1, *fc.shape)
        del fc, _C
        
        if self.planned:
            self.nuFFTplan.setpts(pointing_phi, pointing_theta, None)
            return self.nuFFTplan.execute(_fc)
        
    def adjoint_synthesis_general_notiming(self, lmax, mmax, pointmap, loc, epsilon, nthreads, alm, verbosity):
        pointing_theta, pointing_phi = loc[0], loc[1]
       
        if self.planned:
            _d = pointmap.astype(np.complex64)
            self.nuFFTplan.setpts(pointing_theta, pointing_phi, None)
            pointmap = self.nuFFTplan.execute(_d)
        
        CARdmap = self.iC2C(cufft.fftshift(_d, axes=(1,2))).astype(np.complex128)
        
        CARmap = cp.empty(shape=(self.ntheta_CAR*self.nphi_CAR), dtype=np.float32) if self.single_prec else cp.empty(shape=(self.ntheta_CAR*self.nphi_CAR), dtype=np.double)
        synthmap = self.adjoint_doubling(CARdmap.real.flatten(), int(self.ntheta_CAR), int(self.nphi_CAR), CARmap)
        synthmap = synthmap.reshape(-1,self.nphi_CAR)
        synthmap = synthmap * self.iw[:,None] # TODO replace with shtns_no_weights-flag once it exists
        synthmap = synthmap.T.flatten()

        return self.adjoint_synthesis(synthmap=synthmap, lmax=lmax, mmax=mmax, nthreads=nthreads, out=alm)

            
    def hashdict():
        '''
        Compatibility with delensalot
        '''
        return "GPU_cufinufft_transformer"

