import numpy as np
import os, sys

from lenspyx.utils_hp import Alm, alm2cl, almxfl, alm_copy
from lenspyx.remapping.utils_angles import d2ang

import jax
jax.config.update("jax_enable_x64", True)
from jax import grad, vmap
import jax.numpy as jnp

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

import cunusht.cunusht as cunusht
import cunusht.c.podo_interface as podo
import cunusht.geometry as geometry
from cunusht.geometry import Geom
from cunusht.utils import timer as tim

from cunusht.helper import shape_decorator, debug_decorator, timing_decorator, timing_decorator_close
from cunusht.sht.GPU_sht_transformer import GPU_SHT_cunusht_transformer, GPU_SHTns_transformer
from cunusht.sht.CPU_sht_transformer import CPU_SHT_DUCC_transformer

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
        
        # assert np.multiply(*self.constructor.spat_shape) == self.geom.npix(), "{} => {} =/= {}, ({},{}) <= {}".format(self.constructor.spat_shape, np.multiply(*self.constructor.spat_shape), self.geom.npix(), self.geom.nph[0], len(self.geom.ofs), geominfo) # just a check to see if shtns geometry behaves as cunusht geometry


    def __getattr__(self, name):
        return getattr(self.instance, name)


    # @debug_decorator
    @timing_decorator
    # @shape_decorator
    def dlm2pointing(self, dlm_scaled, mmax_dlm, nthreads, verbose, pointing_theta, pointing_phi):
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
            elif verbose:
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
        self.timer.delete(self.__class__.__name__)
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
        self.ntheta_CAR = int((ducc0.fft.good_size(geominfo_deflection[1]['lmax'] + 2) + 3) // 4 * 4)
        self.nphihalf_CAR = ducc0.fft.good_size(geominfo_deflection[1]['lmax'] + 1)
        self.nphi_CAR = 2 * self.nphihalf_CAR
        self.geominfo_CAR = ('cc',{'lmax': geominfo_deflection[1]['lmax'], 'mmax':geominfo_deflection[1]['lmax'], 'ntheta':self.ntheta_CAR, 'nphi':self.nphi_CAR})
        if shttransformer_desc == 'shtns':
            self.BaseClass = type('GPU_SHTns_transformer', (GPU_SHTns_transformer,), {})
        elif shttransformer_desc == 'ducc':
            self.BaseClass = type('CPU_SHT_DUCC_transformer()', (CPU_SHT_DUCC_transformer,), {})
        elif shttransformer_desc == 'cunusht':
            assert 0, "implement if needed"
            self.BaseClass = type('GPU_SHT_cunusht_transformer', (GPU_SHT_cunusht_transformer,), {})
        else:
            raise ValueError('shttransformer_desc must be either "ducc" or "shtns" or "cunusht"')
        self.instance = self.BaseClass(geominfo=self.geominfo_CAR)
        self.timer.add('init - SHTlib')
        
        w = self.constructor.gauss_wts() * (2*np.pi)/self.constructor.nphi
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
            self.nuFFTshape = np.array([self.nphi_dCAR, self.ntheta_dCAR])
            self.plan(epsilon=epsilon, nuFFTtype=self.nuFFTtype)
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

        # FFT_dtype = cp.float32 if epsilon>1e-6 else cp.float64
        FFT_dtype = cp.float64
        _C = cp.empty(self.nuFFTshape, dtype=FFT_dtype)
        _C = cp.ascontiguousarray(_C.T) if self.nuFFTtype == 1 else cp.ascontiguousarray(_C)
        print("shape plan: {}".format(self.nuFFTshape))
        self.FFTplan = get_fft_plan(_C, axes=(0, 1), value_type='C2C')
        cupyx.scipy.fft.fft2(_C, axes=(0, 1), norm='forward', plan=self.FFTplan)

        nuFFT_dtype = cp.float32 if epsilon>1e-6 else cp.float64
        # nuFFT_dtype = cp.float64
        isign = -1 if self.nuFFTtype == 1 else 1
        self.nuFFTplan = Plan(nuFFTtype, tuple(self.nuFFTshape[-2:][::-1]) if self.nuFFTtype == 1 else tuple(self.nuFFTshape[-2:]), 1, epsilon, isign, nuFFT_dtype, gpu_method=2, gpu_sort=1)#, gpu_kerevalmeth=0)#, upsampfac=1.5)
        # print("waiting after nufftplan - check memory now")
        # time.sleep(5)
            
    @debug_decorator
    # @timing_decorator
    # @shape_decorator
    def dlm2pointing(self, dlm_scaled, mmax_dlm, verbose, nthreads, single_prec=False):
        # TODO let's keep this double precision for now, and check later
        # pointing_theta = cp.zeros((self.deflectionlib.geom.npix()), dtype=cp.float32) if self.deflectionlib.single_prec else cp.zeros((self.deflectionlib.geom.npix()), dtype=cp.double)
        # pointing_phi = cp.zeros((self.deflectionlib.geom.npix()), dtype=cp.float32) if self.deflectionlib.single_prec else cp.zeros((self.deflectionlib.geom.npix()), dtype=cp.double)
        pointing_theta = cp.empty((self.deflectionlib.geom.npix()), dtype=cp.double)
        pointing_phi = cp.empty((self.deflectionlib.geom.npix()), dtype=cp.double)
        
        self.deflectionlib.dlm2pointing(dlm_scaled, mmax_dlm=mmax_dlm, nthreads=nthreads, verbose=verbose, pointing_theta=pointing_theta, pointing_phi=pointing_phi)
        return cp.array([pointing_theta, pointing_phi])       
    
    @debug_decorator
    @timing_decorator
    # @shape_decorator
    def synthesis(self, alm, out, lmax, mmax, nthreads):
        self.synthesis_cupy(alm, out, lmax=lmax, mmax=mmax, nthreads=nthreads)
        return out # np.array(out.get())
    
    @debug_decorator
    @timing_decorator
    # @shape_decorator
    def synthesis_(self, alm, lmax, mmax, nthreads):
        return self.synthesis_jnp(alm, lmax=lmax, mmax=mmax, nthreads=nthreads)

    def grad_synthesis(self, alm, out, lmax, mmax, nthreads):
        return grad(self.synthesis_)(alm, out, lmax, mmax, nthreads)
    
    @debug_decorator
    @timing_decorator
    # @shape_decorator
    def adjoint_synthesis(self, synthmap, lmax, mmax, nthreads, out, spin=0):
        return self.adjoint_synthesis_cupy(synthmap, gclm=out, lmax=lmax, mmax=mmax, nthreads=nthreads)

    @debug_decorator
    @timing_decorator
    # @shape_decorator
    def C2C(self, map_in, norm='forward', fc_out=None):
        return cupyx.scipy.fft.fft2(map_in, axes=(0, 1), norm=norm, plan=self.FFTplan)
    
    @debug_decorator
    @timing_decorator
    # @shape_decorator
    def iC2C(self, fc, norm='backward', map_out=None):
        return cupyx.scipy.fft.ifft2(fc, axes=(0, 1), norm=norm, plan=self.FFTplan)
    
    @debug_decorator
    @timing_decorator
    # @shape_decorator
    def nuFFT2d2(self, fc, x, y, epsilon, map_out=None):
        if self.nuFFTtype:
            self.nuFFTplan.setpts(x, y, None)
            return self.nuFFTplan.execute(fc) #out=map_out
        else:
            return cufinufft.nufft2d2(data=fc, x=x, y=y, isign=1, eps=epsilon) #out=map_out
    
    @debug_decorator
    @timing_decorator
    # @shape_decorator
    def nuFFT2d1(self, pointmap, nmodes, x, y, epsilon, fc_out=None):
        if self.nuFFTtype:
            self.nuFFTplan.setpts(x, y, None)
            return self.nuFFTplan.execute(pointmap)
        else:  
            return cufinufft.nufft2d1(data=pointmap, x=x, y=y, n_modes=nmodes, isign=-1, eps=epsilon)
    
    @debug_decorator
    @timing_decorator
    # @shape_decorator
    def doubling(self, CARmap, ntheta_dCAR, nphi_dCAR, CARdmap):
        # podo.Cdoubling_1D(CARmap, ntheta_dCAR, nphi_dCAR, CARdmap)
        podo.Cdoubling_contig_1D(CARmap, self.ntheta_CAR, nphi_dCAR, CARdmap)
        return CARdmap   
    
    @debug_decorator
    @timing_decorator
    # @shape_decorator        
    def adjoint_doubling(self, CARdmap, ntheta_CAR, nphi_CAR, CARmap):
        podo.Cadjoint_doubling_1D(CARdmap, ntheta_CAR, nphi_CAR, CARmap)
        return CARmap

    @timing_decorator
    def synthesis_general(self, lmax, mmax, alm, loc, epsilon, nthreads, pointmap, verbose):
        """expects nuFFT type 2
        """
        # FFT_dtype = cp.float32 if epsilon>1e-6 else cp.float64
        FFT_dtype = cp.float64
        # FFT_dtype = cp.complex64 if epsilon>1e-6 else cp.complex128
        nuFFT_dtype = cp.complex64 if epsilon>1e-6 else cp.complex128
        pointmap = pointmap.astype(cp.float64)
        # pointmap = pointmap.astype(cp.float32 if epsilon>1e-6 else cp.float64) 
        
        pointing_theta, pointing_phi = loc[0], loc[1]
        # res = self.grad_synthesis(alm, self.CARmap, lmax=lmax, mmax=mmax, nthreads=nthreads)
        # print(f"grad synthesis: {res}")
        self.synthesis(alm, self.CARmap, lmax=lmax, mmax=mmax, nthreads=nthreads)
        del alm
        
        self.doubling(self.CARmap, self.ntheta_dCAR, self.nphi_dCAR, self.CARdmap)
        del self.CARmap
        
        _C = self.CARdmap.reshape(self.nphi_dCAR,-1).astype(FFT_dtype)
        cp.cuda.runtime.deviceSynchronize()
        fc = self.C2C(_C).astype(nuFFT_dtype)
        
        self.nuFFT2d2(cufft.fftshift(fc, axes=(0,1)), pointing_phi, pointing_theta, epsilon, pointmap)

        return pointmap
    
    @timing_decorator
    # @debug_decorator
    def adjoint_synthesis_general(self, lmax, mmax, pointmap, loc, epsilon, nthreads, alm, verbose):
        """expects nuFFT type 1
        """
        FFT_dtype = cp.float32 if epsilon>1e-6 else cp.float64
        nuFFT_dtype = cp.complex64 if epsilon>1e-6 else cp.complex128
        
        pointing_theta, pointing_phi = loc[0], loc[1]
        
        fc = self.nuFFT2d1(pointmap, nmodes=(self.nphi_dCAR, self.ntheta_dCAR), x=pointing_theta, y=pointing_phi, epsilon=epsilon)
        CARdmap = self.iC2C(cufft.fftshift(fc, axes=(0,1))).astype(np.complex128)
        
        CARmap = cp.empty(shape=(self.ntheta_CAR*self.nphi_CAR), dtype=np.float32) if self.single_prec else cp.empty(shape=(self.ntheta_CAR*self.nphi_CAR), dtype=np.double)
        synthmap = self.adjoint_doubling(CARdmap.real.flatten(), int(self.ntheta_CAR), int(self.nphi_CAR), CARmap)
        synthmap = (synthmap.reshape(-1,self.nphi_CAR) * self.iw[:,None]).T.flatten()

        alm = self.adjoint_synthesis(synthmap=synthmap, lmax=lmax, mmax=mmax, nthreads=nthreads, out=alm)
        return alm
    
    @timing_decorator_close
    def gclm2lenmap(self, gclm, lmax, mmax, ptg=None, dlm_scaled=None, nthreads=None, epsilon=None, polrot=True, lenmap=None, verbose=1, execmode='normal'):
        """
        expects nuFFT type 2
        Same as gclm2lenmap, but using cupy allocated intermediate results (synth, doubling, c2c, nuFFt),
        No h2d needed between them.
        
        gclm and dlm are assumed to be on host, will be transfered in _setup().
        Can provide pointing_theta and pointing_phi (ptg) to avoid dlm2pointing() call.
        """
        self.timer.delete('gclm2lenmap()')
        self.timer.start('gclm2lenmap()')
        if self.nuFFTtype:
            assert epsilon == self.epsilon if epsilon is not None else True==True, "epsilon must be the same as in the plan"
            epsilon = self.epsilon
        else:
            assert epsilon is not None, "epsilon must be provided if not planned"
            self.epsilon = epsilon
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
            pointing_theta, pointing_phi = self.dlm2pointing(dlm_scaled, lmax, verbose, nthreads)
            del dlm_scaled
        else:
            pointing_theta, pointing_phi = ptg.T
        pointing_dtype = cp.float32 if self.single_prec else cp.float64
        # pointing_theta = cp.ascontiguousarray(self._ensure_dtype(pointing_theta, self.single_prec, isreal=True), dtype=pointing_dtype)
        # pointing_phi = cp.ascontiguousarray(self._ensure_dtype(pointing_phi, self.single_prec, isreal=True), dtype=pointing_dtype)
        pointing_theta = self._ensure_dtype(pointing_theta, self.single_prec, isreal=True).astype(pointing_dtype)
        pointing_phi = self._ensure_dtype(pointing_phi, self.single_prec, isreal=True).astype(pointing_dtype)
       
        lenmap = self.synthesis_general(lmax, mmax, alm=gclm, loc=(pointing_theta, pointing_phi), epsilon=self.epsilon, nthreads=nthreads, pointmap=lenmap, verbose=verbose)
        
        result = lenmap[0].get()
        # self.timer.add('Transfer <-')
        
        if self.execmode == 'debug':
            print("::debug:: Returned component results")
            self.timer.delete('gclm2lenmap()')
            return self.ret
        del lenmap, pointing_theta, pointing_phi
        self.timer.delete('gclm2lenmap()')
        return result

    @timing_decorator_close
    def lenmap2gclm(self, lenmap:cp.ndarray, dlm_scaled:cp.ndarray, gclm_out:cp.ndarray, lmax:int, mmax:int, epsilon=None, nthreads:int=None, ptg=None, verbose=1, execmode='normal'):
        """
            Note:
                expects nuFFT type 1
                For inverse-lensing, need to feed in lensed maps times unlensed forward magnification matrix.
                lenmap must be theta contiguous
        """
        self.timer.delete('lenmap2gclm()')
        self.timer.start("lenmap2gclm()")
        self.single_prec = True if epsilon > 1e-6 else False
        self._assert_shape(lenmap, gclm_out, dlm_scaled, ndim=2, nbatch=1)
        self._assert_type(lenmap, gclm_out, dlm_scaled)
        self._assert_dtype(lenmap, gclm_out, dlm_scaled)
        self._assert_precision(lenmap, gclm_out)
        
        def setup(self, nthreads):
            assert execmode in ['normal', 'debug', 'timing']
            print('Running in {} execution mode'.format(execmode))
            self.nthreads = self.nthreads if nthreads is None else nthreads
            self.execmode = execmode
            self.deflectionlib.execmode = self.execmode
  
        setup(self, nthreads)
        
        if ptg is None:
            assert dlm_scaled is not None, "Need to provide dlm_scaled if ptg is None"
            pointing_theta, pointing_phi = self.dlm2pointing(dlm_scaled, lmax, verbose, nthreads)
            del dlm_scaled
        else:
            pointing_theta, pointing_phi = ptg.T
        pointing_dtype = cp.float32 if self.single_prec else cp.float64
        pointing_theta = self._ensure_dtype(pointing_theta, self.single_prec, isreal=True).astype(pointing_dtype)
        pointing_phi = self._ensure_dtype(pointing_phi, self.single_prec, isreal=True).astype(pointing_dtype)
        ptg = cp.array([pointing_theta, pointing_phi])
        
        gclm = self.adjoint_synthesis_general(lmax, mmax, lenmap[0], ptg, self.epsilon, nthreads, gclm_out, verbose)

        if self.execmode == 'debug':
            print("::debug:: Returned component results")
            return self.ret
        del lenmap, pointing_theta, pointing_phi
        self.timer.delete("lenmap2gclm()")
        return gclm
    
    @debug_decorator
    @timing_decorator
    # @shape_decorator
    def nuFFT2d2_jnp(self, fc, x, y, epsilon, map_out=None):
        x = cp.array(np.array(x))
        y = cp.array(np.array(y))
        if self.nuFFTtype:
            self.nuFFTplan.setpts(x, y, None)
            return jnp.array(self.nuFFTplan.execute(fc).get()) #out=map_out
        else:
            return jnp.array(cufinufft.nufft2d2(data=fc, x=x, y=y, isign=1, eps=epsilon).get())  #out=map_out

    def synthesis_general_jnp(self, lmax, mmax, alm, loc0, loc1, epsilon):
        """expects nuFFT type 2
        """
        nthreads = 10
        print(lmax, mmax, epsilon)
        FFT_dtype = cp.float32 if epsilon>1e-6 else cp.float64
        # FFT_dtype = cp.complex64 if epsilon>1e-6 else cp.complex128
        nuFFT_dtype = cp.complex64 if epsilon>1e-6 else cp.complex128
        
        pointing_theta, pointing_phi = loc0, loc1
        self.CARmap = self.synthesis_(alm, lmax=lmax, mmax=mmax, nthreads=nthreads)
        del alm
        
        self.doubling(cp.array(self.CARmap), self.ntheta_dCAR, self.nphi_dCAR, self.CARdmap)
        del self.CARmap
        
        _C = self.CARdmap.reshape(self.nphi_dCAR,-1).astype(FFT_dtype)
        fc = self.C2C(_C).astype(nuFFT_dtype)
        
        return self.nuFFT2d2_jnp(cufft.fftshift(fc, axes=(0,1)), pointing_phi, pointing_theta, epsilon)
    
    @timing_decorator_close
    def gclm2lenmap_grad(self, gclm, lmax, mmax, ptg=None, dlm_scaled=None, nthreads=None, epsilon=None, polrot=True, lenmap=None, verbose=1, execmode='normal'):
        """
        expects nuFFT type 2
        Same as gclm2lenmap, but using cupy allocated intermediate results (synth, doubling, c2c, nuFFt),
        No h2d needed between them.
        
        gclm and dlm are assumed to be on host, will be transfered in _setup().
        Can provide pointing_theta and pointing_phi (ptg) to avoid dlm2pointing() call.
        """
        self.timer.start('gclm2lenmap()')
        if self.nuFFTtype:
            assert epsilon == self.epsilon if epsilon is not None else True==True, "epsilon must be the same as in the plan"
            epsilon = self.epsilon
        else:
            assert epsilon is not None, "epsilon must be provided if not planned"
            self.epsilon = epsilon
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
            pointing_theta, pointing_phi = self.dlm2pointing(dlm_scaled, lmax, verbose, nthreads)
            del dlm_scaled
        else:
            pointing_theta, pointing_phi = ptg.T
        pointing_dtype = cp.float32 if self.single_prec else cp.float64
        # pointing_theta = cp.ascontiguousarray(self._ensure_dtype(pointing_theta, self.single_prec, isreal=True), dtype=pointing_dtype)
        # pointing_phi = cp.ascontiguousarray(self._ensure_dtype(pointing_phi, self.single_prec, isreal=True), dtype=pointing_dtype)
        pointing_theta = self._ensure_dtype(pointing_theta, self.single_prec, isreal=True).astype(pointing_dtype)
        pointing_phi = self._ensure_dtype(pointing_phi, self.single_prec, isreal=True).astype(pointing_dtype)
        lenmap_grad = grad(self.synthesis_general_jnp, argnums=(2))
        print(gclm.shape, pointing_theta.shape)
        lenmap_grad = vmap(lenmap_grad, in_axes=(None, None, 0, None, None, None))(float(lmax), float(mmax), jnp.array(np.atleast_2d(gclm.get())), jnp.array(pointing_theta.get()), jnp.array(pointing_phi.get()), float(self.epsilon))
        
        result = lenmap_grad[0].get()
        # self.timer.add('Transfer <-')
        
        if self.execmode == 'debug':
            print("::debug:: Returned component results")
            return self.ret
        del lenmap_grad, pointing_theta, pointing_phi
        return result
        

    def synthesis_general_notiming(self, lmax, mmax, pointmap, loc, epsilon, nthreads, alm, verbose):
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
        
    def adjoint_synthesis_general_notiming(self, lmax, mmax, pointmap, loc, epsilon, nthreads, alm, verbose):
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

