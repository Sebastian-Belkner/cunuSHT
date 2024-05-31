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

import cunusht.c.podo_interface as podo
import cunusht.geometry as geometry
from cunusht.utils import timer as tim

from cunusht.helper_GPU import shape_decorator, debug_decorator, timing_decorator, timing_decorator_close
from cunusht.sht.GPU_sht_transformer import GPU_SHT_cunusht_transformer, GPU_SHTns_transformer
from cunusht.sht.CPU_sht_transformer import CPU_SHT_DUCC_transformer

ctype = {True: np.complex64, False: np.complex128}
rtype = {True: np.float32, False: np.float64}

dtype_r2c = {np.dtype(np.float32): np.complex64, np.dtype(np.float64): np.complex128,
             np.float32: np.complex64, np.float64: np.complex128}
dtype_c2r = {np.dtype(np.complex64): np.float32, np.dtype(np.complex128): np.float64,
             np.complex64: np.float32, np.complex128: np.float64}

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
    def dlm2pointing(self, dlm_scaled, mmax_dlm, verbose, pointing_theta, pointing_phi):
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
            out_theta, out_phi = self.synthesis_der1_cupy(dlm_scaled, out_theta, out_phi)
            return cp.array([out_theta, out_phi])
          
        # @debug_decorator
        @timing_decorator
        def _pointing(self, spin1_theta, spin1_phi, cpt, cpphi0, cpnph, cpofs, pointing_theta, pointing_phi):
            return podo.Cpointing_1Dto1D_lowmem(cpt, cpphi0, cpnph, cpofs, spin1_theta, spin1_phi, pointing_theta, pointing_phi)
        
        dlm_scaled = cp.array(dlm_scaled, dtype=complex)
        cpt = cp.array(self.geom.theta.astype(np.double), dtype=cp.double)
        cpphi0 = cp.array(self.geom.phi0, dtype=cp.double)
        cpnph = cp.array(self.geom.nph, dtype=cp.uint64)
        cpofs = cp.array(self.geom.ofs, dtype=cp.uint64)
        spin1_theta = cp.zeros(self.constructor.spat_shape, dtype=cp.double)
        spin1_phi = cp.zeros(self.constructor.spat_shape, dtype=cp.double)

        spin1_theta, spin1_phi = _spin__1___synth(self, dlm_scaled, spin1_theta, spin1_phi)
        _pointing(self, spin1_theta, spin1_phi, cpt, cpphi0, cpnph, cpofs, pointing_theta, pointing_phi)
        
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
        # FIXME hard-coded flag - remove later.. or perhaps never
        self.nuFFT_single_prec = False
        
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
        # FIXME keep nuFFT double precision for any epsilon for now.
        "nuFFT dtype real"

        self.FFT_dtype = cp.complex64 if epsilon>1e-6 else cp.complex128
        # self.FFT_dtype = cp.float64
        _C = cp.empty(self.nuFFTshape, dtype=self.FFT_dtype)
        _C = cp.ascontiguousarray(_C.T) if self.nuFFTtype == 1 else cp.ascontiguousarray(_C)
        self.FFTplan = get_fft_plan(_C, axes=(0, 1), value_type='C2C')
        cupyx.scipy.fft.fft2(_C, axes=(0, 1), norm='forward', plan=self.FFTplan)

        self.nuFFT_dtype = cp.float32 if self.nuFFT_single_prec else cp.float64
        isign = -1 if self.nuFFTtype == 1 else 1
        # print("waiting before nufftplan - check memory now")
        # time.sleep(5)
        self.nuFFTplan = Plan(
            nuFFTtype,
            tuple(self.nuFFTshape[-2:][::-1]) if self.nuFFTtype == 1 else tuple(self.nuFFTshape[-2:]),
            1, epsilon, isign, self.nuFFT_dtype, gpu_method=2,
            gpu_sort=1, gpu_kerevalmeth=0, upsampfac=1.25)#, modeord=1)
        # print("waiting after nufftplan - check memory now")
        # time.sleep(5)     
    
    @debug_decorator
    @timing_decorator
    # @shape_decorator
    def _synthesis(self, alm, out, lmax, mmax):
        # This is CAR grid, as init sets up SHT transformer with CAR geometry
        assert alm.dtype in [np.complex128, cp.complex128], "alm should be double precision for accurate SHT, but is {}".format(alm.dtype) 
        return self.synthesis_cupy(alm, out, lmax=lmax, mmax=mmax)
    
    @debug_decorator
    @timing_decorator
    # @shape_decorator
    def _adjoint_synthesis(self, synthmap, lmax, mmax, out, spin=0):
        # This is CAR grid, as init sets up SHT transformer with CAR geometry
        assert synthmap.dtype in [np.float64, cp.float64], "synthmap should be double precision for accurate SHT, but is {}".format(synthmap.dtype) 
        
        return self.adjoint_synthesis_cupy(synthmap, gclm=out, lmax=lmax, mmax=mmax)

    @debug_decorator
    @timing_decorator
    # @shape_decorator
    def _C2C(self, map_in, norm='forward', fc_out=None):
        
        assert self.FFT_dtype == map_in.dtype, "map dtype should be same as in the FFT plan ({}), but is {}".format(self.FFT_dtype, map_in.dtype)
        
        return cupyx.scipy.fft.fft2(map_in, axes=(0, 1), norm=norm, plan=self.FFTplan)
    
    @debug_decorator
    @timing_decorator
    # @shape_decorator
    def _iC2C(self, fc, norm='backward', map_out=None):
        
        assert self.FFT_dtype == fc.dtype, "fc dtype should be same as in the FFT plan ({}), but is {}".format(self.FFT_dtype, fc.dtype)
        
        return cupyx.scipy.fft.ifft2(fc, axes=(0, 1), norm=norm, plan=self.FFTplan)
    
    @debug_decorator
    @timing_decorator
    # @shape_decorator
    def _nuFFT2d2(self, fc, x, y, epsilon, map_out=None):
        assert x.dtype == y.dtype
        assert dtype_c2r[fc.dtype] == x.dtype, 'fourier coefficients precision should match pointing precision ({}), but is {}'.format(x.dtype, dtype_c2r[fc.dtype])
        if self.nuFFTtype:
            # self.nuFFTplan.setpts(x, y, None)
            return self.nuFFTplan.execute(fc) #out=map_out
        else:
            return cufinufft.nufft2d2(data=fc, x=x, y=y, isign=1, eps=epsilon) #out=map_out
    
    @debug_decorator
    @timing_decorator
    # @shape_decorator
    def _nuFFT2d1(self, pointmap, nmodes, x, y, epsilon, fc_out=None):
        assert x.dtype == y.dtype
        
        # FIXME why does pointmap have to be complex, when nuFFTplan dtype is set to float?
        assert dtype_c2r[pointmap.dtype] == x.dtype, 'map precision should match pointing precision ({}), but is {}'.format(x.dtype, pointmap.dtype)
        
        if self.nuFFTtype:
            self.nuFFTplan.setpts(x, y, None)
            return self.nuFFTplan.execute(pointmap)
        else:  
            return cufinufft.nufft2d1(data=pointmap, x=x, y=y, n_modes=nmodes, isign=-1, eps=epsilon)
    
    @debug_decorator
    @timing_decorator
    # @shape_decorator
    def _doubling(self, CARmap, ntheta_dCAR, nphi_dCAR, CARdmap):
        print("shapes _doubling: ", CARmap.shape, ntheta_dCAR, nphi_dCAR, CARdmap.shape)
        print(CARmap)
        print('dtypes: ', CARmap.dtype, CARdmap.dtype)
        podo.Cdoubling_contig_1D(CARmap, self.ntheta_CAR, nphi_dCAR, CARdmap)
        return CARdmap   
    
    @debug_decorator
    @timing_decorator
    # @shape_decorator        
    def _adjoint_doubling(self, CARdmap, ntheta_CAR, nphi_CAR, CARmap):
        podo.Cadjoint_doubling_1D(CARdmap, ntheta_CAR, nphi_CAR, CARmap)
        return CARmap
    
    def get_grid_uniform(self):
        """ Convenience function for getting phi and theta positions on the uniform grid. 
                
                Note: this likely only works for Gauss-Legendre at the moment
        """
        lmax = self.deflectionlib.geominfo[1]['lmax']
        pp = cp.ones((self.npix()), dtype=cp.double)
        pt = cp.ones((self.npix()), dtype=cp.double)
        _t = (pt.reshape(lmax+1,-1) * cp.array(self.deflectionlib.geom.theta)[:,cp.newaxis]).flatten()
        _p = ((pp.reshape(lmax+1,-1)) * cp.linspace(0,2*np.pi,int(self.deflectionlib.geom.nph[0]), endpoint=False)).flatten()
        return cp.array([_t, _p])
    
    def npix(self):
        return self.deflectionlib.geom.npix()
    
    @debug_decorator
    @timing_decorator
    # @shape_decorator
    def dlm2pointing(self, dlm_scaled, mmax_dlm, verbose, single_prec=False):
        # NOTE single_prec parameter no needed for CMB lensing applications
        # NOTE let's keep this double precision for now, and check later
        pointing_theta = cp.empty((self.deflectionlib.geom.npix()), dtype=cp.double)
        pointing_phi = cp.empty((self.deflectionlib.geom.npix()), dtype=cp.double)
        
        self.deflectionlib.dlm2pointing(dlm_scaled, mmax_dlm=mmax_dlm, verbose=verbose, pointing_theta=pointing_theta, pointing_phi=pointing_phi)
        return cp.array([pointing_theta, pointing_phi])  
    
    @debug_decorator
    @timing_decorator
    # @shape_decorator
    def synthesis(self, alm, out, lmax, mmax):
        """ This is outward facing function, and uses the pointing grid, as user might expect
        """
        if not isinstance(alm, cp.ndarray):
            alm = cp.array(alm)
            print("WARNING: alm not cupy and has automatically been transferred to device. This may reduce performance. Consider passing a cupy array instead")
        if out is None:
            out = cp.empty(self.npix())
        assert lmax == self.deflectionlib.geominfo[1]['lmax'], "Expected lmax={}. lmax must be same as during init, but is {}".format(lmax, self.deflectionlib.geominfo[1]['lmax'])
        assert mmax == self.deflectionlib.geominfo[1]['lmax'], "Expected mmax={}. mmax must be same as lmax during init, but is {}".format(mmax, self.deflectionlib.geominfo[1]['lmax'])
        
        return self.deflectionlib.synthesis_cupy(alm, out, lmax=lmax, mmax=mmax)
    
    @debug_decorator
    @timing_decorator
    # @shape_decorator
    def adjoint_synthesis(self, synthmap, lmax, mmax, out, spin=0):
        """ This is outward facing function, and uses the pointing grid, as user might expect
        """
        if not isinstance(synthmap, cp.ndarray):
            synthmap = cp.array(synthmap)
            print("WARNING: synthmap not cupy and has automatically been transferred to device. This may reduce performance. Consider passing a cupy array instead")
        if out is None:
            out = cp.empty(self.npix())
        assert lmax == self.deflectionlib.geominfo[1]['lmax'], "Expected lmax={}. lmax must be same as during init, but is {}".format(lmax, self.deflectionlib.geominfo[1]['lmax'])
        assert mmax == self.deflectionlib.geominfo[1]['lmax'], "Expected mmax={}. mmax must be same as lmax during init, but is {}".format(mmax, self.deflectionlib.geominfo[1]['lmax'])
        
        return self.deflectionlib.adjoint_synthesis_cupy(synthmap, gclm=out, lmax=lmax, mmax=mmax)

    @timing_decorator
    def synthesis_general(self, lmax, mmax, alm, loc, pointmap, epsilon=None, verbose=0):
        """ This is outward facing function, and expects nuFFT type 2
        """
        assert lmax == self.deflectionlib.geominfo[1]['lmax'], "Expected lmax={}. lmax must be same as during init, but is {}".format(lmax, self.deflectionlib.geominfo[1]['lmax'])
        assert mmax == self.deflectionlib.geominfo[1]['lmax'], "Expected mmax={}. mmax must be same as lmax during init, but is {}".format(mmax, self.deflectionlib.geominfo[1]['lmax'])
        
        if not isinstance(alm, cp.ndarray):
            alm = cp.array(alm)
            print("WARNING: alm not cupy and has automatically been transferred to device. This may reduce performance. Consider passing a cupy array instead")
        if not isinstance(pointmap, cp.ndarray):
            pointmap = cp.array(pointmap)
            print("WARNING: synthmap not cupy and has automatically been transferred to device. This may reduce performance. Consider passing a cupy array instead")
        if not isinstance(loc, cp.ndarray):
            loc = cp.array(loc)
            print("WARNING: loc not cupy and has automatically been transferred to device. This may reduce performance. Consider passing a cupy array instead")
        
        if self.nuFFTtype:
            # NOTE epsilon parameter only useful if nuFFT not planned
            assert epsilon == self.epsilon if epsilon is not None else True==True, "expected epsilon = {} as set during initialisation of nuFFT plan, but is {}".format(self.epsilon, epsilon)
            epsilon = self.epsilon
        else:
            assert epsilon is not None, "epsilon must be provided if not planned"
            self.epsilon = epsilon
        self.deflectionlib.epsilon = epsilon
        
        # TODO dtype checks - compare to plan if applicable
        
        # TODO shape checks

        pointmap = pointmap #FIXME need dtype checks here - possibly ensures
        # pointmap = pointmap.astype(cp.float32 if epsilon>1e-6 else cp.float64) 
        pointing_theta, pointing_phi = loc.T[0], loc.T[1] # transposing so that we follow DUCC convention
        
        self.CARmap = self._synthesis(alm, self.CARmap, lmax=lmax, mmax=mmax).flatten()
        del alm
        
        #TODO 2d-doubling would be nice-to-have at some point
        self._doubling(self.CARmap, self.ntheta_dCAR, self.nphi_dCAR, self.CARdmap)
        # del self.CARmap
        _C = self.CARdmap.reshape(self.nphi_dCAR,-1).astype(self.FFT_dtype)

        fc = self._C2C(_C).astype(dtype_r2c[self.nuFFT_dtype]) # sp, dp didn't seem to have impact on acc
        # del _C
        pointmap = self._nuFFT2d2(cufft.fftshift(fc, axes=(0,1)), pointing_phi, pointing_theta, epsilon, pointmap)
        # self.nuFFT2d2(fc, pointing_phi, pointing_theta, epsilon, pointmap)

        return pointmap.real
    
    @timing_decorator
    # @debug_decorator
    def adjoint_synthesis_general(self, lmax, mmax, pointmap, loc, epsilon, alm, verbose):
        """ This is outward facing function, and expects nuFFT type 1

            Note: pointmap must be theta contiguous, otherwise result may be wrong.
        """
        assert lmax == self.deflectionlib.geominfo[1]['lmax'], "Expected lmax={}. lmax must be same as during init, but is {}".format(lmax, self.deflectionlib.geominfo[1]['lmax'])
        assert mmax == self.deflectionlib.geominfo[1]['lmax'], "Expected mmax={}. mmax must be same as lmax during init, but is {}".format(mmax, self.deflectionlib.geominfo[1]['lmax'])
        
        if not isinstance(alm, cp.ndarray):
            alm = cp.array(alm)
            print("WARNING: alm not cupy and has automatically been transferred to device. This may reduce performance. Consider passing a cupy array instead")
        if not isinstance(pointmap, cp.ndarray):
            pointmap = cp.array(pointmap)
            print("WARNING: synthmap not cupy and has automatically been transferred to device. This may reduce performance. Consider passing a cupy array instead")
        if not isinstance(loc, cp.ndarray):
            loc = cp.array(loc)
            print("WARNING: loc not cupy and has automatically been transferred to device. This may reduce performance. Consider passing a cupy array instead")
        
        if self.nuFFTtype:
            # NOTE epsilon parameter only useful if nuFFT not planned
            assert epsilon == self.epsilon if epsilon is not None else True==True, "expected epsilon = {} as set during initialisation of nuFFT plan, but is {}".format(self.epsilon, epsilon)
        else:
            assert epsilon is not None, "epsilon must be provided if not planned"
            self.epsilon = epsilon
        self.deflectionlib.epsilon = epsilon
        
        # TODO dtype checks - compare to plan if applicable
        
        # TODO shape checks
        
        FFT_dtype = cp.float32 if epsilon>1e-6 else cp.float64
        nuFFT_dtype = cp.complex64 if epsilon>1e-6 else cp.complex128
        
        pointing_theta, pointing_phi = loc.T[0], loc.T[1]
        
        fc = self._nuFFT2d1(pointmap, nmodes=(self.nphi_dCAR, self.ntheta_dCAR), x=pointing_theta, y=pointing_phi, epsilon=epsilon)
        CARdmap = self._iC2C(cufft.fftshift(fc, axes=(0,1))).astype(np.complex128)
        
        CARmap = cp.empty(shape=(self.ntheta_CAR*self.nphi_CAR), dtype=np.float32) if self.single_prec else cp.empty(shape=(self.ntheta_CAR*self.nphi_CAR), dtype=np.double)
        synthmap = self._adjoint_doubling(CARdmap.real.flatten(), int(self.ntheta_CAR), int(self.nphi_CAR), CARmap)
        synthmap = (synthmap.reshape(-1,self.nphi_CAR) * self.iw[:,None]).T.flatten()

        alm = self._adjoint_synthesis(synthmap=synthmap, lmax=lmax, mmax=mmax, out=alm)
        return alm
    
    @timing_decorator_close
    def gclm2lenmap(self, gclm, lmax, mmax, ptg=None, dlm_scaled=None, epsilon=None, lenmap=None, verbose=1, execmode='normal', runid=None):
        """  This is outward facing function, expects nuFFT type 2
    
                Note: Can provide pointing_theta and pointing_phi (ptg) to avoid dlm2pointing() call, in which case this in principle becomes a pure synthesis_general() call
        """
        self.runid = runid
        self.timer.delete('gclm2lenmap()')
        self.timer.start('gclm2lenmap()')
        
        assert lmax == self.deflectionlib.geominfo[1]['lmax'], "Expected lmax={}. lmax must be same as during init, but is {}".format(lmax, self.deflectionlib.geominfo[1]['lmax'])
        assert mmax == self.deflectionlib.geominfo[1]['lmax'], "Expected mmax={}. mmax must be same as lmax during init, but is {}".format(mmax, self.deflectionlib.geominfo[1]['lmax'])
        
        t0, ti = self.timer.reset()
        cp.cuda.runtime.deviceSynchronize()
        if not isinstance(gclm, cp.ndarray):
            gclm = cp.array(gclm)
            # print("WARNING: gclm not cupy and has automatically been transferred to device. This may reduce performance. Consider passing a cupy array instead")
        if not isinstance(lenmap, cp.ndarray):
            lenmap = cp.array(lenmap)
            # print("WARNING: lenmap not cupy and has automatically been transferred to device. This may reduce performance. Consider passing a cupy array instead")
        if not isinstance(ptg, cp.ndarray) if ptg is not None else False:
            ptg = cp.array(ptg)
            # print("WARNING: ptg not cupy and has automatically been transferred to device. This may reduce performance. Consider passing a cupy array instead")
        if not isinstance(dlm_scaled, cp.ndarray):
            dlm_scaled = cp.array(dlm_scaled)
            # print("WARNING: dlm_scaled not cupy and has automatically been transferred to device. This may reduce performance. Consider passing a cupy array instead")
        cp.cuda.runtime.deviceSynchronize()
        self.timer.add("Transfer ->")
        self.timer.set(t0, ti)
        
        if self.nuFFTtype:
            # NOTE epsilon parameter only useful if nuFFT not planned
            assert epsilon == self.epsilon if epsilon is not None else True==True, "expected epsilon = {} as set during initialisation of nuFFT plan, but is {}".format(self.epsilon, epsilon)
            epsilon = self.epsilon
        else:
            assert epsilon is not None, "epsilon must be provided if not planned"
            self.epsilon = epsilon
        self.deflectionlib.epsilon = epsilon
        
        # TODO dtype checks - compare to plan if applicable
        
        # TODO shape checks
        
        self.single_prec = True if epsilon > 1e-6 else False
        
        
        # @timing_decorator
        def setup(self):
            assert execmode in ['normal', 'debug', 'timing']
            print('Running in {} execution mode'.format(execmode))
            self.execmode = execmode
            self.deflectionlib.execmode = self.execmode

        setup(self)
        
        if ptg is None:
            assert dlm_scaled is not None, "Need to provide dlm_scaled if ptg is None"
            # note: dlm2pointing wrapper here always returns double precsision
            pointing_theta, pointing_phi = self.dlm2pointing(dlm_scaled, lmax, verbose)
            del dlm_scaled
        else:
            pointing_theta, pointing_phi = ptg
        nuFFTdtype = cp.float32 if self.nuFFT_single_prec else cp.float64
        pointing_theta, pointing_phi = pointing_theta.astype(nuFFTdtype), pointing_phi.astype(nuFFTdtype)
        y, x = cp.array([pointing_theta, pointing_phi]).T.T[0], cp.array([pointing_theta, pointing_phi]).T.T[1]
        self.nuFFTplan.setpts(x, y, None)
        
        lenmap = self.synthesis_general(lmax, mmax, alm=gclm, loc=cp.array([pointing_theta, pointing_phi]).T, epsilon=self.epsilon, pointmap=lenmap, verbose=verbose)
        
        if self.execmode == 'timing':
            t0, ti = self.timer.reset()
            cp.cuda.runtime.deviceSynchronize()
            lenmap.get()
            cp.cuda.runtime.deviceSynchronize()
            self.timer.add("Transfer <-")
            self.timer.set(t0, ti)
        
        if self.execmode == 'debug':
            print("::debug:: Returned component results")
            self.timer.delete('gclm2lenmap()')
            return self.ret
        del pointing_theta, pointing_phi
        self.timer.delete('gclm2lenmap()')
        return lenmap

    @timing_decorator_close
    def lenmap2gclm(self, lenmap:cp.ndarray, dlm_scaled:cp.ndarray, gclm:cp.ndarray, lmax:int, mmax:int, epsilon=None, ptg=None, verbose=1, execmode='normal', runid=None):
        """ This is outward facing function, expects nuFFT type 1
            
            Note:
                lenmap must be theta contiguous
                For inverse-lensing, need to feed in lensed maps times unlensed forward magnification matrix.
        """
        self.runid = runid
        self.timer.delete('lenmap2gclm()')
        self.timer.start("lenmap2gclm()")
        
        assert lmax == self.deflectionlib.geominfo[1]['lmax'], "Expected lmax={}. lmax must be same as during init, but is {}".format(lmax, self.deflectionlib.geominfo[1]['lmax'])
        assert mmax == self.deflectionlib.geominfo[1]['lmax'], "Expected mmax={}. mmax must be same as lmax during init, but is {}".format(mmax, self.deflectionlib.geominfo[1]['lmax'])
        
        t0, ti = self.timer.reset()
        cp.cuda.runtime.deviceSynchronize()
        if not isinstance(gclm, cp.ndarray):
            gclm = cp.array(gclm)
            print("WARNING: gclm not cupy and has automatically been transferred to device. This may reduce performance. Consider passing a cupy array instead")
        if not isinstance(lenmap, cp.ndarray):
            lenmap = cp.array(lenmap)
            print("WARNING: lenmap not cupy and has automatically been transferred to device. This may reduce performance. Consider passing a cupy array instead")
        if not isinstance(ptg, cp.ndarray) if ptg is not None else False:
            ptg = cp.array(ptg)
            print("WARNING: ptg not cupy and has automatically been transferred to device. This may reduce performance. Consider passing a cupy array instead")
        if not isinstance(dlm_scaled, cp.ndarray):
            dlm_scaled = cp.array(dlm_scaled)
            print("WARNING: dlm_scaled not cupy and has automatically been transferred to device. This may reduce performance. Consider passing a cupy array instead")
        cp.cuda.runtime.deviceSynchronize()
        self.timer.add("Transfer ->")
        self.timer.set(t0, ti)
        
        if self.nuFFTtype:
            # NOTE epsilon parameter only useful if nuFFT not planned
            assert epsilon == self.epsilon if epsilon is not None else True==True, "expected epsilon = {} as set during initialisation of nuFFT plan, but is {}".format(self.epsilon, epsilon)
            epsilon = self.epsilon
        else:
            assert epsilon is not None, "epsilon must be provided if not planned"
            self.epsilon = epsilon
        self.deflectionlib.epsilon = epsilon
        
        # TODO dtype checks - compare to plan if applicable
        self._assert_dtype(lenmap, gclm, dlm_scaled)
        self._assert_precision(lenmap, gclm)
        self._assert_type(lenmap, gclm, dlm_scaled)
        
        # TODO shape checks
        self._assert_shape(lenmap, gclm, dlm_scaled, ndim=2, nbatch=1)
        
        self.single_prec = True if epsilon > 1e-6 else False
        
        def setup(self):
            assert execmode in ['normal', 'debug', 'timing']
            print('Running in {} execution mode'.format(execmode))
            self.execmode = execmode
            self.deflectionlib.execmode = self.execmode
        setup(self)
        
        if ptg is None:
            assert dlm_scaled is not None, "Need to provide dlm_scaled if ptg is None"
            pointing_theta, pointing_phi = self.dlm2pointing(dlm_scaled, lmax, verbose)
            del dlm_scaled
        else:
            pointing_theta, pointing_phi = ptg.T
        pointing_dtype = cp.float32 if self.single_prec else cp.float64
        pointing_theta = self._ensure_dtype(pointing_theta, self.single_prec, isreal=True).astype(pointing_dtype)
        pointing_phi = self._ensure_dtype(pointing_phi, self.single_prec, isreal=True).astype(pointing_dtype)
        
        gclm = self.adjoint_synthesis_general(lmax, mmax, lenmap[0], cp.array([pointing_theta, pointing_phi]).T, self.epsilon, gclm, verbose)
        
        if self.execmode == 'timing':
            t0, ti = self.timer.reset()
            cp.cuda.runtime.deviceSynchronize()
            gclm.get()
            cp.cuda.runtime.deviceSynchronize()
            self.timer.add("Transfer <-")
            self.timer.set(t0, ti)

        if self.execmode == 'debug':
            print("::debug:: Returned component results")
            return self.ret
        del lenmap, pointing_theta, pointing_phi
        self.timer.delete("lenmap2gclm()")
        return gclm
    

    def __synthesis_general_notiming(self, lmax, mmax, pointmap, loc, epsilon, alm, verbose):
        """ This is a blueprint for later reference
        """
        pointing_theta, pointing_phi = loc[0], loc[1]        
        self.synthesis(alm, self.CARmap, lmax=lmax, mmax=mmax)
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
        
    def __adjoint_synthesis_general_notiming(self, lmax, mmax, pointmap, loc, epsilon, alm, verbose):
        """ This is a blueprint for later reference
        """
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

        return self.adjoint_synthesis(synthmap=synthmap, lmax=lmax, mmax=mmax, out=alm)

            
    def hashdict():
        '''
        Compatibility with delensalot
        '''
        return "GPU_cufinufft_transformer"

