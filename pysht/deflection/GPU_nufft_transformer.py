import numpy as np
import os
from lenspyx.utils_hp import Alm
from lenspyx.utils_hp import Alm, alm2cl, almxfl, alm_copy
from lenspyx.utils import timer, blm_gauss
from lenspyx.remapping.utils_angles import d2ang
from pysht import cacher

import ducc0
import cufinufft
import cupy as cp
import time
import healpy as hp
import ctypes
import functools
import time

import sys
import pysht.c.podo_interface as podo

import pysht
import line_profiler

import cupyx.scipy.fft as cufft
import scipy.fft
scipy.fft.set_global_backend(cufft)

import pysht.geometry as geometry
from pysht.geometry import Geom
from pysht.utils import timer
from pysht.helper import shape_decorator
from pysht.sht.GPU_sht_transformer import GPU_SHT_pySHT_transformer, GPU_SHTns_transformer
from pysht.sht.CPU_sht_transformer import CPU_SHT_DUCC_transformer

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


class c_complex(ctypes.Structure):
    _fields_ = [("real", ctypes.c_double),
                ("imag", ctypes.c_double)]

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
    
try: 
    import skcuda.fft as cu_fft
    import pycuda
    from pycuda import gpuarray
    import pycuda.autoinit
    # import pycuda.driver as cuda
    # cuda.init() 
    HAS_CUFFT = True
    print('Successfully imported skcuda.fft')
except:
    print("Could not import skcuda.fft")
    HAS_CUFFT = False

@staticmethod
def ducc_sht_mode(gclm, spin):
    gclm_ = np.atleast_2d(gclm)
    return 'GRAD_ONLY' if ((gclm_[0].size == gclm_.size) * (abs(spin) > 0)) else 'STANDARD'


class deflection:
    def __init__(self, dlm, mmax_dlm:int or None, geom, epsilon=1e-7, verbosity=0, single_prec=True, planned=False, nthreads=10):
        self.single_prec = single_prec
        self.verbosity = 1
        self.tim = timer(verbose=self.verbosity)
        self.nthreads = nthreads
        self.planned = False
        self._cis = False
        self.cacher = cacher.cacher_mem()
        self.epsilon = epsilon
        
        dlm = np.atleast_2d(dlm)
        self.dlm = dlm
        
        self.lmax_dlm = Alm.getlmax(dlm[0].size, mmax_dlm)
        self.mmax_dlm = mmax_dlm
        
        s2_d = np.sum(alm2cl(dlm[0], dlm[0], self.lmax_dlm, mmax_dlm, self.lmax_dlm) * (2 * np.arange(self.lmax_dlm + 1) + 1)) / (4 * np.pi)
        if dlm.shape[0]>1:
            s2_d += np.sum(alm2cl(dlm[1], dlm[1], self.lmax_dlm, mmax_dlm, self.lmax_dlm) * (2 * np.arange(self.lmax_dlm + 1) + 1)) / (4 * np.pi)
            s2_d /= np.sqrt(2.)
        sig_d = np.sqrt(s2_d / geom.fsky())
        sig_d_amin = sig_d / np.pi * 180 * 60
        if sig_d >= 0.01:
            print('deflection std is %.2e amin: this is really too high a value for something sensible'%sig_d_amin)
        elif self.verbosity:
            print('deflection std is %.2e amin' % sig_d_amin)

    # @profile
    def _build_angles(self, synth_spin1_map, lmax_dlm, mmax_dlm, calc_rotation=True, HAS_DUCCPOINTING=True):
        """Builds deflected positions and angles
            Returns (npix, 3) array with new tht, phi and -gamma
        """
        fns = ['ptg'] + calc_rotation * ['gamma']
        if not np.all([self.cacher.is_cached(fn) for fn in fns]):
            # Probably want to keep red, imd double precision for the calc?
            if HAS_DUCCPOINTING:
                print('Calculating pointing angles using ducc0.misc.get_deflected_angles()')
                # FIXME if shtns is used, we need to access SHTns info about geom.
                tht, phi0, nph, ofs = self.geom.theta, self.geom.phi0, self.geom.nph, self.geom.ofs
                tht_phip_gamma = get_deflected_angles(theta=tht, phi0=phi0, nphi=nph, ringstart=ofs, deflect=synth_spin1_map.T,
                                                        calc_rotation=calc_rotation, nthreads=self.nthreads)
                if calc_rotation:
                    self.cacher.cache(fns[0], tht_phip_gamma[:, 0:2])
                    self.cacher.cache(fns[1], tht_phip_gamma[:, 2] if not self.single_prec else tht_phip_gamma[:, 2].astype(np.float32))
                else:
                    self.cacher.cache(fns[0], tht_phip_gamma)
                return
            print('Calculating pointing angles using lenspyx')
            npix = self.geom.npix()
            thp_phip_gamma = np.empty((3, npix), dtype=float)  # (-1) gamma in last arguement
            startpix = 0
            assert np.all(self.geom.theta > 0.) and np.all(self.geom.theta < np.pi), 'fix this (cotangent below)'
            red, imd = synth_spin1_map
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


    def pointing_GPU(self, synth_spin1_map):
        # print('+++++++++ Inside pointing GPU +++++++++')
        thetas_, phi0_, nphis_, ringstarts_ = self.geom.theta.astype(float), self.geom.phi0.astype(float), self.geom.nph.astype(int), self.geom.ofs.astype(int)
        red_, imd_ = synth_spin1_map.astype(np.double)
        npix_ = int(sum(nphis_))
        nrings_ = int(nphis_.size)
        output_array_ = np.zeros(shape=synth_spin1_map.size, dtype=np.double)
        thetas, phi0, nphis, ringstarts = thetas_, phi0_, nphis_, ringstarts_
        red, imd = red_, imd_
        npix = npix_
        nrings = nrings_
        output_array = output_array_
        
        self.timer.add('get pointing - data grab')
        popy.Cpointing(thetas, phi0, nphis, ringstarts, red, imd, nrings, npix, output_array)
        self.timer.add('get pointing - calc and transfer<->')
        
        ret = np.array(output_array, dtype=np.double)
        print("done: max value = {}, shape = {}, snynthmapshape: {}".format(np.max(ret), ret.shape, synth_spin1_map.shape))
        _ = ret.reshape(synth_spin1_map.shape).T
        self.cacher.cache('ptg', _)
        return _
        # return output_array
    

    def change_dlm(self, dlm:list or np.ndarray, mmax_dlm:int or None, cacher:cacher.cacher or None=None):
        assert len(dlm) == 2, (len(dlm), 'gradient and curl mode (curl can be none)')
        self.dlm = dlm
        self.mmax_dlm = mmax_dlm
        return self


class GPU_cufinufft_transformer(deflection):
    def __init__(self, shttransformer_desc, geominfo, deflection_kwargs):
        self.backend = 'GPU'
        self.shttransformer_desc = shttransformer_desc
        
        if shttransformer_desc == 'shtns':
            self.BaseClass = type('GPU_SHTns_transformer', (GPU_SHTns_transformer,), {})
            self.instance = self.BaseClass(geominfo)
        elif shttransformer_desc == 'ducc':
            self.BaseClass = type('CPU_SHT_DUCC_transformer()', (CPU_SHT_DUCC_transformer,), {})
            self.instance = self.BaseClass(geominfo)
        elif shttransformer_desc == 'pysht':
            assert 0, "implement if needed"
            self.BaseClass = type('GPU_SHT_pySHT_transformer', (GPU_SHT_pySHT_transformer,), {})
            self.instance = self.BaseClass(geominfo)
        else:
            raise ValueError('shttransformer_desc must be either "ducc" or "shtns" or "pysht"')
            
        self.geominfo = geominfo
        self.set_geometry(geominfo)
        if 'mmax' in geominfo[1]:
            del geominfo[1]['mmax']
        self.nufftgeom = geometry.get_geom(geominfo)
        deflection_kwargs.update({'geom':self.nufftgeom})    
        super().__init__(**deflection_kwargs)


    def __getattr__(self, name):
        return getattr(self.instance, name)


    def set_nufftgeometry(self, geominfo):
        self.geominfo = geominfo
        self.nufftgeom = geometry.get_geom(geominfo)
        self.set_geometry(geominfo)

    def gclm2lenmap_cupy(self, gclm, dlm, lmax, mmax, spin, nthreads, polrot=True, cc_transformer=None, HAS_DUCCPOINTING=True, mode=0):
        """
        Same as gclm2lenmap, but using cupy allocated gclm and dlm.
        SHTns can work with this via cu_SH_to_spat()
        The returned pointer is a pointer to the device memory, we pass it to the doubling kernel.
        
        SHTns also provides the synthesis_der1_cupy() method, which is a wrapper to the gradient synthesis,
        this returns a pointer to device memory for red and imd, these are passed to the pointing kernel
        
        the doubling map and the pointing map are now both in device memory, we can pass both to cufinufft
        """
        ret = []
        def timing_decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                tkey = func.__name__.replace('___', ' ').replace('__', '-').replace('_', '')
                self.timer.reset()
                cp.cuda.runtime.deviceSynchronize()
                _ = func(*args, **kwargs)
                cp.cuda.runtime.deviceSynchronize()
                self.timer.add(tkey)
                print(15*"- "+"Timing {}: {:.3f} seconds".format(tkey, self.timer.keys[tkey]) + 15*"- ")
                return _
            return wrapper
        
        @timing_decorator
        def _setup(gclm, lmax, mmax, mode):
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
                
            gclm = np.atleast_2d(gclm)
            lmax_unl = Alm.getlmax(gclm[0].size, mmax)
            if mmax is None:
                mmax = lmax_unl
            if self.single_prec and gclm.dtype != np.complex64:
                gclm = gclm.astype(np.complex64)

            if False: #not debug:
                # FIXME this only works if CAR grid is initialized with good fft size, otherwise this clashes with doubling
                ntheta = ducc0.fft.good_size(lmax_unl + 2)
                nphihalf = ducc0.fft.good_size(lmax_unl + 1)
                nphi = 2 * nphihalf
            else:
                ntheta = lmax+1
                nphihalf = lmax+1
                nphi = 2 * nphihalf
            return gclm, lmax, mmax, ntheta, nphihalf, nphi, timing, debug
        
        @timing_decorator
        def _synthesis(gclm, out):
            """In goes a (1,nalm) gclm, out comes a (1,nalm) in out
            """
            # cc_transformer.synthesis_cupy(gclm, out, spin=0, lmax=lmax, mmax=mmax, nthreads=nthreads)
            return cc_transformer.synthesis_cupy(gclm, out, spin=0, lmax=lmax, mmax=mmax, nthreads=nthreads)

        @timing_decorator
        def _spin__1___synth(dlm, out_theta, out_phi):
            self.synthesis_der1_cupy(dlm, out_theta, out_phi, nthreads=self.nthreads)

        @timing_decorator
        def _doubling(map, ntheta, nphi, out):
            podo.Cdoubling_1D(map, ntheta, nphi, out)
        
        @timing_decorator
        def _C2C(map):
            def from_cupy(arr):
                shape = arr.shape
                dtype = arr.dtype
                def alloc(x):
                    return arr.data.ptr
                if arr.flags.c_contiguous:
                    order = 'C'
                elif arr.flags.f_contiguous:
                    order = 'F'
                else:
                    raise ValueError('arr order cannot be determined')
                return gpuarray.GPUArray(shape=shape,
                                        dtype=dtype,
                                        allocator=alloc,
                                        order=order)
            return scipy.fft.ifft(map)
        
        @timing_decorator
        def _pointing(spin1_theta, spin1_phi, cpt, cpphi0, cpnph, cpofs, pointing_theta, pointing_phi):
            podo.Cpointing_1Dto1D(cpt, cpphi0, cpnph, cpofs, spin1_theta, spin1_phi, pointing_theta, pointing_phi)
        
        @timing_decorator
        def _nufft(fc, ptg_theta, ptg_phi):
            time.sleep(0.07)
            return 0
            # return cufinufft.nufft2d2(x=pointing_theta, y=pointing_phi, data=fc, isign=1)
        
        
        cpCARmap = np.random.randn(cc_transformer.constructor.nlat, cc_transformer.constructor.nphi)
        cpCARmap = cp.array(cpCARmap, dtype=np.double)
        self.timer = timer(1, prefix=self.backend)
        self.timer.start('gclm2lenmap()')
        gclm, lmax, mmax, ntheta, nphihalf, nphi, timing, debug = _setup(gclm, lmax, mmax, mode)
        
        #TODO These allocations can move out of the pipeline
        doubledmap = cp.zeros(2*self.geom.npix(), dtype=cp.double)
        spin1_theta = cp.zeros(self.geom.npix(), dtype=cp.double)
        spin1_phi = cp.zeros(self.geom.npix(), dtype=cp.double)
        pointing_theta = cp.zeros(self.geom.npix(), dtype=cp.double)
        pointing_phi = cp.zeros(self.geom.npix(), dtype=cp.double)
        cpgclm = cp.array(gclm, dtype=np.complex)
        cpt = cp.array(self.geom.theta.astype(np.double), dtype=cp.double)
        cpphi0 = cp.array(self.geom.phi0, dtype=cp.double)
        cpnph = cp.array(self.geom.nph, dtype=cp.uint64)
        cpofs = cp.array(self.geom.ofs, dtype=cp.uint64)
        self.timer.add('Transfers ->')
        
        ll = np.arange(0,self.lmax_dlm+1,1)
        scaled = hp.almxfl(dlm, np.nan_to_num(np.sqrt(1/(ll*(ll+1)))))
        scaled = cp.array(scaled, dtype=np.complex)
        self.timer.add('dlm scaling')
        
        _spin__1___synth(scaled, spin1_theta, spin1_phi)
        # print("Shape of spin1_theta: ", spin1_theta.shape)
        # print("spin1_theta: ", spin1_theta)
        # print("Shape of spin1_phi: ", spin1_theta.shape)
        # print("spin1_phi: ", spin1_phi)
        
        _synthesis(cpgclm, cpCARmap)
        # print("Shape of cpCARmap: ", cpCARmap.shape)
        # print("cpCARmap: ", cpCARmap)
        
        _doubling(cpCARmap.flatten(), np.int(ntheta), np.int(nphi), doubledmap)
        # print("Shape of doubledmap: ", doubledmap.shape)
        # print("doubledmap: ", doubledmap)
        
        cpfc = _C2C(doubledmap)
        # print("Shape of cpfc: ", cpfc.shape)
        # print("cpfc: ", cpfc)
        
        _pointing(spin1_theta, spin1_phi,cpt, cpphi0, cpnph, cpofs, pointing_theta, pointing_phi)
        # print("Shape of doubledmap: ", doubledmap.shape)
        # print("doubledmap: ", doubledmap)
        
        deflectedmap = _nufft(cpfc.reshape(2*ntheta,-1), pointing_theta, pointing_phi)
        cpCARmap.get()
        self.timer.add('Transfer <-')
        if timing:
            self.timer.dumpjson(os.path.dirname(pysht.__file__)[:-5]+'/test/benchmark/timings/GPU_cufinufft_{}'.format(lmax))
            print(self.timer)
        
        print("stored new data")
        # return deflectedmap.get().real.flatten()

    # @profile
    def gclm2lenmap(self, gclm, dlm, lmax, mmax, spin, nthreads, polrot=True, cc_transformer=None, HAS_DUCCPOINTING=True, mode=0):
        """GPU algorithm for spin-n remapping using cufinufft
            Args:
                gclm: input alm array, shape (ncomp, nalm), where ncomp can be 1 (gradient-only) or 2 (gradient or curl)
                mmax: mmax parameter of alm array layout, if different from lmax
                spin: spin (>=0) of the transform
                backwards: forward or backward (adjoint) operation
        """ 
        ret = []
        
        def _setup(gclm, lmax, mmax, mode):
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
                
            gclm = np.atleast_2d(gclm)
            lmax_unl = Alm.getlmax(gclm[0].size, mmax)
            if mmax is None:
                mmax = lmax_unl
            if self.single_prec and gclm.dtype != np.complex64:
                gclm = gclm.astype(np.complex64)

            if False: #not debug:
                # FIXME this only works if CAR grid is initialized with good fft size, otherwise this clashes with doubling
                ntheta = ducc0.fft.good_size(lmax_unl + 2)
                nphihalf = ducc0.fft.good_size(lmax_unl + 1)
                nphi = 2 * nphihalf
            else:
                ntheta = lmax+1
                nphihalf = lmax+1
                nphi = 2 * nphihalf
            self.timer.add('setup')
            return gclm, lmax, mmax, ntheta, nphihalf, nphi, timing, debug
        
        @shape_decorator
        def _synthCAR(gclm):
            """Generates phi contiguuous array from any SHT transformer. However, only view is set, so likely in memory still theta contiguous.
            #FIXME keep this theta contiguos and rewrite the doubling to theta contiguous
            Args:
                gclm (_type_): _description_
            Returns:
                _type_: _description_
            """
            # shtns cc_transformer returns theta contiguous 1d array
            map = cc_transformer.synthesis(gclm, spin=0, lmax=lmax, mmax=mmax, nthreads=nthreads)
            
            # the double .T are only needed to get the added dimension as the leading dimension
            if cc_transformer.theta_contiguous:
                map = np.atleast_3d(map.reshape(-1, lmax+1)).T
            else:
                map = np.atleast_3d(map.reshape(lmax+1,-1).T).T
            self.timer.add('CAR synthesis')
            if debug:
                ret.append(np.copy(map))
            return map
         
        def _CAR2dmap(map):
            """expects a phi contiguous array
            Args:
                map (_type_): _description_
            Returns:
                _type_: _description_
            """
            # %timeit say 80 ms for 2058x4096 on CPU, what about GPU?
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
            self.timer.add('doubling (CPU)')
            if debug:
                ret.append(np.copy(map_dfs))
            return map_dfs
             
        def _C2C(map):
            """Map to fourier space, i.e. inverse FFT. Expect a phi contiguous doubled map

            Args:
                map (_type_): _description_

            Returns:
                _type_: _description_
            """
            # cufft interface gives me 11 ms as opposed to duccs 65 ms for 2058x4096
            if True: #HAS_CUFFT:
                if spin == 0:
                    if True: # cupy version
                        BATCH, NX = map.T.shape
                        fc  = gpuarray.empty((BATCH,NX),dtype=np.complex128)
                        plan = cu_fft.Plan(map.shape, np.complex128, np.complex128)
                        data_t_gpu  = gpuarray.to_gpu(map)
                        self.timer.add('c2c - GPU transfer<-')
                        cu_fft.ifft(data_t_gpu, fc, plan)
                        fc = fc.get()
                        self.timer.add('c2c - GPU calc and transfer->')
                    elif False:# cupy version
                        # BATCH, NX = map.T.shape
                        fc  = cp.asarray(np.zeros_like(map.T))
                        data_gpu  = cp.asarray(map.T)
                        plan = cu_fft.Plan(map.T.shape, np.complex128, np.complex128)
                        self.timer.add('c2c - GPU transfer<-')
                        cu_fft.ifft(data_gpu, fc, plan)
                        self.timer.add('c2c - GPU calc and transfer->')
                else:
                    assert 0, "implement if needed"
            else:
                fc = np.empty(map.shape, dtype=np.complex128)
                if spin == 0:
                    ducc0.fft.c2c(map, axes=(0, 1), inorm=2, nthreads=self.nthreads, out=fc)
                else:
                    fc = ducc0.fft.c2c(map, axes=(0, 1), inorm=2, nthreads=self.nthreads, out=fc)
            # self.timer.add('c2c')
            if debug:
                ret.append(np.copy(fc))
            return fc
        
        def _getdeflectedgrid():
            ptg = None
            if ptg is None:
                synth_spin1_map = self._build_d1(dlm, lmax, mmax)
                # self.timer.add('spin-1 maps')
                if True:
                    ptg = self.pointing_GPU(synth_spin1_map)
                    ptg = self.cacher.load('ptg')
                else:
                    self._build_angles(synth_spin1_map, mmax, mmax, HAS_DUCCPOINTING=HAS_DUCCPOINTING) if not self._cis else self._build_angleseig()
                    ptg = self.cacher.load('ptg')
            # self.timer.add('get pointing')
            if debug:
                ret.append(np.copy(ptg))
            return ptg, synth_spin1_map
        
        def _nufft(fc, ptg, smap):
            fcshifted = np.fft.fftshift(fc, axes=(0,1))
            self.timer.add('fftshift')
            # fcshifted = fcshifted.astype(np.complex128)
            data_ = cp.array(fcshifted)
            x_ = cp.array(ptg[:,0])
            y_ = cp.array(ptg[:,1])
            # data_ = dgpu #cp.array(fcshifted)
            # x_ = cp.array(ptg[:smap.shape[0]])
            # y_ = cp.array(ptg[:smap.shape[0]])
            # self.timer.add('cupy array creation')
            v_ = cufinufft.nufft2d2(x=x_, y=y_, data=data_, isign=1)
            self.timer.add('nuFFT')
            values = np.real(v_.get())
            if debug:
                ret.append(np.copy(values))
            return values
                
        self.timer = timer(1, prefix=self.backend)
        self.timer.start('gclm2lenmap()')
        
        gclm, lmax, mmax, ntheta, nphihalf, nphi, timing, debug = _setup(gclm, lmax, mmax, mode)
        CARmap = _synthCAR(gclm) # alm to phi cont
        map_dfs = _CAR2dmap(CARmap) # phi cont to theta cont
        fc = _C2C(map_dfs) # theta cont to fourier space
        ptg, smap = _getdeflectedgrid()
        deflectedmap = _nufft(fc, ptg, smap)

        if polrot * spin:
            if self._cis:
                cis = self._get_cischi()
                for i in range(polrot * abs(spin)):
                    deflectedmap *= cis
            else:
                if HAS_DUCCROTATE:
                    lensing_rotate(deflectedmap, self._get_gamma(), spin, self.nthreads)
                else:
                    func = fremap.apply_inplace if deflectedmap.dtype == np.complex128 else fremap.apply_inplacef
                    func(deflectedmap, self._get_gamma(), spin, self.nthreads)
            self.timer.add('rotation')
        if debug:
            ret.append(np.copy(deflectedmap.real.flatten())) 
        
        if timing:
            self.timer.dumpjson(os.path.dirname(pysht.__file__)[:-5]+'/test/benchmark/timings/GPU_cufinufft_{}'.format(lmax))
        if debug:
            return ret
        else:
            return deflectedmap.real.flatten() if spin == 0 else deflectedmap.view(rtype[deflectedmap.dtype]).reshape((deflectedmap.size, 2)).T


    def lenmap2gclm(self, points:np.ndarray[complex or float], dlm, spin:int, lmax:int, mmax:int, nthreads:int, gclm_out=None, sht_mode='STANDARD'):
        """
            Note:
                points mst be already quadrature-weigthed
                For inverse-lensing, need to feed in lensed maps times unlensed forward magnification matrix.
        """
        self.tim.reset()
        if spin == 0 and not np.iscomplexobj(points):
            points = points.astype(ctype[points.dtype]).squeeze()
        if spin > 0 and not np.iscomplexobj(points):
            points = (points[0] + 1j * points[1]).squeeze()
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
        slm = ducc0.sht.experimental.adjoint_synthesis_2d(map=map, spin=spin, lmax=lmax, mmax=mmax, geometry="CC", nthreads=self.nthreads, mode=sht_mode, alm=gclm_out)
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
        self.tim.start(stri)
        self.tim.reset()
        input_sht_mode = ducc_sht_mode(gclm, spin)
        if mmax_out is None:
            mmax_out = lmax_out
        m = self.gclm2lenmap(gclm, dlm=dlm, lmax=lmax_out, mmax=lmax_out, spin=spin, nthreads=nthreads, polrot=polrot)
        self.tim.reset()
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

    def flip_tpg_2d(self, m):
        # FIXME this should probably be lmax, not lmax_dlm
        # dim of m supposedly (2, -1)
        buff = np.array([_.reshape(2*(self.lmax_dlm+1),-1).T.flatten() for _ in m])
        return buff

    def _build_d1(self, dlm, lmax_dlm, mmax_dlm, dclm=None):
        '''
        # FIXME this is a bit of a mess, this function should not distinguish between different SHT backends.
        # Instead, there should be a _build_d1() for each, and they should sit in the repsective transformer modules.
        
        This depends on the backend. If SHTns, we can use the synthesis_der1 method. If not, we use a spin-1 SHT
        '''
        ll = np.arange(0,lmax_dlm+1,1)
        if self.shttransformer_desc == 'shtns':
            if dclm is None:
                scaled = hp.almxfl(dlm, np.nan_to_num(np.sqrt(1/(ll*(ll+1)))))
                self.timer.add(('spin-1 maps - alm scaling'))
                synth_spin1_map = self.synthesis_der1(scaled, nthreads=self.nthreads)
                self.timer.add(('spin-1 maps - synthesis'))
            else:
                assert 0, "implement if needed, not sure if this is possible with SHTns"
                dgclm = np.empty((2, dlm.size), dtype=dlm.dtype)
                dgclm[0] = dlm
                dgclm[1] = dclm
                synth_spin1_map = self.synthesis_der1(hp.almxfl(dlm, np.nan_to_num(np.sqrt(1/(ll*(ll+1))))), nthreads=self.nthreads)
            flipped = self.flip_tpg_2d(synth_spin1_map)
            self.timer.add(('spin-1 maps - flip'))
            return flipped
        elif self.shttransformer_desc == 'pysht':
            assert 0, "implement if needed"
        else:
            assert 0, "Not sure what to do with {}".format(self.shttransformer_desc)
            
            
    def hashdict():
        '''
        Compatibility with delensalot
        '''
        return "GPU_cufinufft_transformer"

