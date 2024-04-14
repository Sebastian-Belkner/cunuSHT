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
    def dlm2pointing(self, dlm, pointing_theta, pointing_phi):
        
        # @debug_decorator
        @timing_decorator
        # @shape_decorator
        def _spin__1___synth(self, dlm, out_theta, out_phi):
            self.synthesis_der1_cupy(dlm, out_theta, out_phi, nthreads=self.nthreads)
            return tuple([out_theta, out_phi])
          
        # @debug_decorator
        @timing_decorator
        @shape_decorator
        def _pointing(self, spin1_theta, spin1_phi, cpt, cpphi0, cpnph, cpofs, pointing_theta, pointing_phi):
            podo.Cpointing_1Dto1D(cpt, cpphi0, cpnph, cpofs, spin1_theta, spin1_phi, pointing_theta, pointing_phi)
            return tuple([pointing_theta, pointing_phi])
        
        ll = np.arange(0,self.lmax_dlm+1,1)
        scaled = hp.almxfl(dlm, np.nan_to_num(np.sqrt(1/(ll*(ll+1)))))
        self.timer.add('dlm2pointing - dlm scaling')
        
        scaled = cp.array(scaled, dtype=np.complex)
        cpt = cp.array(self.geom.theta.astype(np.double), dtype=cp.double)
        cpphi0 = cp.array(self.geom.phi0, dtype=cp.double)
        cpnph = cp.array(self.geom.nph, dtype=cp.uint64)
        cpofs = cp.array(self.geom.ofs, dtype=cp.uint64)
        spin1_theta = cp.zeros((self.constructor.spat_shape), dtype=cp.double)
        spin1_phi = cp.zeros((self.constructor.spat_shape), dtype=cp.double)
        self.timer.add('dlm2pointing - allocation')

        _spin__1___synth(self, scaled, spin1_theta, spin1_phi)
        _pointing(self, spin1_theta.T.flatten(), spin1_phi.T.flatten(), cpt, cpphi0, cpnph, cpofs, pointing_theta, pointing_phi)
        
        del spin1_theta, spin1_phi, cpt, cpphi0, cpnph, cpofs, scaled
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
    
    def __getattr__(self, name):
        return getattr(self.instance, name)
        

    def gclm2lenmap_cupy(self, gclm, dlm, lmax, mmax, spin, pointing_theta:cp.array = None, pointing_phi:cp.array = None, nthreads=None, polrot=True, execmode='calc'):
        """
        Same as gclm2lenmap, but using cupy allocated intermediate results (synth, doubling, c2c, nuFFt),
        so no transfers needed to CPU between them.
        
        gclm and dlm are assumed to be on host, will be transfered in _setup().
        Can provide pointing_theta and pointing_phi to avoid dlm2pointing() call.
        """
        self.ret = {}
            
        @timing_decorator
        def _setup(self, gclm, execmode, nthreads):
            assert execmode in ['normal','debug', 'timing']
            print('Running in {} execution mode')
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
            cp.cuda.runtime.deviceSynchronize()
            return tuple([out])
        
        @debug_decorator
        @timing_decorator
        @shape_decorator
        def _C2C(self, map_in, out):
            out = scipy.fft.fft2(map_in, norm='forward')
            cp.cuda.runtime.deviceSynchronize()
            return tuple([out])
        
        @debug_decorator
        @timing_decorator
        @shape_decorator
        def _nuFFT(self, fc, ptg_theta, ptg_phi, result):
            result = cufinufft.nufft2d2(data=fc, x=ptg_theta, y=ptg_phi, isign=1, eps=self.epsilon)
            cp.cuda.runtime.deviceSynchronize()
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
        result = cufinufft.nufft2d2(x=pointing_phi, y=pointing_theta, data=fc, isign=1)
        self.timer.add("nuFFT init")
        lenmap = _nuFFT(self, fc, pointing_phi, pointing_theta, lenmap)
        self.timer.add('gclm2lenmap')
        result = lenmap[0].get()
        self.timer.add('Transfer <-')
        
        if self.execmode == 'timing':
            self.timer.dumpjson(os.path.dirname(pysht.__file__)[:-5]+'/test/benchmark/timings/GPU_cufinufft_{}'.format(lmax))
            print(self.timer)
            print("::timing:: stored new timing data for lmax {}".format(lmax))
        if self.execmode == 'debug':
            print("::debug:: Returned component results")
            return self.ret
        del fc, lenmap, pointing_theta, pointing_phi
        return result

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
        input_sht_mode = ducc_sht_mode(gclm, spin)
        if mmax_out is None:
            mmax_out = lmax_out
        m = self.gclm2lenmap(gclm, dlm=dlm, lmax=lmax_out, mmax=lmax_out, spin=spin, nthreads=nthreads, polrot=polrot)
        if gclm_out is not None:
            assert gclm_out.dtype == ctype[m.dtype], 'type precision must match'
        gclm_out = self.adjoint_synthesis(m, spin=spin, lmax=lmax_out, mmax=mmax_out, nthreads=nthreads, alm=gclm_out, mode=input_sht_mode)
        return gclm_out.squeeze()
    
    
    def synthesis_general(self, lmax, mmax, map, loc, spin, epsilon, nthreads, sht_mode, alm, verbose):
        assert 0, "implement if needed"
        # return self.gclm2lenmap_cupy(lmax=lmax, mmax=mmax, alm=alm, pointing_theta=loc[0], pointing_phi=loc[1], spin=spin, epsilon=self.epsilon, nthreads=self.nthreads, mode=sht_mode, verbose=self.verbosity)
    
    def adjoint_synthesis_general(self, lmax, mmax, map, loc, spin, epsilon, nthreads, sht_mode, alm, verbose):
        assert 0, "implement if needed"
        # return self.lenmap2gclm_cupy(lmax=lmax, mmax=mmax, map=map, loc=loc, spin=spin, epsilon=self.epsilon, nthreads=self.nthreads, mode=sht_mode, alm=alm, verbose=self.verbosity)

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

