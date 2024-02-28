import numpy as np

from lenspyx.utils_hp import Alm
from lenspyx.utils_hp import Alm, alm2cl, almxfl, alm_copy
from lenspyx.utils import timer, blm_gauss
from lenspyx.remapping.utils_angles import d2ang
from lenspyx import cachers

import ducc0
import finufft

import pysht.geometry as geometry
from pysht.geometry import Geom
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
    def __init__(self, lens_geom:Geom, dglm, mmax_dlm:int or None, numthreads:int=0, cacher:cachers.cacher or None=None, dclm:np.ndarray or None=None, epsilon=1e-5, verbosity=0, single_prec=True, planned=False):
        self.single_prec = True
        self.verbosity = 1
        self.tim = timer(verbose=self.verbosity)
        self.sht_tr = 4
        self.planned = False
        self._cis = False
        self.cacher = cachers.cacher_mem()
        self.epsilon = 1e-7
        
        # TODO these guys need to be set
        self.dlm = None
        self.lmax_dlm = None
        self.mmax_dlm = None
        
        s2_d = np.sum(alm2cl(dlm, dlm, lmax, mmax, lmax) * (2 * np.arange(lmax + 1) + 1)) / (4 * np.pi)
        sig_d = np.sqrt(s2_d / self.geom.fsky())
        sig_d_amin = sig_d / np.pi * 180 * 60
        if self.sig_d >= 0.01:
            print('deflection std is %.2e amin: this is really too high a value for something sensible'%sig_d_amin)
        elif self.verbosity:
            print('deflection std is %.2e amin' % sig_d_amin)
            

    def set_nufftgeometry(self, geom_desc):
        self.nufftgeom = geometry.get_geom(geom_desc)
        self.set_geometry(geom_desc)


    def _build_d1(self, dlm, lmax_dlm, mmax_dlm, dclm=None):
        if dclm is None:
            # undo p2d to use
            d1 = self.synthesis(dlm, spin=1, lmax=lmax_dlm, mmax=mmax_dlm, nthreads=self.sht_tr, mode='GRAD_ONLY')
        else:
            # FIXME: want to do that only once
            dgclm = np.empty((2, dlm.size), dtype=dlm.dtype)
            dgclm[0] = dlm
            dgclm[1] = dclm
            d1 = self.synthesis(dgclm, spin=1, lmax=lmax_dlm, mmax=mmax_dlm, nthreads=self.sht_tr)
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


class CPU_finufft_transformer(deflection):
    def __init__(self, shttransformer_desc):
        self.backend = 'CPU'
        self.single_prec = True
        self.verbosity = 1
        self.tim = timer(verbose=self.verbosity)
        self.sht_tr = 4
        self.planned = False
        self._cis = False
        self.cacher = cachers.cacher_mem()
        self.epsilon = 1e-7
        
        if shttransformer_desc == 'ducc':
            self.BaseClass = type('CPU_SHT_DUCC_transformer()', (CPU_SHT_DUCC_transformer,), {})
            self.instance = self.BaseClass()
        elif shttransformer_desc == 'shtns':
            self.BaseClass = type('CPU_SHT_SHTns_transformer()', (CPU_SHT_SHTns_transformer,), {})
            self.instance = self.BaseClass()
        else:
            raise ValueError('shttransformer_desc must be either "ducc" or "shtns"')


    def __getattr__(self, name):
        return getattr(self.instance, name)


    def set_nufftgeometry(self, geom_desc):
        self.nufftgeom = geometry.get_geom(geom_desc)
        self.set_geometry(geom_desc)


    def gclm2lenmap(self, gclm, dlm, lmax, mmax, spin, nthreads, polrot=True):
        """CPU algorithm for spin-n remapping using finufft
            Args:
                gclm: input alm array, shape (ncomp, nalm), where ncomp can be 1 (gradient-only) or 2 (gradient or curl)
                mmax: mmax parameter of alm array layout, if different from lmax
                spin: spin (>=0) of the transform
                backwards: forward or backward (adjoint) operation
        """ 
        s2_d = np.sum(alm2cl(dlm, dlm, lmax, mmax, lmax) * (2 * np.arange(lmax + 1) + 1)) / (4 * np.pi)
        sig_d = np.sqrt(s2_d / self.geom.fsky())
        sig_d_amin = sig_d / np.pi * 180 * 60
        if sig_d >= 0.01:
            print('deflection std is %.2e amin: this is really too high a value for something sensible'%sig_d_amin)
        elif self.verbosity:
            print('deflection std is %.2e amin' % sig_d_amin)
            
        gclm = np.atleast_2d(gclm)
        lmax_unl = Alm.getlmax(gclm[0].size, mmax)
        if mmax is None:
            mmax = lmax_unl
        if self.single_prec and gclm.dtype != np.complex64:
            gclm = gclm.astype(np.complex64)
            self.tim.add('type conversion')

        # transform slm to Clenshaw-Curtis map
        ntheta = ducc0.fft.good_size(lmax_unl + 2)
        nphihalf = ducc0.fft.good_size(lmax_unl + 1)
        nphi = 2 * nphihalf
        # Is this any different to scarf wraps ? NB: type of map, map_df, and FFTs will follow that of input gclm
        mode = ducc_sht_mode(gclm, spin)
        map = ducc0.sht.experimental.synthesis_2d(alm=gclm, ntheta=ntheta, nphi=nphi,
                                spin=spin, lmax=lmax_unl, mmax=mmax, geometry="CC", nthreads=self.sht_tr, mode=mode)
        # extend map to double Fourier sphere map
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

        # go to Fourier space
        if spin == 0:
            tmp = np.empty(map_dfs.T.shape, dtype=np.complex128)
            map_dfs = ducc0.fft.c2c(map_dfs.T, axes=(0, 1), inorm=2, nthreads=self.sht_tr, out=tmp)
            del tmp
        else:
            map_dfs = ducc0.fft.c2c(map_dfs, axes=(0, 1), inorm=2, nthreads=self.sht_tr, out=map_dfs)

        if self.planned: # planned nufft
            assert ptg is None
            plan = self.make_plan(lmax_unl, spin)
            values = plan.u2nu(grid=map_dfs, forward=False, verbosity=self.verbosity)
            self.tim.add('planned u2nu')
        else:
            ptg = None
            if ptg is None:
                ptg = self._get_ptg(dlm, mmax)
            self.tim.add('get ptg')

            map_shifted = np.fft.fftshift(map_dfs, axes=(0,1))
            v_ = finufft.nufft2d2(x=ptg[:,0], y=ptg[:,1], f=map_shifted.astype(np.complex128))
            values = np.roll(np.real(v_).reshape(lmax+1,-1), int(self.geom.nph[0]/2-1), axis=1)
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
        return (values.real, ptg, map_dfs) if spin == 0 else values.view(rtype[values.dtype]).reshape((values.size, 2)).T


    def lenmap2gclm(self, points:np.ndarray[complex or float], spin:int, lmax:int, mmax:int, gclm_out=None, sht_mode='STANDARD'):
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
        ptg = self._get_ptg()

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
                                       epsilon=self.epsilon, nthreads=self.sht_tr, verbosity=self.verbosity,
                                       periodicity=2 * np.pi, fft_order=True)
        # go to position space
        map_dfs = ducc0.fft.c2c(map_dfs, axes=(0, 1), forward=False, inorm=2, nthreads=self.sht_tr, out=map_dfs)

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
        slm = ducc0.sht.experimental.adjoint_synthesis_2d(map=map, spin=spin, lmax=lmax, mmax=mmax, geometry="CC", nthreads=self.sht_tr, mode=sht_mode, alm=gclm_out)
        return slm.squeeze()
    

    def lensgclm(self, gclm:np.ndarray, mmax:int or None, spin:int, lmax_out:int, mmax_out:int or None, gclm_out:np.ndarray=None, backwards=False, nomagn=False, polrot=True, out_sht_mode='STANDARD'):
        """Adjoint remapping operation from lensed alm space to unlensed alm space

            Args:
                gclm: input gradient and possibly curl mode ((1 or 2, nalm)-shaped complex numpy.ndarray)
                mmax: set this for non-standard mmax != lmax in input array
                spin: spin-weight of the fields (larger or equal 0)
                lmax_out: desired output array lmax
                mmax_out: desired output array mmax (defaults to lmax_out if None)
                gclm_out(optional): output array (can be same as gclm provided it is large enough)
                backwards: forward or adjoint (not the same as inverse) lensing operation
                polrot(optional): includes small rotation of spin-weighted fields (defaults to True)
                out_sht_mode(optional): e.g. 'GRAD_ONLY' if only the output gradient mode is desired


            Note:
                 nomagn=True is a backward comptability thing to ask for inverse lensing


        """
        stri = 'lengclm ' + 'bwd' * backwards + 'fwd' * (not backwards)
        self.tim.start(stri)
        self.tim.reset()
        input_sht_mode = ducc_sht_mode(gclm, spin)
        if nomagn:
            assert backwards
        if mmax_out is None:
            mmax_out = lmax_out
        if self.sig_d <= 0 and np.abs(self.geom.fsky() - 1.) < 1e-6:
            # no actual deflection and single-precision full-sky
            ncomp_out = 1 + (spin != 0) * (out_sht_mode == 'STANDARD')
            if gclm_out is None:
                gclm_out = np.empty((ncomp_out, Alm.getsize(lmax_out, mmax_out)),  dtype=gclm.dtype)
            assert gclm_out.ndim == 2 and gclm_out.shape[0] == ncomp_out, (gclm_out.shape, ncomp_out)
            gclm_2d = np.atleast_2d(gclm)
            gclm_out[0] = alm_copy(gclm_2d[0], mmax, lmax_out, mmax_out)
            if ncomp_out > 1:
                gclm_out[1] = 0. if input_sht_mode == 'GRAD_ONLY' else alm_copy(gclm_2d[1], mmax, lmax_out, mmax_out)
            self.tim.close(stri)
            return gclm_out.squeeze()
        if not backwards:
            m = self.gclm2lenmap(gclm, mmax, spin, backwards, polrot=polrot)
            self.tim.reset()
            if gclm_out is not None:
                assert gclm_out.dtype == ctype[m.dtype], 'type precision must match'
            gclm_out = self.geom.adjoint_synthesis(m, spin, lmax_out, mmax_out, self.sht_tr, alm=gclm_out,
                                                   mode=out_sht_mode)
            return gclm_out.squeeze()
        else:
            if self.single_prec and gclm.dtype != np.complex64:
                gclm = gclm.astype(np.complex64)

                lmax_unl = Alm.getlmax(gclm.size, mmax)
                points = self.geom.synthesis(gclm, spin, lmax_unl, mmax, self.sht_tr, mode=input_sht_mode)
                self.tim.add('points synthesis (%s)'%input_sht_mode)
                if nomagn:
                    points *= self.dlm2A()

            assert points.ndim == 2 and not np.iscomplexobj(points)
            for ofs, w, nph in zip(self.geom.ofs, self.geom.weight, self.geom.nph):
                points[:, ofs:ofs + nph] *= w

            slm = self.lenmap2gclm(points, spin, lmax_out, mmax_out, sht_mode=out_sht_mode, gclm_out=gclm_out)
            self.tim.close(stri)

            return slm
    
        
    def adjoint_synthesis_general(self, gclm, dlm, lmax, mmax, spin, nthreads, polrot=True):
        """
            Note:
                points mst be already quadrature-weigthed

            Note:
                For inverse-lensing, need to feed in lensed maps times unlensed forward magnification matrix.

        """
        self.tim.start('lenmap2gclm')
        self.tim.reset()
        mode = ducc_sht_mode(gclm, spin)
        if spin == 0 and not np.iscomplexobj(points):
            points = points.astype(ctype[points.dtype]).squeeze()
        if spin > 0 and not np.iscomplexobj(points):
            points = (points[0] + 1j * points[1]).squeeze()
        ptg = self._get_ptgdlm, mmax()


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
            map_dfs = finufft.nufft2d1(x=ptg[:,0], y=ptg[:,1], c=map_dfs, n_modes=(lmax,lmax))
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
                            lmax=lmax, mmax=mmax, geometry="CC", nthreads=self.sht_tr, mode=mode, alm=gclm_out)
        self.tim.add('adjoint_synthesis_2d (%s)'%mode)
        self.tim.close('lenmap2gclm')
        return slm.squeeze()


class CPU_DUCCnufft_transformer(deflection):
    def __init__(self, shttransformer_desc):
        self.backend = 'CPU'
        self.single_prec = False
        self.verbosity = 1
        self.tim = timer(verbose=self.verbosity)
        self.sht_tr = 4
        self.planned = False
        self._cis = False
        self.cacher = cachers.cacher_mem()
        self.epsilon = 1e-7

        if shttransformer_desc == 'ducc':
            self.BaseClass = type('CPU_SHT_DUCC_transformer()', (CPU_SHT_DUCC_transformer,), {})
            self.instance = self.BaseClass()
        elif shttransformer_desc == 'shtns':
            self.BaseClass = type('CPU_SHT_SHTns_transformer()', (CPU_SHT_SHTns_transformer,), {})
            self.instance = self.BaseClass()
        else:
            raise ValueError('shttransformer_desc must be either "ducc" or "shtns"')

    def __getattr__(self, name):
        return getattr(self.instance, name)


    def set_nufftgeometry(self, geom_desc):
        self.nufftgeom = geometry.get_geom(geom_desc)
        self.set_geometry(geom_desc)


    def gclm2lenmap(self, gclm, dlm, lmax, mmax, spin, nthreads, polrot=True):
        """CPU algorithm for spin-n remapping using duccnufft

            Args:
                gclm: input alm array, shape (ncomp, nalm), where ncomp can be 1 (gradient-only) or 2 (gradient or curl)
                mmax: mmax parameter of alm array layout, if different from lmax
                spin: spin (>=0) of the transform
                backwards: forward or backward (adjoint) operation
        """ 
        s2_d = np.sum(alm2cl(dlm, dlm, lmax, mmax, lmax) * (2 * np.arange(lmax + 1) + 1)) / (4 * np.pi)
        sig_d = np.sqrt(s2_d / self.geom.fsky())
        sig_d_amin = sig_d / np.pi * 180 * 60
        if sig_d >= 0.01:
            print('deflection std is %.2e amin: this is really too high a value for something sensible'%sig_d_amin)
        elif self.verbosity:
            print('deflection std is %.2e amin' % sig_d_amin)
            
        gclm = np.atleast_2d(gclm)
        lmax_unl = Alm.getlmax(gclm[0].size, mmax)
        if mmax is None:
            mmax = lmax_unl
        if self.single_prec and gclm.dtype != np.complex64:
            gclm = gclm.astype(np.complex64)
            self.tim.add('type conversion')

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
            _ = map_dfs.copy()
            tmp = np.empty(map_dfs.shape, dtype=ctype[map_dfs.dtype])
            map_dfs = ducc0.fft.c2c(map_dfs.copy(), axes=(0, 1), inorm=2, nthreads=self.sht_tr, out=tmp)
            tmp_ = np.empty(map_dfs.T.shape, dtype=np.complex128)
            map_dfs_return = ducc0.fft.c2c(_.T, axes=(0, 1), inorm=2, nthreads=self.sht_tr, out=tmp_)
            del tmp
        else:
            map_dfs = ducc0.fft.c2c(map_dfs, axes=(0, 1), inorm=2, nthreads=self.sht_tr, out=map_dfs)

        if self.planned: # planned nufft
            assert ptg is None
            plan = self.make_plan(lmax_unl, spin)
            values = plan.u2nu(grid=map_dfs, forward=False, verbosity=self.verbosity)
            self.tim.add('planned u2nu')
        else:
            ptg = None
            if ptg is None:
                ptg = self._get_ptg(dlm, mmax)
            self.tim.add('get ptg')
            
            values = ducc0.nufft.u2nu(grid=map_dfs, coord=ptg, forward=True,
                                        epsilon=self.epsilon, nthreads=self.sht_tr,
                                        verbosity=self.verbosity, periodicity=2 * np.pi, fft_order=True)
            self.tim.add('u2nu')
            # return(map_dfs)

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
        return (values.real, ptg, map_dfs) if spin == 0 else values.view(rtype[values.dtype]).reshape((values.size, 2)).T


    def lenmap2gclm(self, points:np.ndarray[complex or float], spin:int, lmax:int, mmax:int, gclm_out=None, sht_mode='STANDARD'):
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
        ptg = self._get_ptg()

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
                                       epsilon=self.epsilon, nthreads=self.sht_tr, verbosity=self.verbosity,
                                       periodicity=2 * np.pi, fft_order=True)
        # go to position space
        map_dfs = ducc0.fft.c2c(map_dfs, axes=(0, 1), forward=False, inorm=2, nthreads=self.sht_tr, out=map_dfs)

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
        slm = ducc0.sht.experimental.adjoint_synthesis_2d(map=map, spin=spin, lmax=lmax, mmax=mmax, geometry="CC", nthreads=self.sht_tr, mode=sht_mode, alm=gclm_out)
        return slm.squeeze()
    
    
    def lensgclm(self, gclm:np.ndarray, mmax:int or None, spin:int, lmax_out:int, mmax_out:int or None,
                 gclm_out:np.ndarray=None, backwards=False, nomagn=False, polrot=True, out_sht_mode='STANDARD'):
        """Adjoint remapping operation from lensed alm space to unlensed alm space

            Args:
                gclm: input gradient and possibly curl mode ((1 or 2, nalm)-shaped complex numpy.ndarray)
                mmax: set this for non-standard mmax != lmax in input array
                spin: spin-weight of the fields (larger or equal 0)
                lmax_out: desired output array lmax
                mmax_out: desired output array mmax (defaults to lmax_out if None)
                gclm_out(optional): output array (can be same as gclm provided it is large enough)
                backwards: forward or adjoint (not the same as inverse) lensing operation
                polrot(optional): includes small rotation of spin-weighted fields (defaults to True)
                out_sht_mode(optional): e.g. 'GRAD_ONLY' if only the output gradient mode is desired


            Note:
                 nomagn=True is a backward comptability thing to ask for inverse lensing


        """
        stri = 'lengclm ' + 'bwd' * backwards + 'fwd' * (not backwards)
        self.tim.start(stri)
        self.tim.reset()
        input_sht_mode = ducc_sht_mode(gclm, spin)
        if nomagn:
            assert backwards
        if mmax_out is None:
            mmax_out = lmax_out
        if self.sig_d <= 0 and np.abs(self.geom.fsky() - 1.) < 1e-6:
            # no actual deflection and single-precision full-sky
            ncomp_out = 1 + (spin != 0) * (out_sht_mode == 'STANDARD')
            if gclm_out is None:
                gclm_out = np.empty((ncomp_out, Alm.getsize(lmax_out, mmax_out)),  dtype=gclm.dtype)
            assert gclm_out.ndim == 2 and gclm_out.shape[0] == ncomp_out, (gclm_out.shape, ncomp_out)
            gclm_2d = np.atleast_2d(gclm)
            gclm_out[0] = alm_copy(gclm_2d[0], mmax, lmax_out, mmax_out)
            if ncomp_out > 1:
                gclm_out[1] = 0. if input_sht_mode == 'GRAD_ONLY' else alm_copy(gclm_2d[1], mmax, lmax_out, mmax_out)
            self.tim.close(stri)
            return gclm_out.squeeze()
        if not backwards:
            m = self.gclm2lenmap(gclm, mmax, spin, backwards, polrot=polrot)
            self.tim.reset()
            if gclm_out is not None:
                assert gclm_out.dtype == ctype[m.dtype], 'type precision must match'
            gclm_out = self.geom.adjoint_synthesis(m, spin, lmax_out, mmax_out, self.sht_tr, alm=gclm_out,
                                                   mode=out_sht_mode)
            return gclm_out.squeeze()
        else:
            if self.single_prec and gclm.dtype != np.complex64:
                gclm = gclm.astype(np.complex64)

                lmax_unl = Alm.getlmax(gclm.size, mmax)
                points = self.geom.synthesis(gclm, spin, lmax_unl, mmax, self.sht_tr, mode=input_sht_mode)
                self.tim.add('points synthesis (%s)'%input_sht_mode)
                if nomagn:
                    points *= self.dlm2A()

            assert points.ndim == 2 and not np.iscomplexobj(points)
            for ofs, w, nph in zip(self.geom.ofs, self.geom.weight, self.geom.nph):
                points[:, ofs:ofs + nph] *= w

            slm = self.lenmap2gclm(points, spin, lmax_out, mmax_out, sht_mode=out_sht_mode, gclm_out=gclm_out)
            self.tim.close(stri)

            return slm
    
        
    def adjoint_synthesis_general(gclm, dlm, lmax, mmax, spin, nthreads, polrot=True):

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
        ptg = self._get_ptgdlm, mmax()


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


    def alm2lenmap_spin(gclm: np.ndarray or list, dlms:np.ndarray or list, spin:int, geometry: tuple[str, dict] = ('healpix', {'nside':2048}), epsilon: float=1e-7, verbose=0, nthreads: int=0):
        r"""Computes a deflected spin-weight lensed CMB map from its gradient and curl modes and deflection field alm.

            Args:
                gclm:  undeflected map healpy gradient (and curl, if relevant) modes
                        (e.g. polarization Elm and Blm).

                dlms: The spin-1 deflection, in the form of one or two arrays.

                        The two arrays are the gradient and curl deflection healpy alms:

                        :math:`\sqrt{L(L+1)}\phi_{LM}` with :math:`\phi` the lensing potential

                        :math:`\sqrt{L(L+1)}\Omega_{LM}` with :math:`\Omega` the lensing curl potential


                        The curl can be omitted if zero, resulting in principle in slightly faster transforms


                spin(int >= 0): spin-weight of the maps to deflect (e.g. 2 for polarization).
                geometry(optional): sphere pixelization, tuple with geometry name and argument dictionary,
                                    defaults to Healpix with nside 2048
                epsilon(optional): target accuracy of the result (defaults to 1e-7)
                verbose(optional): If set, prints a bunch of timing and other info. Defaults to 0.
                nthreads(optional): number of threads to use (defaults to os.cpu_count())


            Returns:
                lensed maps for input geometry (real and imaginary parts),
                arrays of size given by the number of pixels of input geometry

            Note:

                If curl modes are zero (deflection and/or alm's to lens), they can be omitted, which can result in slightly faster transforms


        """
        if spin == 0:
            return alm2lenmap(gclm, dlms, geometry=geometry, epsilon=epsilon, verbose=verbose, nthreads=nthreads)
        if isinstance(dlms, list) or dlms.ndim > 1:
            assert len(dlms) <= 2
            dglm = dlms[0]
            dclm = None if len(dlms) == 1 else dlms[1]
        else:
            dglm = dlms
            dclm = None

        if nthreads <= 0:
            nthreads = cpu_count()
            if verbose:
                print('alm2lenmap_spin: using %s nthreads'%nthreads)

        defl = deflection(get_geom(geometry), dglm, None, dclm=dclm, epsilon=epsilon, numthreads=nthreads, verbosity=0,
                        cacher=cachers.cacher_mem(safe=False))
        if isinstance(gclm, list) and gclm[1] is None:
            gclm = gclm[0]
        ret = defl.gclm2lenmap(gclm, None, spin, False)
        if verbose:
            print(defl.tim)
        return ret   

# TODO this could just be lenspyx, then becomes LENSPYX_transformer?
class CPU_DUCC_transformer(deflection):
    def __init__(self, shttransformer_desc):
        self.backend = 'CPU'

        if shttransformer_desc == 'ducc':
            self.BaseClass = type('CPU_SHT_DUCC_transformer()', (CPU_SHT_DUCC_transformer,), {})
            self.instance = self.BaseClass()
        elif shttransformer_desc == 'shtns':
            self.BaseClass = type('CPU_SHT_SHTns_transformer()', (CPU_SHT_SHTns_transformer,), {})
            self.instance = self.BaseClass()
        else:
            raise ValueError('shttransformer_desc must be either "ducc" or "shtns"')

    def __getattr__(self, name):
        return getattr(self.instance, name)


    def gclm2lenmap(self, gclm:np.ndarray, dlm, mmax:int or None, spin:int, backwards:bool, polrot=True, ptg=None, epsilon=1e-8, single_prec=True, dclm=None):
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
