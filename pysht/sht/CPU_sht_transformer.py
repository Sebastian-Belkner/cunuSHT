import numpy as np
import healpy as hp

from ducc0.sht.experimental import (
    synthesis as ducc_synthesis,
    adjoint_synthesis as ducc_adjoint_synthesis
)
import lenspyx.utils_hp as utils_hp
import shtns

import pysht.geometry as geometry

class CPU_SHT_DUCC_transformer():
    def __init__(self, geom=None):
        self.geom = geometry.get_geom(geom)


    def set_geometry(self, geom_desc):
        self.geom = geometry.get_geom(geom_desc)


    def synthesis(self, gclm: np.ndarray, spin, lmax, mmax, nthreads, map:np.ndarray=None, **kwargs):
        """Wrapper to ducc forward SHT
            Return a map or a pair of map for spin non-zero, with the same type as gclm
        """
        # signature: (alm, theta, lmax, mmax, nphi, spin, phi0, nthreads, ringstart, map, **kwargs)
        gclm = np.atleast_2d(gclm)
        return ducc_synthesis(alm=gclm, theta=self.geom.theta, lmax=lmax, mmax=mmax, nphi=self.geom.nph, spin=spin, phi0=self.geom.phi0,
                         nthreads=nthreads, ringstart=self.geom.ofs, map=map, **kwargs)

    def adjoint_synthesis(self, m: np.ndarray, spin:int, lmax:int, mmax:int, nthreads:int, alm=None, apply_weights=True, **kwargs):
        """Wrapper to ducc backward SHT

            Return an array with leading dimension 1 for spin-0 or 2 for spin non-zero

            Note:
                This modifies the input map

        """
        m = np.atleast_2d(m)
        if apply_weights:
            for of, w, npi in zip(self.geom.ofs, self.geom.weight, self.geom.nph):
                m[:, of:of + npi] *= w
        if alm is not None:
            assert alm.shape[-1] == utils_hp.Alm.getsize(lmax, mmax)
        # print("map={map}, theta={theta}, nphi={nphi}, phi0={phi0}, nthreads={nthreads}, ringstart={ringstart}, alm={alm}".format(map=m.shape, theta=self.geom.theta.shape, nphi=self.geom.nph.shape, phi0=self.geom.phi0.shape,
                                #  nthreads=nthreads, ringstart=self.geom.ofs, alm=alm) )
        return ducc_adjoint_synthesis(map=m, theta=self.geom.theta, lmax=lmax, mmax=mmax, nphi=self.geom.nph, spin=spin, phi0=self.geom.phi0,
                                 nthreads=nthreads, alm=alm, ringstart=self.geom.ofs, **kwargs)
        
    def alm2map_spin(self, gclm:np.ndarray, spin:int, lmax:int, mmax:int, nthreads:int, zbounds=(-1., 1.), **kwargs):
        # FIXME: method only here for backwards compatiblity
        assert zbounds[0] == -1 and zbounds[1] == 1., zbounds
        return self.synthesis(gclm, spin, lmax, mmax, nthreads, **kwargs)

    def map2alm_spin(self, m:np.ndarray, spin:int, lmax:int, mmax:int, nthreads:int, zbounds=(-1., 1.), **kwargs):
        # FIXME: method only here for backwards compatiblity
        assert zbounds[0] == -1 and zbounds[1] == 1., zbounds
        return self.adjoint_synthesis(m.copy(), spin, lmax, mmax, nthreads, **kwargs)

    def alm2map(self, gclm:np.ndarray, lmax:int, mmax:int, nthreads:int, zbounds=(-1., 1.), **kwargs):
        # FIXME: method only here for backwards compatiblity
        assert zbounds[0] == -1 and zbounds[1] == 1., zbounds
        return self.synthesis(gclm, 0, lmax, mmax, nthreads, **kwargs).squeeze()

    def map2alm(self, m:np.ndarray, lmax:int, mmax:int, nthreads:int, zbounds=(-1., 1.), **kwargs):
        # FIXME: method only here for backwards compatiblity
        assert zbounds[0] == -1 and zbounds[1] == 1., zbounds
        return self.adjoint_synthesis(m.copy(), 0, lmax, mmax, nthreads, **kwargs).squeeze()

class CPU_SHT_SHTns_transformer():
    def __init__(self, geominfo):
        self.geom = geometry.get_geom(geominfo)
        self.set_geometry(geominfo)


    def set_geometry(self, geominfo):
        # TODO set_geometry is more a constructor + set_grid in shtns
        # self.geom = geometry.get_geom(geom_desc)
        self.geom = geometry.get_geom(geominfo)
        if geominfo[0] == 'cc':
            self.constructor = shtns.sht(int(geominfo[1]['ntheta']-1), int(geominfo[1]['ntheta']-1))
            self.constructor.set_grid(flags=shtns.SHT_THETA_CONTIGUOUS, nlat=int(geominfo[1]['ntheta']), nphi=int(geominfo[1]['nphi']))
        else:
            self.constructor = shtns.sht(int(geominfo[1]['lmax']), int(geominfo[1]['lmax']))
            self.constructor.set_grid(flags=shtns.SHT_THETA_CONTIGUOUS, nlat=len(self.geom.nph), nphi=int(self.geom.nph[0]))
           
        
    def set_constructor(self, lmax, mmax):
        assert 0, "implement if needed"
        self.constructor = shtns.sht(int(lmax), int(mmax))
        self.constructor.set_grid(flags=shtns.SHT_THETA_CONTIGUOUS)


    def synthesis(self, gclm: np.ndarray, spin, lmax, mmax, mode=None, nthreads=None):
        #TODO all other than gclm not supported. Want same interface for each backend, 
        # could check grid for each synth and ana call and update if needed
        """Wrapper to SHTns forward SHT
            Return a map or a pair of map for spin non-zero, with the same type as gclm
        """
        gclm = np.atleast_2d(gclm)
        return np.atleast_2d(self.constructor.synth(gclm).flatten())


    def synthesis_der1(self, gclm: np.ndarray, nthreads=None):
        #TODO all other than gclm not supported. Want same interface for each backend, 
        # could check grid for each synth and ana call and update if needed
        """Wrapper to SHTns forward SHT
            Return a map or a pair of map for spin non-zero, with the same type as gclm
        """
        # gclm = np.atleast_2d(gclm)
        buff = self.constructor.synth_grad(gclm)
        ret = np.array([a.flatten() for a in buff])
        return ret


    def analysis(self, map: np.ndarray, spin=None, lmax=None, mmax=None, nthreads=None, alm=None, mode=None):
        #TODO all other than gclm not supported. Want same interface for each backend, 
        # could check grid for each synth and ana call and update if needed
        """Wrapper to SHTns forward SHT
            Return a map or a pair of map for spin non-zero, with the same type as gclm
        """
        if len(np.shape(map)) == 1:
            map = map.reshape(*self.constructor.spat_shape)
        return np.atleast_2d(self.constructor.analys(map).flatten())
    

    def adjoint_synthesis(self, map: np.ndarray, **kwargs):
        #TODO all other than gclm not supported. Want same interface for each backend, 
        # could check grid for each synth and ana call and update if needed
        """Wrapper to SHTns forward SHT
            Return a map or a pair of map for spin non-zero, with the same type as gclm
        """
        return self.analysis(map, **kwargs)

    def alm2map_spin(self, gclm:np.ndarray, spin:int, lmax:int, mmax:int, nthreads:int, zbounds=(-1., 1.), **kwargs):
        # FIXME: method only here for backwards compatiblity
        assert zbounds[0] == -1 and zbounds[1] == 1., zbounds
        return self.synthesis(gclm, spin, lmax, mmax, nthreads, **kwargs)

    def map2alm_spin(self, m:np.ndarray, spin:int, lmax:int, mmax:int, nthreads:int, zbounds=(-1., 1.), **kwargs):
        # FIXME: method only here for backwards compatiblity
        assert zbounds[0] == -1 and zbounds[1] == 1., zbounds
        return self.adjoint_synthesis(m.copy(), spin, lmax, mmax, nthreads, **kwargs)

    def alm2map(self, gclm:np.ndarray, lmax:int, mmax:int, nthreads:int, zbounds=(-1., 1.), **kwargs):
        # FIXME: method only here for backwards compatiblity
        assert zbounds[0] == -1 and zbounds[1] == 1., zbounds
        return self.synthesis(gclm, 0, lmax, mmax, nthreads, **kwargs).squeeze()

    def map2alm(self, m:np.ndarray, lmax:int, mmax:int, nthreads:int, zbounds=(-1., 1.), **kwargs):
        # FIXME: method only here for backwards compatiblity
        assert zbounds[0] == -1 and zbounds[1] == 1., zbounds
        return self.adjoint_synthesis(m.copy(), 0, lmax, mmax, nthreads, **kwargs).squeeze()