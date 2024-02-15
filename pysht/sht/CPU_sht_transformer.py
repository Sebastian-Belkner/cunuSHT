import numpy as np

from ducc0.sht.experimental import (
    synthesis as ducc_synthesis,
    adjoint_synthesis as ducc_adjoint_synthesis
)

import shtns

import pysht.geometry as geometry

class CPU_SHT_DUCC_transformer():
    def __init__(self, geom=None):
        pass
        # self.geom = geometry.get_geom(geom)


    def set_geometry(self, geom_desc):
        self.geom = geometry.get_geom(geom_desc)


    def synthesis(self, gclm: np.ndarray, **kwargs):
        """Wrapper to ducc forward SHT
            Return a map or a pair of map for spin non-zero, with the same type as gclm
        """
        
        # signature: (alm, theta, lmax, mmax, nphi, spin, phi0, nthreads, ringstart, map, **kwargs)
        gclm = np.atleast_2d(gclm)
        return ducc_synthesis(alm=gclm, theta=self.geom.theta, nphi=self.geom.nph, phi0=self.geom.phi0, ringstart=self.geom.ofs, **kwargs)


class CPU_SHT_SHTns_transformer():
    def __init__(self, geom_desc=None):
        pass
        # self.geom = geometry.get_geom(geom_desc)


    def set_geometry(self, geominfo):
        # TODO set_geometry is more a constructor + set_grid in shtns
        # self.geom = geometry.get_geom(geom_desc)
        self.constructor = shtns.sht(int(geominfo[1]['lmax']), int(geominfo[1]['mmax']))
        self.constructor.set_grid(flags=shtns.SHT_THETA_CONTIGUOUS)
        
        
    def set_constructor(self, lmax, mmax):
        assert 0, "implement if needed"
        self.constructor = shtns.sht(int(lmax), int(mmax))
        self.constructor.set_grid(flags=shtns.SHT_THETA_CONTIGUOUS)


    def synthesis(self, gclm: np.ndarray, **kwargs):
        """Wrapper to SHTns forward SHT
            Return a map or a pair of map for spin non-zero, with the same type as gclm
        """

        gclm = np.atleast_2d(gclm)
        return self.constructor.synth(gclm)
    
    
class CPU_SHT_pySHT_transformer():
    """
    CPU_SHT_pySHT_transformer class for performing spherical harmonic transformations using pySHT library.
    """

    def __init__(self, geom_desc=None):
        pass
        # self.geom = geometry.get_geom(geom_desc)


    def set_geometry(self, geominfo):
        # TODO set_geometry is more a constructor + set_grid in shtns
        self.geom = geometry.get_geom(geom_desc)
        self.constructor = shtns.sht(int(geominfo[1]['lmax']), int(geominfo[1]['mmax']))
        self.constructor.set_grid(flags=shtns.SHT_THETA_CONTIGUOUS)
        
        
    def set_constructor(self, lmax, mmax):
        assert 0, "implement if needed"
        self.constructor = shtns.sht(int(lmax), int(mmax))
        self.constructor.set_grid(flags=shtns.SHT_THETA_CONTIGUOUS)


    def synthesis(self, gclm: np.ndarray, **kwargs):
        """Wrapper to SHTns forward SHT
            Return a map or a pair of map for spin non-zero, with the same type as gclm
        """

        gclm = np.atleast_2d(gclm)
        return self.constructor.synth(gclm)