import numpy as np

from ducc0.sht.experimental import (
    synthesis as ducc_synthesis,
    adjoint_synthesis as ducc_adjoint_synthesis
)

import shtns

class CPU_DUCC_transformer():
    def __init__(self, geom=None):
        self.geom = geom


    def set_geometry(self, geom):
        self.geom = geom


    def synthesis(self, gclm: np.ndarray, **kwargs):
        """Wrapper to ducc forward SHT
            Return a map or a pair of map for spin non-zero, with the same type as gclm
        """
        
        # signature: (alm, theta, lmax, mmax, nphi, spin, phi0, nthreads, ringstart, map, **kwargs)
        gclm = np.atleast_2d(gclm)
        return ducc_synthesis(alm=gclm, theta=self.geom.theta, nphi=self.geom.nph, phi0=self.geom.phi0, ringstart=self.geom.ofs, **kwargs)


class CPU_SHTns_transformer():
    def __init__(self, lmax, mmax, geom=None):
        self.geom = geom
        self.constructor = shtns.sht(int(lmax), int(mmax))
        self.constructor.set_grid(flags=shtns.SHT_ALLOW_GPU + shtns.SHT_THETA_CONTIGUOUS)


    def set_geometry(self, geom):
        self.geom = geom


    def synthesis(self, gclm: np.ndarray, **kwargs):
        """Wrapper to SHTns forward SHT
            Return a map or a pair of map for spin non-zero, with the same type as gclm
        """

        gclm = np.atleast_2d(gclm)
        return self.constructor.synth(alm=gclm, theta=self.geom.theta, nphi=self.geom.nph, phi0=self.geom.phi0, ringstart=self.geom.ofs, **kwargs)


class GPU_SHTns_transformer():
    def __init__(self, lmax, mmax, geom=None):
        self.geom = geom
        self.constructor = shtns.sht(int(lmax), int(mmax))
        self.constructor.set_grid(flags=shtns.SHT_ALLOW_GPU + shtns.SHT_THETA_CONTIGUOUS)


    def set_geometry(self, geom):
        self.geom = geom


    def synthesis(self, gclm: np.ndarray, **kwargs):
        """Wrapper to SHTns forward SHT
            Return a map or a pair of map for spin non-zero, with the same type as gclm
        """

        gclm = np.atleast_2d(gclm)
        return self.constructor.synth(alm=gclm, theta=self.geom.theta, nphi=self.geom.nph, phi0=self.geom.phi0, ringstart=self.geom.ofs, **kwargs)