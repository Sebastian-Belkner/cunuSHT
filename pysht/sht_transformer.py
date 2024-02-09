import numpy as np

from ducc0.sht.experimental import (
    synthesis as ducc_synthesis,
    adjoint_synthesis as ducc_adjoint_synthesis
)

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
