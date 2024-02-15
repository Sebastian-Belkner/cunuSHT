import numpy as np

from ducc0.sht.experimental import (
    synthesis as ducc_synthesis,
    adjoint_synthesis as ducc_adjoint_synthesis
)

import shtns

import pysht.geometry as geometry


class GPU_SHTns_transformer():
    def __init__(self, lmax, mmax, geom=None):
        self.geom = geom
        self.constructor = shtns.sht(int(lmax), int(mmax))
        self.constructor.set_grid(flags=shtns.SHT_ALLOW_GPU + shtns.SHT_THETA_CONTIGUOUS)

    def set_geometry(self, geominfo):
        # TODO set_geometry is more a constructor + set_grid in shtns
        # self.geom = geometry.get_geom(geom_desc)
        self.constructor = shtns.sht(int(geominfo[1]['lmax']), int(geominfo[1]['mmax']))
        self.constructor.set_grid(flags=shtns.SHT_ALLOW_GPU + shtns.SHT_THETA_CONTIGUOUS)        
        
    def set_constructor(self, lmax, mmax):
        assert 0, "implement if needed"
        self.constructor = shtns.sht(int(lmax), int(mmax))
        self.constructor.set_grid(flags=shtns.SHT_ALLOW_GPU + shtns.SHT_THETA_CONTIGUOUS)


    def synthesis(self, gclm: np.ndarray, **kwargs):
        """Wrapper to SHTns forward SHT
            Return a map or a pair of map for spin non-zero, with the same type as gclm
        """

        gclm = np.atleast_2d(gclm)
        return self.constructor.synth(alm=gclm, theta=self.geom.theta, nphi=self.geom.nph, phi0=self.geom.phi0, ringstart=self.geom.ofs, **kwargs)