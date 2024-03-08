import numpy as np

import shtns
import pysht.geometry as geometry


class GPU_SHTns_transformer():
    
    def __init__(self, geominfo):
        self.geom = geometry.get_geom(geominfo)
        self.set_geometry(geominfo)


    def set_geometry(self, geominfo):
        # TODO set_geometry is more a constructor + set_grid in shtns
        # self.geom = geometry.get_geom(geom_desc)
        self.geom = geometry.get_geom(geominfo)
        if geominfo[0] == 'cc':
            self.constructor = shtns.sht(int(geominfo[1]['ntheta']-1), int(geominfo[1]['ntheta']-1))
            self.constructor.set_grid(flags=shtns.SHT_ALLOW_GPU + shtns.SHT_THETA_CONTIGUOUS, nlat=int(geominfo[1]['ntheta']), nphi=int(geominfo[1]['nphi']))
        else:
            self.constructor = shtns.sht(int(geominfo[1]['lmax']), int(geominfo[1]['lmax']))
            self.constructor.set_grid(flags=shtns.SHT_ALLOW_GPU + shtns.SHT_THETA_CONTIGUOUS, nlat=len(self.geom.nph), nphi=int(self.geom.nph[0]))
        

        
    def set_constructor(self, lmax, mmax):
        assert 0, "implement if needed"
        self.constructor = shtns.sht(int(lmax), int(mmax))
        self.constructor.set_grid(flags=shtns.SHT_ALLOW_GPU + shtns.SHT_THETA_CONTIGUOUS)


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
        return np.atleast_2d(self.constructor.analys(map).flatten())


    def adjoint_synthesis(self, map: np.ndarray, **kwargs):
        #TODO all other than gclm not supported. Want same interface for each backend, 
        # could check grid for each synth and ana call and update if needed
        """Wrapper to SHTns forward SHT
            Return a map or a pair of map for spin non-zero, with the same type as gclm
        """
        def check_dim():
            if len(np.shape(map)) == 1:
                return map.reshape(*self.constructor.spat_shape)
            return map
        map = check_dim()
        return self.analysis(map, **kwargs)


    def map2alm(self, m: np.ndarray, **kwargs):
        return self.analysis(m, **kwargs)
    
    
    def alm2map(self, gclm: np.ndarray, **kwargs):
        return self.synthesis(gclm, **kwargs)


class GPU_SHT_pySHT_transformer():
    """
    GPU_SHT_pySHT_transformer class for performing spherical harmonic transformations using pySHT library.
    This will be the self-implemented spin-n SHT transforms. 
    """
    def __init__(self, geominfo):
        self.geom = geometry.get_geom(geominfo)


    def set_geometry(self, geominfo):
        pass
   
        
    def set_constructor(self, lmax, mmax):
        assert 0, "implement if needed"


    def synthesis(self, gclm: np.ndarray, **kwargs):
        """Wrapper to SHTns forward SHT
            Return a map or a pair of map for spin non-zero, with the same type as gclm
        """
        assert 0, "implement if needed"
        # TODO here goes the assocLeg.cu implementation


    def analysis(self, gclm: np.ndarray, **kwargs):
        """Wrapper to SHTns forward SHT
            Return a map or a pair of map for spin non-zero, with the same type as gclm
        """
        assert 0, "implement if needed"
        # TODO here goes the assocLeg.cu implementation


    def map2alm(self, m: np.ndarray, **kwargs):
        return self.synthesis(m, **kwargs)
    
    
    def alm2map(self, gclm: np.ndarray, **kwargs):
        return self.analysis(gclm, **kwargs)