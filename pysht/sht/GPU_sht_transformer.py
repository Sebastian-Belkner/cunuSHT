import numpy as np

import shtns
import pysht.geometry as geometry


class GPU_SHTns_transformer():
    
    def __init__(self):
        pass


    def set_geometry(self, geominfo):
        #TODO perhaps namechange: set_geometry is more a constructor + set_grid in shtns
        #TODO get geom from SHTns
        self.geom = geometry.get_geom(geominfo)
        import copy
        self.geominfo = copy.deepcopy(geominfo)
        #TODO add mmax to geominfo at userlevel
        if 'mmax' not in geominfo[1]:
            geominfo[1].update({'mmax': int(geominfo[1]['lmax'])})
        self.constructor = shtns.sht(int(geominfo[1]['lmax']), int(geominfo[1]['mmax']))
        self.constructor.set_grid(flags=shtns.SHT_ALLOW_GPU + shtns.SHT_THETA_CONTIGUOUS)

        
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


    def analysis(self, gclm: np.ndarray, spin, lmax, mmax, mode=None, nthreads=None):
        #TODO all other than gclm not supported. Want same interface for each backend, 
        # could check grid for each synth and ana call and update if needed
        """Wrapper to SHTns forward SHT
            Return a map or a pair of map for spin non-zero, with the same type as gclm
        """
        return np.atleast_2d(self.constructor.analys(alm=gclm).flatten())


    def map2alm(self, m: np.ndarray, **kwargs):
        return self.synthesis(m, **kwargs)
    
    
    def alm2map(self, gclm: np.ndarray, **kwargs):
        return self.analysis(gclm, **kwargs)


class GPU_SHT_pySHT_transformer():
    """
    GPU_SHT_pySHT_transformer class for performing spherical harmonic transformations using pySHT library.
    This will be the self-implemented spin-n SHT transforms. 
    """
    def __init__(self, geom_desc=None):
        pass


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