"""
TBD

python3 -m unittest test_accuracy.py
"""
import unittest

import numpy as np
import time
import healpy as hp
import pysht
import sys
from time import process_time
from delensalot.sims.sims_lib import Xunl, Xsky
import cupy as cp

test_cases = [ 
    (lmax, lmax) for lmax in [2**n-1 for n in np.arange(6, 8)]
    ]

runinfos = [
    ("CPU", "lenspyx", 'ducc'),
    # ("CPU", "duccnufft", 'ducc'),
    # ("GPU", "cufinufft", 'shtns')
    ]
epsilons = [1e-6]
# lmaxs = [256*n-1 for n in np.arange(int(sys.argv[1]), 24)]
lmaxs = [256*int(sys.argv[1])-1]
phi_lmaxs = [lmax for lmax in lmaxs]
defres = {}
Tsky = None
Tsky2 = None
        
class Test(unittest.TestCase):

    @unittest.skip("Skipping this test method for now")
    def test_gclm2lenmap_accuracy(self):
        for test_case in test_cases:
            with self.subTest(input_value=test_case):
                lmaxs = [1023]
                phi_lmaxs = [lmax for lmax in lmaxs]
                defres = {}
                Tsky = None
                Tsky2 = None
                for lmax, phi_lmax in zip(lmaxs, phi_lmaxs):
                    geominfo = ('gl',{'lmax':lmax})
                    lenjob_geominfo = ('gl',{'lmax':phi_lmax})
                    lldlm = np.arange(0,phi_lmax+1)
                    synunl = Xunl(lmax=lmax, geominfo=geominfo, phi_lmax=phi_lmax)
                    synsky = Xsky(lmax=lmax, unl_lib=synunl, geominfo=geominfo, lenjob_geominfo=geominfo, epsilon=1e-10)
                    philm = synunl.get_sim_phi(0, space='alm')
                    toydlm = hp.almxfl(philm, np.sqrt(lldlm*(lldlm+1)))
                    toyunllm = synunl.get_sim_unl(0, spin=0, space='alm', field='temperature')
                    Tsky = synsky.get_sim_sky(0, spin=0, space='map', field='temperature')
                    Tsky2 = synsky.unl2len(toyunllm, philm, spin=0)
                    for runinfo in runinfos:
                        kwargs = {
                            'geominfo': lenjob_geominfo,
                            'planned': False,
                            'single_prec': False,
                        } 
                        print(runinfo)
                        backend = runinfo[0]
                        defres.update({backend: {}}) if backend not in defres.keys() else None
                        solver = runinfo[1]
                        defres[backend].update({solver : None}) if solver not in defres[backend].keys() else None
                        for mode in ['nuFFT']:
                            print("\nTesting:: solver = {} backend = {} mode = {} ...".format(solver, backend, mode))
                            t = pysht.get_transformer(solver, mode, backend)
                            t = t(**kwargs)
                            
                            nrings = len(t.geom.ofs)
                            pixs = t.geom.ofs[nrings//4*2:nrings//4*2+10]
                            Tsky_bruteforce = t.gclm2lenpixs(toydlm, lmax, spin=0, pixs=pixs)
                            if backend == 'CPU':
                                if solver == 'lenspyx':
                                    defres[backend][solver] = t.gclm2lenmap(
                                            toyunllm.copy(), dlm=toydlm, lmax=lmax, mmax=lmax, spin=0, nthreads=10, execmode='normal', ptg=None)
                                else:
                                    defres[backend][solver] = t.gclm2lenmap(
                                            toyunllm.copy(), dlm=toydlm, lmax=lmax, mmax=lmax, spin=0, nthreads=10, execmode='normal', ptg=None)
                            elif backend == 'GPU':
                                lenmap = cp.empty(t.constructor.spat_shape, dtype=cp.complex128)
                                ll = np.arange(0,deflection_kwargs["mmax_dlm"]+1,1)
                                dlm_scaled = hp.almxfl(toydlm, np.nan_to_num(np.sqrt(1/(ll*(ll+1)))))
                                dlm_scaled = cp.array(np.atleast_2d(dlm_scaled), dtype=np.complex128) if not deflection_kwargs["single_prec"] else cp.array(np.atleast_2d(dlm_scaled).astype(np.complex64))
                                defres[backend][solver] = t.gclm2lenmap(cp.array(toyunllm.copy()), dlm_scaled=dlm_scaled, lmax=lmax, mmax=lmax, nthreads=10, lenmap=lenmap, execmode='normal')
                        # Only take pixels of interest:
                        defres[backend][solver]
    @unittest.skip("Skipping this test method for now")
    def test_lenmap2gclm_accuracy(self):
        for test_case in test_cases:
            with self.subTest(input_value=test_case):
                pass
         

class TestIntegration(unittest.TestCase):
    
    @unittest.skip("Skipping this test method for now")
    def test_XXX(self):
        for test_case in test_cases:
            with self.subTest(input_value=test_case):
                pass

    @unittest.skip("Skipping this test method for now")
    def test_XXX(self):
        for test_case in test_cases:
            with self.subTest(input_value=test_case):
                pass

if __name__ == '__main__':
    unittest.main()