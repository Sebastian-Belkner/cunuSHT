"""
This test script is for testing the dlm2pointing().

python3 -m unittest test_class_deflection.py
"""
import unittest

import numpy as np
import cupy as cp
import shtns

"""
Debug gclm2lenmap
"""
import numpy as np
import time
import healpy as hp
import matplotlib.pyplot as plt
import pysht
import sys
from time import process_time

from delensalot.sims.sims_lib import Xunl, Xsky


test_cases = [ 
    (lmax, lmax) for lmax in [2**n-1 for n in np.arange(6, 10)]
]
        
class TestUnit(unittest.TestCase):

    def test_unit_dlm2pointing(self):
        for lmax, phi_lmax in test_cases:
            with self.subTest(input_value=(lmax, phi_lmax)):
                geominfo = ('gl',{'lmax':lmax})
                lenjob_geominfo = ('gl',{'lmax':phi_lmax})
                
                lldlm = np.arange(0,phi_lmax+1)
                synunl = Xunl(lmax=lmax, geominfo=geominfo, phi_lmax=phi_lmax)
                synsky = Xsky(lmax=lmax, unl_lib=synunl, geominfo=geominfo, lenjob_geominfo=geominfo)
                philm = synunl.get_sim_phi(0, space='alm')
                toydlm = hp.almxfl(philm, np.sqrt(lldlm*(lldlm+1)))
                # toyunllm = synunl.get_sim_unl(0, spin=0, space='alm', field='temperature')
                # nlm = hp.Alm.getsize(phi_lmax)
                # toydlm = np.random.randn(nlm)*1e-6 + 1j*np.random.randn(nlm)*1e-6

                kwargs = {
                    'geominfo': geominfo,
                    'nthreads': 10,
                    'epsilon': 1e-10,
                    'verbosity': 0,
                    'planned': False,
                    'single_prec': False,
                    'shttransformer_desc': 'shtns'
                }

                deflection_kwargs = {
                    'dlm': toydlm,
                    'mmax_dlm': phi_lmax,
                    'epsilon': 1e-10,
                    'verbosity': 0,  
                    'single_prec': False,
                    'nthreads': 10,
                    'geominfo': lenjob_geominfo,
                }

                tGPU = pysht.get_transformer(solver='cufinufft', mode='nuFFT', backend='GPU')(**kwargs, deflection_kwargs=deflection_kwargs)
                pointing_theta = cp.zeros((tGPU.deflectionlib.geom.npix()), dtype=cp.double)
                pointing_phi = cp.zeros((tGPU.deflectionlib.geom.npix()), dtype=cp.double)
                tGPU.deflectionlib.dlm2pointing(toydlm, pointing_theta, pointing_phi)

                kwargs = {
                    'geominfo': geominfo,
                    'nthreads': 10,
                    'epsilon': 1e-10,
                    'verbosity': 0,
                    'planned': False,
                    'single_prec': False,
                    'shttransformer_desc': 'ducc'
                }
                tCPU = pysht.get_transformer(solver='duccnufft', mode='nuFFT', backend='CPU')(**kwargs, deflection_kwargs=deflection_kwargs)   
                pt_CPU, pp_CPU = tCPU.deflectionlib.dlm2pointing(toydlm)
                
                res_theta = np.std(pt_CPU - pointing_theta.get())
                res_phi = np.std(pp_CPU - pointing_phi.get())
                
                # import matplotlib.pyplot as plt
                # plt.imshow((pp_CPU - pointing_phi.get()).reshape(-1,tGPU.constructor.nphi), cmap='seismic', vmin=-1e-15, vmax=1e-15)
                # plt.savefig("/mnt/home/sbelkner/git/pySHT/test/test_class_deflection_phi.png")
                
                # plt.imshow((pt_CPU - pointing_theta.get()).reshape(-1,tGPU.constructor.nphi), cmap='seismic', vmin=-1e-15, vmax=1e-15)
                # plt.savefig("/mnt/home/sbelkner/git/pySHT/test/test_class_deflection_theta.png")
                
                self.assertLess(res_theta, 1e-8, msg="res_theta: {}".format(res_theta))
                # self.assertLess(res_phi, 1e-8, msg="res_phi: {}".format(res_phi))


class TestIntegration(unittest.TestCase):
    
    @unittest.skip("Skipping this test method for now")
    def test_integration_lensingreconstruction(self):
        for test_case in test_cases:
            with self.subTest(input_value=test_case):
                pass


if __name__ == '__main__':
    unittest.main()