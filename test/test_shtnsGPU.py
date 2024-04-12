"""
python3 -m unittest test_shtns_cupyarray.py
"""
import unittest

import numpy as np
import cupy as cp
import shtns

import pysht
import pysht.c.podo_interface as podo

test_cases = [ 
              (lmax, lmax) for lmax in [2**n-1 for n in np.arange(6, 8)]
              ]
        
class TestUnit(unittest.TestCase):
    """
    Unittests here bypass pysht interface whenever possible, and directly call the functions.
    """

    def test_unit_SHTns_synthgrad_cupyarray(self):
        for test_case in test_cases:
            with self.subTest(input_value=test_case):
                cGPU = shtns.sht(int(test_case[0]), int(test_case[1]))
                cGPU.set_grid(flags=shtns.SHT_ALLOW_GPU + shtns.SHT_THETA_CONTIGUOUS, nlat=int(test_case[0]+1), nphi=int(2*(test_case[1]+1)))

                alm_random = np.array([np.random.randn(cGPU.nlm) + 1j*np.random.randn(cGPU.nlm)], dtype=np.complex128)
                alm = cp.array(alm_random)

                # print("shape of alm: ", alm.shape)
                # print("shape of alm_random: ", alm_random.shape)

                out_theta = cp.empty(shape=(1,cGPU.nphi,cGPU.nlat), dtype=np.double)
                out_phi = cp.empty(shape=(1,cGPU.nphi,cGPU.nlat), dtype=np.double)

                grad_theta, grad_phi = cGPU.synth_grad(alm_random)
                cGPU.cu_SHsph_to_spat(alm.data.ptr, out_theta.data.ptr, out_phi.data.ptr)

                self.assertEqual(0., np.linalg.norm(grad_theta - out_theta.get()))
                self.assertEqual(0., np.linalg.norm(grad_phi - out_phi.get()))
                  
    def test_unit_SHTns_synthesis_cupyarray(self):
        for test_case in test_cases:
            with self.subTest(input_value=test_case):
                cGPU = shtns.sht(int(test_case[0]), int(test_case[1]))
                cGPU.set_grid(flags=shtns.SHT_ALLOW_GPU + shtns.SHT_THETA_CONTIGUOUS)#, nlat=int(test_case[0]+1), nphi=int(2*(test_case[1]+1)))
                alm_random = np.random.randn(cGPU.nlm) + 1j*np.random.randn(cGPU.nlm)
                alm = cp.array(alm_random, dtype=complex)
                synthesis_out = cp.empty((cGPU.nphi, cGPU.nlat), dtype=np.double)

                cGPU.cu_SH_to_spat(alm.data.ptr, synthesis_out.data.ptr)
                synth = cGPU.synth(alm_random)

                self.assertEqual(0., np.linalg.norm(synth - synthesis_out.get()))


class TestIntegration(unittest.TestCase):
    """
    Integration tests here use pysht interface whenever possible.
    """
    
    @unittest.skip("Skipping this test method for now")
    def test_integration_synthesis2doubling(self):
        """
        Random toy data, (lmax, mmax).
        
        Asserts:
            synth unequal 0
            compare synth to CPU code
            compare doubling to CPU code
        """
        for test_case in test_cases:
            with self.subTest(input_value=test_case):
                ntheta_CAR, nphi_CAR = None, None # comes from good_size
                ntheta_dCAR, nphi_dCAR = 2*ntheta_CAR-2, nphi_CAR
                nlm = None # comes from phi_lmax
                
                alm_random = np.atleast_2d(np.random.randn(nlm) + 1j*np.random.randn(nlm)).astype(np.complex128)
                kwargs = {
                    'nthreads': 10,
                    'mmax_dlm':test_case[1],
                    'epsilon':1e-10,
                    'verbosity':0,
                    'single_prec':False,
                }
                t = pysht.get_transformer(solver='shtns', mode='SHT', backend='GPU')
                t = t(sht_solver='shtns', **kwargs)
                CARmap = cp.empty((ntheta_CAR, nphi_CAR), dtype=np.double)
                CARdmap = cp.zeros((ntheta_dCAR)*nphi_dCAR, dtype=np.double)
                
                t.synthesis_cupy(alm_random, CARmap, spin=0, lmax=test_case[0], mmax=test_case[1], nthreads=10)
                podo.Cdoubling_1D(CARmap.reshape(nphi_CAR,-1).T.flatten(), int(ntheta_CAR), int(nphi_CAR), CARdmap)
                
                assert None # TODO


    @unittest.skip("Skipping this test method for now")
    def test_integration_synthgrad2pointing(self):
        for test_case in test_cases:
            with self.subTest(input_value=test_case):
                pass

if __name__ == '__main__':
    unittest.main()