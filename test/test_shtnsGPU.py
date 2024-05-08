"""
TBD

python3 -m unittest test_shtnsGPU.py
"""
import unittest

import numpy as np
import cupy as cp
import shtns
import ducc0
import healpy as hp

import cunusht
import cunusht.c.podo_interface as podo

test_cases = [ 
              (lmax, lmax) for lmax in [2**n-1 for n in np.arange(6, 8)]
              ]
        
class TestUnit(unittest.TestCase):
    """
    Unittests here bypass cunusht interface whenever possible, and directly call the functions.
    """

    def test_unit_SHTns_synthgrad_cupyarray(self):
        for test_case in test_cases:
            with self.subTest(input_value=test_case):
                cGPU = shtns.sht(int(test_case[0]), int(test_case[1]))
                cGPU.set_grid(flags=shtns.SHT_ALLOW_GPU + shtns.SHT_THETA_CONTIGUOUS)#, nlat=int(test_case[0]+1), nphi=int(2*(test_case[1]+1)))

                alm_random = np.random.randn(cGPU.nlm) + 1j*np.random.randn(cGPU.nlm)
                alm = cp.array(alm_random)
                print("alm shape: ", alm.shape)

                out_theta = cp.empty(shape=(cGPU.nphi, cGPU.nlat), dtype=np.double)
                out_phi = cp.empty(shape=(cGPU.nphi, cGPU.nlat), dtype=np.double)
                print("GPU cupy output shape: {}, lmax = {}".format(out_theta.shape, test_case[0]))

                grad_theta, grad_phi = cGPU.synth_grad(alm_random)
                print("GPU non-cupy output shape: {}".format(grad_theta.shape))
                cGPU.cu_SHsph_to_spat(alm.data.ptr, out_theta.data.ptr, out_phi.data.ptr)

                self.assertEqual(0., np.linalg.norm(grad_theta - out_theta.get()))
                self.assertEqual(0., np.linalg.norm(grad_phi - out_phi.get()))
                # print(grad_theta, out_theta.get())
    
    @unittest.skip("Skipping this test method for now")
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
    Integration tests here use cunusht interface whenever possible.
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
                geominfo = ('gl',{'lmax':test_case[0]})
                nlm = hp.Alm.getsize(test_case[0]) # comes from phi_lmax
                alm_random = np.atleast_2d(
                    np.random.randn(nlm) +
                    1j*np.random.randn(nlm))
                alm_random = cp.array(alm_random)
                ntheta_CAR = (ducc0.fft.good_size(geominfo[1]['lmax'] + 2) + 3) // 4 * 4
                nphihalf_CAR = ducc0.fft.good_size(geominfo[1]['lmax'] + 1)
                nphi_CAR = 2 * nphihalf_CAR
                geominfo_CAR = (
                    'cc',
                        {
                    'lmax': geominfo[1]['lmax'],
                    'mmax': geominfo[1]['lmax'],
                    'ntheta': ntheta_CAR,
                    'nphi': nphi_CAR,
                        }
                )
                kwargs = {
                    'geominfo': geominfo_CAR,
                    'verbosity': 0,
                    'single_prec': False,
                    'nthreads': 10
                } 
                tCAR = cunusht.get_transformer('shtns', 'SHT', 'GPU')(**kwargs)
                CARmap = cp.empty((ntheta_CAR, nphi_CAR), dtype=np.double)
                
                ntheta_dCAR = 2 * ntheta_CAR-2
                nphi_dCAR = nphi_CAR
                CARdmap = cp.zeros(ntheta_dCAR * nphi_dCAR, dtype=np.double)
                tCAR.synthesis_cupy(alm_random, CARmap, spin=0, lmax=test_case[0], mmax=test_case[1], nthreads=10)
                podo.Cdoubling_1D(CARmap.reshape(nphi_CAR,-1).T.flatten(), int(ntheta_CAR), int(nphi_CAR), CARdmap)
                
                doubling1D_py = np.zeros((ntheta_dCAR, nphi_dCAR))
                synth1D_py = np.copy(CARmap.get().reshape(ntheta_CAR, nphi_CAR))
                doubling1D_py[:ntheta_CAR, :] = synth1D_py
                doubling1D_py[ntheta_CAR:, :nphihalf_CAR] = doubling1D_py[ntheta_CAR-2:0:-1, nphihalf_CAR:]
                doubling1D_py[ntheta_CAR:, nphihalf_CAR:] = doubling1D_py[ntheta_CAR-2:0:-1, :nphihalf_CAR]
                
                # assert None # TODO


    @unittest.skip("Skipping this test method for now")
    def test_integration_synthgrad2pointing(self):
        for test_case in test_cases:
            with self.subTest(input_value=test_case):
                pass

if __name__ == '__main__':
    unittest.main()