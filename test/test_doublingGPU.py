import unittest

import cupy as cp
import numpy as np

from pysht.c.podo_interface import Cdoubling_cparr2D, Cdoubling_1D, Cdoubling_ptrs, Cdoubling_2Dto1D

def input_values(nring, npix):
    nring = nring
    npix = npix
    synthmaps = cp.random.randn((npix),dtype=np.double) +1j*cp.random.randn((npix),dtype=np.double)
    thetas = cp.linspace(0, np.pi, nring, dtype=cp.double)
    phi0 = cp.zeros(nring, dtype=cp.double)
    ringstarts = cp.linspace(0, npix-npix//nring, nring, dtype=cp.int32)
    nphis = cp.array([npix/nring for a in np.arange(nring)], dtype=int)
    return locals()

test_cases = [ 
    (nring, nring**2*2) for nring in [2**n-1 for n in np.arange(8, 9)]
    #   (8192, 1024*2**n) for n in np.arange(1, 10) #range(16) is maximum possible
    ]
        
class TestUnit(unittest.TestCase):
    
    @unittest.skip("Skipping this test method for now")
    def test_unit_Cdoubling_cparr(self):
        for test_case in test_cases:
            with self.subTest(input_value=test_case):
                CARmap = cp.random.randn((test_case[-1]),dtype=np.double)
                doublinga = Cdoubling_ptrs(synth2D=CARmap, nring=test_case[0], nphi=test_case[1]//test_case[0])  
                print("return value: {}".format(doublinga))
                self.assertEqual(doublinga, 100)
                
    def test_unit_Cdoubling1D(self):
        for test_case in test_cases:
            with self.subTest(input_value=test_case):
                ntheta = test_case[0]
                npix = test_case[1]
                nphi = npix//ntheta
                CARmap = cp.random.randn((ntheta * nphi), dtype=np.double)
                out = cp.empty(((2 * ntheta-2) * nphi), dtype=np.double)
                Cdoubling_1D(CARmap, ntheta, nphi, out)
                
                CARmap_py = np.arange((ntheta * nphi))
                doubling1D_py = np.zeros((2 * ntheta-2, nphi))
                CARmap_py = CARmap_py.reshape(ntheta, nphi)

                nphihalf = nphi//2
                doubling1D_py[:ntheta, :] = CARmap_py
                doubling1D_py[ntheta:, :nphihalf] = doubling1D_py[ntheta-2:0:-1, nphihalf:]
                doubling1D_py[ntheta:, nphihalf:] = doubling1D_py[ntheta-2:0:-1, :nphihalf]
                
                self.assertAlmostEquals(out, doubling1D_py)
                
    @unittest.skip("Skipping this test method for now")
    def test_unit_Cdoubling_cparr_synth2D(self):
        for test_case in test_cases:
            with self.subTest(input_value=test_case):
                nring = test_case[0]
                npix = test_case[1]
                nphi = npix//nring
                CARmap = cp.random.randn((nring * nphi), dtype=np.double)
                out = cp.empty(((2 * nring-2) * nphi), dtype=np.double)
                Cdoubling_2Dto1D(CARmap, nring, nphi, out)
                print(out.shape)

    @unittest.skip("Skipping this test method for now")   
    def test_unit_Cdoubling_cparr2D(self):
        for test_case in test_cases:
            with self.subTest(input_value=test_case):
                CARmap = cp.random.randn((test_case[0], test_case[1]//test_case[-1]),dtype=np.double)
                Cdoubling_cparr2D(synth2D=CARmap, nring=test_case[0], nphi=test_case[1]//test_case[0])  


class TestIntegration(unittest.TestCase):
    
    @unittest.skip("Skipping this test method for now")
    def test_integration_synth2doubling(self):
        for test_case in test_cases:
            with self.subTest(input_value=test_case):
                input_value = input_values(test_case)
                nlat=int(test_case+1)
                nphi=int(2*(test_case+1))
                npix = nlat * nphi
                output_array = np.zeros(2*npix, dtype=cp.double)
                
    @unittest.skip("Skipping this test method for now")
    def test_integration_doubling2C2C(self):
        for test_case in test_cases:
            with self.subTest(input_value=test_case):
                input_value = input_values(test_case)
                nlat=int(test_case+1)
                nphi=int(2*(test_case+1))
                npix = nlat * nphi
                output_array = np.zeros(2*npix, dtype=cp.double)

if __name__ == '__main__':
    unittest.main()