import unittest

import cupy as cp
import numpy as np

from pysht.c.podo_interface import Cdoubling_cparr2D, Cdoubling_1D

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
    (4, 40),
    (8, 80),
    (16, 160),
    (32, 320),
    (64, 640),
    (128, 1280),
    (256, 2560),
    (512, 5120),
    (1024, 10240),
    # (2048, 20480),
    # (4096, 40960),
    # (8192, 81920),
    # (16384, 163840),
    # (32768, 327680),
    # (65536, 655360),
    # (131072, 1310720),
    # (262144, 2621440)
]
        
class TestUnit(unittest.TestCase):
    # def test_unit_Cdoubling_cparr(self):
    #     for test_case in test_cases:
    #         with self.subTest(input_value=test_case):
    #             CARmap = cp.random.randn((test_case[-1]),dtype=np.double)
    #             doublinga = Cdoubling_ptrs(synth2D=CARmap, nring=test_case[0], nphi=test_case[1]//test_case[0])  
    #             print("return value: {}".format(result))
    #             self.assertEqual(doublinga, 100)
    
    def test_unit_Cdoubling_cparr1D(self):
        for test_case in test_cases:
            with self.subTest(input_value=test_case):
                nring = test_case[0]
                npix = test_case[1]
                nphi = npix//nring
                CARmap = cp.random.randn((nring * nphi), dtype=np.double)
                out = cp.empty(((2 * nring-2) * nphi), dtype=np.double)
                Cdoubling_1D(CARmap, nring, nphi, out)
                print(out.shape)
                
                
    # def test_unit_Cdoubling_cparr_synth2D(self):
    #     for test_case in test_cases:
    #         with self.subTest(input_value=test_case):
    #             nring = test_case[0]
    #             npix = test_case[1]
    #             nphi = npix//nring
    #             CARmap = cp.random.randn((nring * nphi), dtype=np.double)
    #             out = cp.empty(((2 * nring-2) * nphi), dtype=np.double)
    #             Cdoubling_2Dto1D(CARmap, nring, nphi, out)
    #             print(out.shape)

          
    # def test_unit_Cdoubling_cparr2D(self):
    #     for test_case in test_cases:
    #         with self.subTest(input_value=test_case):
    #             CARmap = cp.random.randn((test_case[0], test_case[1]//test_case[-1]),dtype=np.double)
    #             Cdoubling_cparr2D(synth2D=CARmap, nring=test_case[0], nphi=test_case[1]//test_case[0])  


# class TestIntegration(unittest.TestCase):


if __name__ == '__main__':
    unittest.main()