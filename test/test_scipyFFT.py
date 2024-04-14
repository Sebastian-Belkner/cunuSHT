"""
TBD

python3 -m unittest test_scipyFFT.py
"""
import unittest

import numpy as np
import cupy as cp
import shtns

test_cases = [ 
              (lmax, lmax) for lmax in [2**n-1 for n in np.arange(6, 8)]
              ]
        
class TestUnit(unittest.TestCase):

    @unittest.skip("Skipping this test method for now")
    def test_unit_scipy_iFFT(self):
        for test_case in test_cases:
            with self.subTest(input_value=test_case):
                pass
       
    @unittest.skip("Skipping this test method for now")      
    def test_unit_scipy_FFT(self):
        for test_case in test_cases:
            with self.subTest(input_value=test_case):
                pass
                    

class TestIntegration(unittest.TestCase):
    
    @unittest.skip("Skipping this test method for now")
    def test_integration_ddoubling2C2C(self):
        for test_case in test_cases:
            with self.subTest(input_value=test_case):
                pass

    @unittest.skip("Skipping this test method for now")
    def test_integration_C2C2nuFFT(self):
        for test_case in test_cases:
            with self.subTest(input_value=test_case):
                pass

if __name__ == '__main__':
    unittest.main()