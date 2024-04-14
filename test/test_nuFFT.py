"""
TBD

python3 -m unittest test_nuFFT.py
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
    def test_unit_DUCCnuFFT(self):
        for test_case in test_cases:
            with self.subTest(input_value=test_case):
                pass
       
    @unittest.skip("Skipping this test method for now")      
    def test_unit_finuFFT(self):
        for test_case in test_cases:
            with self.subTest(input_value=test_case):
                pass

    @unittest.skip("Skipping this test method for now")      
    def test_unit_cufinuFFT(self):
        for test_case in test_cases:
            with self.subTest(input_value=test_case):
                pass
                    

class TestIntegration(unittest.TestCase):
    
    @unittest.skip("Skipping this test method for now")
    def test_integration_C2Candpointing2nuFFT(self):
        for test_case in test_cases:
            with self.subTest(input_value=test_case):
                pass

if __name__ == '__main__':
    unittest.main()