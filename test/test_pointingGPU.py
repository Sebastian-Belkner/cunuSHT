import unittest

import cupy as cp
import numpy as np

from pysht.c.podo_interface import Cpointing_ptrs, Cpointing_cparr, Cpointing_1Dto1D

def input_values(nring):
    npix = int(nring+1)*2*(nring+1)
    thetas = cp.linspace(1, np.pi, nring, dtype=cp.float64)
    phi0 = cp.zeros(nring, dtype=cp.float64)
    ringstarts = cp.linspace(0, npix-npix//nring, nring, dtype=np.uint64)
    nphis = cp.array([npix/nring for a in np.arange(nring)], dtype=np.uint64)
    del nring, npix
    return locals()

test_cases = [ 
    lmax for lmax in [2**n-1 for n in np.arange(8, 9)]
    #   (8192, 1024*2**n) for n in np.arange(1, 10) #range(16) is maximum possible
    ]

class TestUnit(unittest.TestCase):
    
    def test_unit_Cpointing_1Dto1D(self):
        for test_case in test_cases:
            with self.subTest(input_value=test_case):
                input_value = input_values(test_case)
                nlat=int(test_case+1)
                nphi=int(2*(test_case+1))
                npix = nlat * nphi
                spin1theta = cp.empty(shape=(1,nphi,nlat), dtype=cp.float64)
                spin1phi = cp.empty(shape=(1,nphi,nlat), dtype=cp.float64)
                out_pt = cp.zeros(npix, dtype=cp.float64)
                out_pp = cp.zeros(npix, dtype=cp.float64)
                Cpointing_1Dto1D(**input_value, spin1_theta=spin1theta.flatten(), spin1_phi=spin1phi.flatten(), out_ptheta=out_pt, out_pphi=out_pp)
                # self.assertEqual(np.mean(out_pt.get()), np.mean(out_pt.get()))
                # print("{:.2f}, {:.2f}\t (nring, npix) = ({}, {})".format(np.mean(out_pt.get()),np.mean(out_pp.get()), *test_case))
                print(out_pt.get())
                print(out_pp.get())
                # print(np.where(np.isnan(out_pt.get())))   
  
    @unittest.skip("Skipping this test method for now")
    def test_unit_Cpointing_ptrs(self):
        for test_case in test_cases:
            with self.subTest(input_value=test_case):
                input_value = input_values(*test_case)
                output_array = np.zeros(2*input_value["npix"], dtype=np.double)
                pointingsa = Cpointing_ptrs(**input_value, host_result=output_array)
                for key, val in input_value.items():
                    if type(val) != int:
                        del val
                for pointinga in pointingsa:
                    del pointinga
                del output_array
    
    @unittest.skip("Skipping this test method for now")
    def test_unit_Cpointing_cparr(self):
        for test_case in test_cases:
            with self.subTest(input_value=test_case):
                input_value = input_values(*test_case)
                out_pt = cp.zeros(input_value["npix"], dtype=cp.float64)
                out_pp = cp.zeros(input_value["npix"], dtype=cp.float64)
                
                Cpointing_cparr(**input_value, outarr_pt=out_pt, outarr_pp=out_pp)
                # self.assertEqual(np.mean(out_pt.get()), np.mean(out_pt.get()))
                print("{:.2f}, {:.2f}\t (nring, npix) = ({}, {})".format(np.mean(out_pt.get()),np.mean(out_pp.get()), *test_case))
                # print(out_pt.get())
                # print(np.where(np.isnan(out_pt.get())))


class TestIntegration(unittest.TestCase):
    
    @unittest.skip("Skipping this test method for now")
    def test_integration_synthgrad2pointing(self):
        for test_case in test_cases:
            with self.subTest(input_value=test_case):
                input_value = input_values(test_case)
                nlat=int(test_case+1)
                nphi=int(2*(test_case+1))
                npix = nlat * nphi
                output_array = np.zeros(2*npix, dtype=cp.double)
    
    @unittest.skip("Skipping this test method for now")
    def test_integration_pointing2nufft(self):
        for test_case in test_cases:
            with self.subTest(input_value=test_case):
                input_value = input_values(test_case)
                nlat=int(test_case+1)
                nphi=int(2*(test_case+1))
                npix = nlat * nphi
                output_array = np.zeros(2*npix, dtype=cp.double) 

if __name__ == '__main__':
    unittest.main()