import unittest

import cupy as cp
import numpy as np

from pysht.c.podo_interface import Cpointing_ptrs, Cpointing_cparr, Cpointing_1Dto1D

def input_values(nring, npix):
    nring = nring
    npix = npix
    synthmap = cp.random.randn((2*npix),dtype=cp.float64)
    thetas = cp.linspace(0, np.pi, nring, dtype=cp.float64)
    phi0 = cp.zeros(nring, dtype=cp.float64)
    ringstarts = cp.linspace(0, npix-npix//nring, nring, dtype=np.uint64)
    nphis = cp.array([npix/nring for a in np.arange(nring)], dtype=np.uint64)
    return locals()

test_cases = [ 
            #   (lmax, lmax**2*2) for lmax in [2**n-1 for n in np.arange(1, 12)]
              (8192, 1024*2**n) for n in np.arange(1, 10) #range(16) is maximum possible
              ]
class TestUnit(unittest.TestCase):
    # def test_unit_Cpointing_ptrs(self):
    #     for test_case in test_cases:
    #         with self.subTest(input_value=test_case):
    #             input_value = input_values(*test_case)
    #             output_array = np.zeros(2*input_value["npix"], dtype=np.double)
    #             pointingsa = Cpointing_ptrs(**input_value, host_result=output_array)
    #             for key, val in input_value.items():
    #                 if type(val) != int:
    #                     del val
    #             for pointinga in pointingsa:
    #                 del pointinga
    #             del output_array
                
    # def test_unit_Cpointing_cparr(self):
    #     for test_case in test_cases:
    #         with self.subTest(input_value=test_case):
    #             input_value = input_values(*test_case)
    #             out_pt = cp.zeros(input_value["npix"], dtype=cp.float64)
    #             out_pp = cp.zeros(input_value["npix"], dtype=cp.float64)
                
    #             Cpointing_cparr(**input_value, outarr_pt=out_pt, outarr_pp=out_pp)
    #             # self.assertEqual(np.mean(out_pt.get()), np.mean(out_pt.get()))
    #             print("{:.2f}, {:.2f}\t (nring, npix) = ({}, {})".format(np.mean(out_pt.get()),np.mean(out_pp.get()), *test_case))
    #             # print(out_pt.get())
    #             # print(np.where(np.isnan(out_pt.get())))
                
    def test_unit_Cpointing_1Dto1D(self):
        def input_values(nring, npix):
            nring = nring
            npix = npix
            spin1_theta = cp.random.randn((npix),dtype=cp.float64)
            spin1_phi = cp.random.randn((npix),dtype=cp.float64)
            thetas = cp.linspace(0, np.pi, nring, dtype=cp.float64)
            phi0 = cp.zeros(nring, dtype=cp.float64)
            ringstarts = cp.linspace(0, npix-npix//nring, nring, dtype=np.uint64)
            nphis = cp.array([npix/nring for a in np.arange(nring)], dtype=np.uint64)
            return locals()
        for test_case in test_cases:
            with self.subTest(input_value=test_case):
                input_value = input_values(*test_case)
                out_pt = cp.zeros(input_value["npix"], dtype=cp.float64)
                out_pp = cp.zeros(input_value["npix"], dtype=cp.float64)
                
                Cpointing_1Dto1D(**input_value, outarr_pt=out_pt, outarr_pp=out_pp)
                # self.assertEqual(np.mean(out_pt.get()), np.mean(out_pt.get()))
                print("{:.2f}, {:.2f}\t (nring, npix) = ({}, {})".format(np.mean(out_pt.get()),np.mean(out_pp.get()), *test_case))
                # print(out_pt.get())
                # print(np.where(np.isnan(out_pt.get())))
                

# class TestIntegration(unittest.TestCase):
#     def test_integration_pointing2doubling(self):
#         for test_case in test_cases:
#             with self.subTest(input_value=test_case):
#                 input_value = input_values(*test_case)
#                 output_array = np.zeros(2*input_value["npix"], dtype=cp.double)
#                 pointings = Cpointing_ptrs(**input_value, host_result=output_array)
#                 doublinga = Cdoubling_ptrs(pointings=pointings, nring=test_case[0], nphi=test_case[1]//test_case[0])  

if __name__ == '__main__':
    unittest.main()