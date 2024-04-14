"""
TBD

python3 -m unittest test_doublingGPU.py
"""

import unittest

import cupy as cp
import numpy as np
from delensalot.sims.sims_lib import Xunl, Xsky

import pysht
import pysht.c.podo_interface as podo
from pysht.c.podo_interface import Cdoubling_cparr2D, Cdoubling_1D, Cdoubling_ptrs

import ducc0

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
    (lmax, lmax**2*2) for lmax in [2**n-1 for n in np.arange(6, 7)]
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
                lmax = test_case[0]
                ntheta = (ducc0.fft.good_size(lmax + 2) + 3) // 4 * 4
                nphi = ducc0.fft.good_size(lmax + 1)*2
                CARmap = cp.random.randn((ntheta * nphi), dtype=np.double)
                out = cp.empty(((2 * ntheta-2) * nphi), dtype=np.double)
                Cdoubling_1D(CARmap, int(2*ntheta-2), int(nphi), out)
 
                CARmap_py = np.arange((ntheta * nphi))
                doubling1D_py = np.zeros((2 * ntheta-2, nphi))
                CARmap_py = CARmap_py.reshape(ntheta, nphi)

                nphihalf = nphi//2
                doubling1D_py[:ntheta, :] = CARmap_py
                doubling1D_py[ntheta:, :nphihalf] = doubling1D_py[ntheta-2:0:-1, nphihalf:]
                doubling1D_py[ntheta:, nphihalf:] = doubling1D_py[ntheta-2:0:-1, :nphihalf]
                
                # self.assertAlmostEquals(np.array(out.get()), doubling1D_py.flatten())

    @unittest.skip("Skipping this test method for now")   
    def test_unit_Cdoubling_cparr2D(self):
        for test_case in test_cases:
            with self.subTest(input_value=test_case):
                CARmap = cp.random.randn((test_case[0], test_case[1]//test_case[-1]),dtype=np.double)
                Cdoubling_cparr2D(synth2D=CARmap, nring=test_case[0], nphi=test_case[1]//test_case[0])  


class TestIntegration(unittest.TestCase):
    
    def test_integration_synth2doubling(self):
        for test_case in test_cases:
            with self.subTest(input_value=test_case):
                lmax = test_case[0]
                geominfo = ('gl',{'lmax':lmax})
                ntheta_CAR = (ducc0.fft.good_size(geominfo[1]['lmax'] + 2) + 3) // 4 * 4
                nphihalf_CAR = ducc0.fft.good_size(geominfo[1]['lmax'] + 1)
                nphi_CAR = 2 * nphihalf_CAR
                
                synunl = Xunl(lmax=lmax, geominfo=geominfo)
                synsky = Xsky(lmax=lmax, unl_lib=synunl, geominfo=geominfo, lenjob_geominfo=geominfo)
                toyunllm = synsky.get_sim_sky(0, spin=0, space='alm', field='temperature')
                gclm = np.atleast_2d(toyunllm)
                geominfo_CAR = ('cc',{'lmax': geominfo[1]['lmax'], 'mmax':geominfo[1]['lmax'], 'ntheta':ntheta_CAR, 'nphi':nphi_CAR})
                cc_transformer = pysht.get_transformer('shtns', 'SHT', 'GPU')(geominfo_CAR)
                
                CARmap = cp.empty((ntheta_CAR, nphi_CAR), dtype=np.double)
                ntheta_dCAR, nphi_dCAR = 2*ntheta_CAR-2, nphi_CAR
                CARdmap = cp.zeros(ntheta_dCAR*nphi_dCAR, dtype=np.double)
                cc_transformer.synthesis_cupy(cp.array(gclm), CARmap, spin=0, lmax=lmax, mmax=lmax, nthreads=10)
                
                print("shapes: ", CARmap.shape, CARdmap.shape, ntheta_CAR, nphi_CAR, ntheta_dCAR, nphi_dCAR)
                podo.Cdoubling_1D(CARmap.flatten(), int(ntheta_dCAR), int(nphi_dCAR), CARdmap)
                
                doubling1D_py = np.zeros((ntheta_dCAR, nphi_dCAR))
                synth1D_py = np.copy(CARmap.get().reshape(ntheta_CAR, nphi_CAR))
                doubling1D_py[:ntheta_CAR, :] = synth1D_py
                doubling1D_py[ntheta_CAR:, :nphihalf_CAR] = doubling1D_py[ntheta_CAR-2:0:-1, nphihalf_CAR:]
                doubling1D_py[ntheta_CAR:, nphihalf_CAR:] = doubling1D_py[ntheta_CAR-2:0:-1, :nphihalf_CAR]
                np.testing.assert_almost_equal(CARdmap.get(), doubling1D_py.flatten())


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