"""
TBD

python3 -m unittest test_doublingGPU.py
"""

import unittest

import cupy as cp
import numpy as np
from delensalot.sims.sims_lib import Xunl, Xsky

import cunusht
import cunusht.c.podo_interface as podo
from cunusht.c.podo_interface import Cdoubling_cparr2D, Cdoubling_1D, Cadjoint_doubling_1D

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
    (lmax, 'single_prec') for lmax in [2**n-1 for n in np.arange(5, 7)]
    #   (8192, 1024*2**n) for n in np.arange(1, 10) #range(16) is maximum possible
    ]
        
class TestUnit(unittest.TestCase):
              
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
                
                
    def test_unit_Cadjoint_doubling1D(self):
        """doubling into adjoint doubling, compares against Python implementation
        """
        for test_case in test_cases:
            with self.subTest(input_value=test_case):
                lmax = test_case[0]
                single_prec = True if test_case[1] == 'single_prec' else False
                dtype = np.float32 if single_prec else np.double
                
                ntheta_CAR = (ducc0.fft.good_size(lmax + 2) + 3) // 4 * 4
                nphi_CAR = ducc0.fft.good_size(lmax + 1) * 2
                synth1D = np.arange(ntheta_CAR * nphi_CAR, dtype=dtype).reshape(nphi_CAR,-1).T.flatten()
                doubling1D_py = np.zeros((2 * ntheta_CAR-2, nphi_CAR), dtype=dtype)
                synth1D_py = np.copy(synth1D.reshape(ntheta_CAR, nphi_CAR))
                
                # This does the doubling
                nphihalf = nphi_CAR//2
                doubling1D_py[:ntheta_CAR, :] = synth1D_py
                doubling1D_py[ntheta_CAR:, :nphihalf] = doubling1D_py[ntheta_CAR-2:0:-1, nphihalf:]
                doubling1D_py[ntheta_CAR:, nphihalf:] = doubling1D_py[ntheta_CAR-2:0:-1, :nphihalf]

                # This does the adjoint doubling
                d1D_py = np.copy(doubling1D_py)
                d1D_py[1:ntheta_CAR - 1, :nphihalf] += d1D_py[-1:ntheta_CAR - 1:-1, nphihalf:]#d1D_py[-1:ntheta - 1:-1, nphihalf:]
                d1D_py[1:ntheta_CAR - 1, nphihalf:] += d1D_py[-1:ntheta_CAR - 1:-1, :nphihalf]#d1D_py[-1:ntheta - 1:-1, :nphihalf]
                d1D_py = d1D_py[:ntheta_CAR, :]
                map_adj = np.empty((1, ntheta_CAR, nphi_CAR), dtype=dtype)
                map_adj[0] = d1D_py.real

                # This is what we want to test
                CARmap = cp.empty(shape=(ntheta_CAR*nphi_CAR), dtype=np.float32) if single_prec else cp.empty(shape=(ntheta_CAR*nphi_CAR), dtype=np.double)
                CARdmap = cp.array(doubling1D_py.flatten(), dtype=np.float32) if single_prec else cp.array(doubling1D_py.flatten(), dtype=np.double)
                Cadjoint_doubling_1D(CARdmap, int(ntheta_CAR), int(nphi_CAR), CARmap)
                
                
                self.assertEqual(np.mean(CARmap.get()), np.mean(map_adj))


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
                cc_transformer = cunusht.get_transformer('shtns', 'SHT', 'GPU')(geominfo_CAR)
                
                CARmap = cp.empty((ntheta_CAR, nphi_CAR), dtype=np.double)
                ntheta_dCAR, nphi_dCAR = 2*ntheta_CAR-2, nphi_CAR
                CARdmap = cp.zeros(ntheta_dCAR*nphi_dCAR, dtype=np.double)
                cc_transformer.synthesis_cupy(cp.array(gclm), CARmap, spin=0, lmax=lmax, mmax=lmax, nthreads=10)
                
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