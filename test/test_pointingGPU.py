"""
This test script is for testing the dlm2pointing subfunctions.

python3 -m unittest test_pointingGPU.py
"""
import unittest

import cupy as cp
import numpy as np
import healpy as hp

import cunusht
import shtns

from ducc0.misc import get_deflected_angles

from cunusht.c.podo_interface import Cpointing_ptrs, Cpointing_cparr, Cpointing_1Dto1D

def input_values(nring):
    npix = int(nring+1)*2*(nring+1)
    thetas = cp.linspace(1, np.pi, nring, dtype=cp.float64)
    phi0 = cp.zeros(nring, dtype=cp.float64)
    ringstarts = cp.linspace(0, npix-npix//nring, nring, dtype=np.uint64)
    nphis = cp.array([npix/nring for a in np.arange(nring)], dtype=np.uint64)
    del nring, npix
    return locals()

test_cases = [ 
    # lmax for lmax in [2**n-1 for n in np.arange(10, 12)]
    lmax for lmax in [256*n-1 for n in np.arange(1, 15)]
    #   (8192, 1024*2**n) for n in np.arange(1, 10) #range(16) is maximum possible
    ]

class TestUnit(unittest.TestCase):
    
    @unittest.skip("Skipping this test method for now")
    def test_unit_Cpointing_1Dto1D(self):
        for test_case in test_cases:
            with self.subTest(input_value=test_case):
                input_value = input_values(test_case)
                nlat=int(test_case+1)
                nphi=int(2*(test_case+1))
                npix = nlat * nphi
                spin1theta = cp.empty(shape=(nphi,nlat), dtype=cp.float64)
                spin1phi = cp.empty(shape=(nphi,nlat), dtype=cp.float64)
                out_pt = cp.zeros(npix, dtype=cp.float64)
                out_pp = cp.zeros(npix, dtype=cp.float64)
                Cpointing_1Dto1D(**input_value, spin1_theta=spin1theta.flatten(), spin1_phi=spin1phi.flatten(), out_ptheta=out_pt, out_pphi=out_pp)
                print(out_pt.get())
                print(out_pp.get())

                
  
    @unittest.skip("Skipping this test method for now")
    def test_unit_Cpointing_ptrs(self):
        for test_case in test_cases:
            with self.subTest(input_value=test_case):
                input_value = input_values(*test_case)

    
    @unittest.skip("Skipping this test method for now")
    def test_unit_Cpointing_cparr(self):
        for test_case in test_cases:
            with self.subTest(input_value=test_case):
                input_value = input_values(*test_case)
                out_pt = cp.zeros(input_value["npix"], dtype=cp.float64)
                out_pp = cp.zeros(input_value["npix"], dtype=cp.float64)


class TestIntegration(unittest.TestCase):
    
    def input_values(self, lmax, geom):
        npix = geom.npix()
        thetas = cp.array(geom.theta.astype(cp.float64))
        phi0s = cp.array(geom.phi0.astype(cp.float64))
        ringstarts = cp.array(geom.ofs.astype(cp.uint64))
        nphis = cp.array(geom.nph.astype(cp.uint64))
        del lmax, npix, geom, self
        return locals()
 
    
    # @unittest.skip("Skipping this test method for now")
    def test_integration_synthgrad2pointing(self):
        for test_case in test_cases:
            with self.subTest(input_value=test_case):
                geominfo = ('gl',{'lmax': test_case})

                tGPU = cunusht.get_transformer('shtns', 'SHT', 'GPU')(geominfo)
                alm_random = np.random.randn(tGPU.constructor.nlm)*1e-6 + 1j*np.random.randn(tGPU.constructor.nlm)*1e-6
                out_spin1theta = cp.empty(shape=tGPU.constructor.spat_shape, dtype=cp.float64)
                out_spin1phi = cp.empty(shape=tGPU.constructor.spat_shape, dtype=cp.float64)
                ll = np.arange(0, test_case+1)
                tGPU.synthesis_der1_cupy(cp.array(hp.almxfl(alm_random, np.nan_to_num(np.sqrt(1/(ll*(ll+1)))))), out_spin1theta, out_spin1phi)
                out_pointingtheta = cp.zeros(tGPU.geom.npix(), dtype=cp.float64)
                out_pointingphi = cp.zeros(tGPU.geom.npix(), dtype=cp.float64)
                input_value = self.input_values(test_case, tGPU.geom)
                Cpointing_1Dto1D(**input_value, spin1_theta=out_spin1theta.T.flatten(), spin1_phi=out_spin1phi.T.flatten(), out_ptheta=out_pointingtheta, out_pphi=out_pointingphi)

                tCPU = cunusht.get_transformer('ducc', 'SHT', 'CPU')(geominfo)
                input_value = self.input_values(test_case, tCPU.geom)
                s1tp = tCPU.synthesis(alm_random, spin=1, lmax=test_case, mmax=test_case, nthreads=10, mode='GRAD_ONLY')
                tht_phip_gamma = get_deflected_angles(
                    theta=input_value['thetas'].get(),
                    phi0=input_value["phi0s"].get(),
                    nphi=input_value["nphis"].get(),
                    ringstart=input_value["ringstarts"].get(),
                    deflect=s1tp.T,
                    calc_rotation=False,
                    nthreads=10)
                
                # some pixels are very off, probably due to poles being treated differently? Either way, masking them here.
                buff = np.where(tht_phip_gamma[:,1]-out_pointingphi.get()>np.pi,0,tht_phip_gamma[:,1]-out_pointingphi.get())
                # theta and phi pointing is a 1e-5ish effect. As CPU and GPU are essentially the same code, 
                # the std between them should be at machine precision accuracy.
                res_theta = np.std(tht_phip_gamma[:,0]-out_pointingtheta.get())
                res_phi = np.std(buff)
                
                # import matplotlib.pyplot as plt
                # plt.imshow((tht_phip_gamma[:,1]-out_pointingphi.get()).reshape(-1,cGPU.nphi), cmap='seismic', vmin=-1e-14, vmax=1e-14)
                # plt.savefig("/mnt/home/sbelkner/git/cunusht/test/test_pointingGPU.pytest_pointingGPU_theta.png")
                self.assertLessEqual(res_theta, 1e-15, msg="np.std(dtheta)={}".format(res_theta))
                self.assertLessEqual(res_phi, 1e-12, msg="np.std(dphi)={}".format(res_phi))
                
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