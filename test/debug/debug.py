import os
print(os.environ['LD_LIBRARY_PATH'])
os.environ['LD_LIBRARY_PATH']="/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/lib64:$LD_LIBRARY_PATH"
import numpy as np
import healpy as hp
import pysht
from pysht import get_geom

lmax, mmax = 255, 255
ll = np.arange(0,lmax)
lldlm = np.arange(0,lmax+1024)
solver = 'shtns'
geominfo = ('gl',{'lmax':lmax}) 

shape = hp.Alm.getsize(lmax, mmax)
philm = np.random.uniform(-1, 1, shape) + 1.j * np.random.uniform(-1, 1, shape)
print(philm.shape)
dlm = hp.almxfl(philm, np.sqrt(lldlm*(lldlm+1)))*2e-8
print(dlm.shape)
Tunl = np.random.uniform(-1, 1, shape) + 1.j * np.random.uniform(-1, 1, shape)
deflection_kwargs = {
    'nthreads': 10,
    'dlm':dlm,
    'mmax_dlm':len(ll),
    'epsilon':1e-5,
    'verbosity':0,
    'single_prec':False,
    'planned':False}

defres = {}
geominfo_CAR = ('cc',{'nphi':2*(lmax+1), 'ntheta':lmax+1})
for backend in ["GPU"]:
    if backend == 'GPU':
        solvers = ['cufinufft']
        sht_solver = 'shtns' # 'shtns'
        tCAR = pysht.get_transformer('ducc', 'SHT', 'CPU')(geominfo_CAR)
    elif backend == 'CPU':
        solvers = ['duccnufft']
        sht_solver = 'ducc' # 'shtns'
        tCAR = pysht.get_transformer('ducc', 'SHT', 'CPU')(geominfo_CAR)

    for solver in solvers:
        for mode in ['nuFFT']:
            print("Testing solver={} backend={} mode={}...".format(solver, backend, mode))
            t = pysht.get_transformer(solver, mode, backend)
            t = t(sht_solver, geominfo, deflection_kwargs)

            print("\n----Testing function gclm2lenmap...----")          

import ctypes
def pointing_GPU(self, synth_spin1_map):
    cuda_lib = ctypes.CDLL('/mnt/home/sbelkner/git/pySHT/pysht/c/pointing.so')
    cuda_lib.pointing.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_double),
    ]
    cuda_lib.pointing.restype = None
    
    thetas_, phi0_, nphis_, ringstarts_ = self.geom.theta.astype(float), self.geom.phi0.astype(float), self.geom.nph.astype(int), self.geom.ofs.astype(int)
    red_, imd_ = synth_spin1_map.astype(np.double)
    npix_ = int(sum(nphis_))
    nrings_ = int(nphis_.size)
    output_array_ = np.zeros(shape=synth_spin1_map.size, dtype=np.double)

    thetas =        (ctypes.c_float * nrings_)(*thetas_)
    phi0 =          (ctypes.c_float * nrings_)(*phi0_)
    nphis =         (ctypes.c_int * nrings_)(*nphis_)
    ringstarts =    (ctypes.c_int * nrings_)(*ringstarts_)
    red =           (ctypes.c_double * npix_)(*red_)
    imd =           (ctypes.c_double * npix_)(*imd_)
    nrings =         ctypes.c_int(nrings_)
    npix =           ctypes.c_int(npix_)
    output_array = (ctypes.c_double * output_array_.size)(*output_array_)
    
    cuda_lib.pointing(thetas, phi0, nphis, ringstarts, red, imd, nrings, npix, output_array)
    # cuda_lib.pointing(thetas, phi0, nphis, ringstarts, red, imd, nrings, npix, output_array)
    
    ret = np.array(output_array, dtype=np.double)
    print("done: max value = {}, shape = {}".format(np.max(ret), ret.shape))
    _ = ret.reshape(synth_spin1_map.shape).T
    print(_.shape)
    return _

synth_spin1_map = t._build_d1(dlm, lmax, mmax)
ptg = pointing_GPU(t, synth_spin1_map)
np.sum(ptg[:,0].reshape(lmax+1,-1)[:,0]), ptg[:,0].reshape(lmax+1,-1)[:,0][:40]