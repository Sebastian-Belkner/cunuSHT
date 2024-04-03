import os, sys
import numpy as np
os.environ['LD_LIBRARY_PATH']="/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/lib64:$LD_LIBRARY_PATH"
import cupy as cp

red = cp.random.randn((20),dtype=np.double) +1j*cp.random.randn((20),dtype=np.double)
imd = cp.random.randn((20),dtype=np.double) +1j*cp.random.randn((20),dtype=np.double)

import pysht.c.podo_interface as podo

npix = 40
nring = 4

thetas = cp.linspace(0, np.pi, nring, dtype=np.double)
phi0 = cp.zeros(nring, dtype=np.double)

ringstarts = cp.linspace(0, 30, nring, dtype=np.int32)
nphis = cp.array([npix/nring for a in np.arange(nring)], dtype=int)
output_array = np.zeros(2*npix, dtype=np.double)

print(nphis, ringstarts, thetas, phi0)

pp = podo.Cpointing_ptrs(thetas, phi0, nphis, ringstarts, red, imd, nring, npix, output_array)
print("return value: {}".format(pp))
print("output array: {}".format(output_array))

pp = podo.Cdoubling_ptrs(pp, nring, npix//nring)
print("return value: {}".format(pp))

del pp