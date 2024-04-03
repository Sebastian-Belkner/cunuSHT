import os, sys
import numpy as np
os.environ['LD_LIBRARY_PATH']="/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/lib64:$LD_LIBRARY_PATH"
import cupy as cp

npix = 40
nring = 4

synthmaps = cp.random.randn((npix),dtype=np.double) +1j*cp.random.randn((npix),dtype=np.double)

import pysht.c.podo_interface as podo

thetas = cp.linspace(0, np.pi, nring, dtype=cp.double)
phi0 = cp.zeros(nring, dtype=cp.double)
ringstarts = cp.linspace(0, 30, nring, dtype=cp.int32)
nphis = cp.array([npix/nring for a in np.arange(nring)], dtype=cp.int)

output_array = np.zeros(2*npix, dtype=cp.double)

print(nphis, ringstarts, thetas, phi0)

pp = podo.Cpointing_ptrs(thetas, phi0, nphis, ringstarts, synthmaps, nring, npix, output_array)
print("return value: {}".format(pp))

pp = podo.Cdoubling_ptrs(pp, nring, npix//nring)
print("return value: {}".format(pp))

del pp