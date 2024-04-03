import os, sys
import numpy as np
os.environ['LD_LIBRARY_PATH']="/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/lib64:$LD_LIBRARY_PATH"
import cupy as cp

red = cp.random.randn((20),dtype=np.double) +1j*cp.random.randn((20),dtype=np.double)
imd = cp.random.randn((20),dtype=np.double) +1j*cp.random.randn((20),dtype=np.double)

print(red)

import _pointingpy as popy


npix = 40
nring = 4

thetas_CPU = np.linspace(0, np.pi, nring, dtype=np.double)
red = np.random.randn((npix))
imd = np.random.randn((npix))
output_array = np.zeros(npix, dtype=np.double)

red_GPU = cp.random.randn((npix))

pp = popy.test_Cpointing(thetas_CPU, nring, npix, red, imd, red_GPU, output_array)
print("return value: {}".format(pp))
print("output array: {}".format(output_array))