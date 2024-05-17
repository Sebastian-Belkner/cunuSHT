from cufinufft import Plan
import numpy as np
import sys
import time
epsilon = 1e-10
isign = 1
n_trans = 1
nuFFT_dtype = np.float32
nuFFTtype = 2
nuFFTshape = (sys.argv[1], sys.argv[1])
nuFFTplan = Plan(nuFFTtype, nuFFTshape, n_trans, epsilon, isign, nuFFT_dtype, gpu_method=2, gpu_sort=1, gpu_kerevalmeth=0, upsampfac=2.0)
time.sleep(5)
# del nuFFTplan