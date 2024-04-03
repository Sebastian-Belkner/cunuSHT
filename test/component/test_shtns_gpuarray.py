import numpy as np

from pycuda import gpuarray
import pycuda.autoinit
import shtns
lmax, mmax = 100, 100
constructor = shtns.sht(int(lmax), int(mmax))
constructor.set_grid(flags=shtns.SHT_ALLOW_GPU + shtns.SHT_THETA_CONTIGUOUS)

# import cupy as cp
# alm = gpuarray.zeros(constructor.nlm, dtype=complex)   # or get a conforming cupy array from somewhere else

alm_random = np.random.randn(constructor.nlm).astype(np.complex)
alm = gpuarray.to_gpu(alm_random)
x = gpuarray.empty((constructor.nphi, constructor.nlat), dtype=np.double)   # theta contiguous, array that will hold the result
constructor.cu_SH_to_spat(alm.ptr, x.ptr)  # will fill x with the spatial data synthesized from alm

print(x)