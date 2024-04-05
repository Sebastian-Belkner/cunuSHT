import numpy as np
import time
import cupy as cp
import shtns

lmax, mmax = 2048-1, 2048-1
_ = shtns.sht(int(lmax), int(mmax))
_.set_grid(flags=shtns.SHT_ALLOW_GPU + shtns.SHT_THETA_CONTIGUOUS)

alm_random = np.random.randn(_.nlm).astype(complex)
alm = cp.array(alm_random, dtype=complex)


start = time.time()
cp.cuda.runtime.deviceSynchronize()
buff = _.synth_grad(alm_random)
cp.cuda.runtime.deviceSynchronize()
stop = time.time()
ret = np.array([a.flatten() for a in buff])
print("synth_grad:: time elapsed: ", stop-start)

constructor = shtns.sht(int(lmax), int(mmax))
constructor.set_grid(flags=shtns.SHT_ALLOW_GPU + shtns.SHT_THETA_CONTIGUOUS, nlat=(lmax+1), nphi=2*(lmax+1))

alm_random = np.random.randn(constructor.nlm).astype(complex)
alm = cp.array(alm_random, dtype=complex)
x = cp.empty((constructor.nlat, constructor.nphi), dtype=np.double)

print(alm.shape, x.shape)
start = time.time()
cp.cuda.runtime.deviceSynchronize()
constructor.cu_SH_to_spat(alm.data.ptr, x.data.ptr)
cp.cuda.runtime.deviceSynchronize()
stop = time.time()
print(x)
print("cu_SH_to_spat:: time elapsed: ", stop-start) 