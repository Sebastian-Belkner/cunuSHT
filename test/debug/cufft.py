import os
import pycuda.autoinit
from pycuda import gpuarray
import numpy as np
from skcuda import cufft as cf
import time
import healpy as hp
import cupy as cp
import lenspyx
from time import process_time
import pysht
from pysht import get_geom
import delensalot
from delensalot import utils
from delensalot.sims.sims_lib import Xunl, Xsky, Xobs


lmax, mmax = 2047, 2047
ll = np.arange(0,lmax)
lldlm = np.arange(0,lmax+1024)
# geominfo = ('tgl',{'lmax': lmax, 'smax':3})
# geominfo = ('gl',{'lmax':lmax})
solver = 'shtns'
geominfo = ('gl',{'lmax':lmax}) # if solver == 'shtns' else ('cc',{'nphi':4116, 'ntheta':2056})
# geominfo = ('cc',{'nphi':1032, 'ntheta':2058})
# geominfo = ('healpix',{'nside': 512})

synunl = Xunl(lmax=lmax, geominfo=geominfo)
synsky = Xsky(lmax=lmax, unl_lib=synunl, geominfo=geominfo, lenjob_geominfo=geominfo)

philm = synunl.get_sim_phi(0, space='alm')
dlm = hp.almxfl(philm, np.sqrt(lldlm*(lldlm+1)))

Tunl = synunl.get_sim_unl(0, spin=0, space='alm', field='temperature')
Tunlmap = synunl.get_sim_unl(0, spin=0, space='map', field='temperature')
Tsky = synsky.get_sim_sky(0, spin=0, space='map', field='temperature')
Tskyalm = synsky.get_sim_sky(0, spin=0, space='alm', field='temperature')

deflection_kwargs = {
    'nthreads': 4,
    'dlm':dlm,
    'mmax_dlm':len(ll),
    'epsilon':1e-5,
    'verbosity':0,
    'single_prec':False,
    'planned':False}

def flip_tpg_2d(m):
    # FIXME this should probably be lmax, not lmax_dlm
    # dim of m supposedly (2, -1)
    buff = np.array([_.reshape(2*(lmax+1),-1).T.flatten() for _ in m])
    return buff



philm = synunl.get_sim_phi(0, space='alm')
dlm = hp.almxfl(philm, np.sqrt(lldlm*(lldlm+1)))
solver = 'ducc'
geominfo = ('gl',{'lmax':lmax}) # if solver == 'shtns' else ('cc',{'nphi':4096, 'ntheta':2048})
print(geominfo)
tducc = pysht.get_transformer(solver, 'SHT', 'CPU')
tducc = tducc(geominfo)
deriv1ducc = tducc.synthesis(dlm, spin=1, lmax=lmax, mmax=mmax, nthreads=4, mode='GRAD_ONLY')
print(deriv1ducc.shape)


import ducc0

# transform slm to Clenshaw-Curtis map
ntheta = ducc0.fft.good_size(lmax + 2)
nphihalf = ducc0.fft.good_size(lmax + 1)
nphi = 2 * nphihalf
spin=0
gclm = np.atleast_2d(Tunl.copy())

def ducc_sht_mode(gclm, spin):
    gclm_ = np.atleast_2d(gclm)
    return 'GRAD_ONLY' if ((gclm_[0].size == gclm_.size) * (abs(spin) > 0)) else 'STANDARD'

m = ducc0.sht.experimental.synthesis_2d(alm=gclm, ntheta=ntheta, nphi=nphi, spin=spin, lmax=lmax, mmax=lmax, geometry="CC", nthreads=4, mode=ducc_sht_mode(gclm, spin))
print(m.shape)

geominfo = ('cc',{'nphi':4096, 'ntheta':2058})
tducccc = pysht.get_transformer('ducc', 'SHT', 'CPU')(geominfo)
m = tducccc.synthesis(gclm, spin=0, lmax=lmax, mmax=mmax, nthreads=4).reshape(2058,-1)




def doubling():
    map_dfs = np.empty((2 * ntheta - 2, nphi), dtype=np.complex128 if spin == 0 else ctype[m.dtype])
    if spin == 0:
        map_dfs[:ntheta, :] = m[0]
    else:
        map_dfs[:ntheta, :].real = m[0]
        map_dfs[:ntheta, :].imag = m[1]

    map_dfs[ntheta:, :nphihalf] = map_dfs[ntheta - 2:0:-1, nphihalf:]
    map_dfs[ntheta:, nphihalf:] = map_dfs[ntheta - 2:0:-1, :nphihalf]
    if (spin % 2) != 0:
        map_dfs[ntheta:, :] *= -1
    return map_dfs
        
t1 = process_time()
map_dfs = doubling()
t2 = process_time()
print('\ndoubling CPU time is: {:.3f} ms'.format((t2-t1)*100))

import skcuda.fft as cu_fft
from pycuda import gpuarray
BATCH, NX = map_dfs.T.shape

t1 = process_time()
data_o_gpu  = gpuarray.empty((BATCH,NX),dtype=np.complex64)
data_t = map_dfs.reshape((BATCH,NX))
plan = cu_fft.Plan(data_t.shape, np.complex64, np.complex64)
data_t_gpu  = gpuarray.to_gpu(data_t)
cu_fft.ifft(data_t_gpu, data_t_gpu, plan)
dataFft_gpu = data_o_gpu.get()
print(dataFft_gpu.shape)
t2 = process_time()
print('\nc2c GPU time is: {:.3f} ms'.format((t2-t1)*100))

tmp = np.empty(map_dfs.T.shape, dtype=np.complex128)
t1 = process_time()

print(ducc0.fft.c2c(map_dfs.T, axes=(0, 1), inorm=2, nthreads=4, out=tmp).shape)
t2 = process_time()
print('\nc2c CPU time is: {:.3f} ms'.format((t2-t1)*100))