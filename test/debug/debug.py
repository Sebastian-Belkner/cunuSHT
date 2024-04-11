import os
print(os.environ['LD_LIBRARY_PATH'])
os.environ['LD_LIBRARY_PATH']="/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/lib64:$LD_LIBRARY_PATH"

import sys

import numpy as np
import time
import healpy as hp
import matplotlib.pyplot as plt
import cupy as cp
import lenspyx
import pysht
from pysht import get_geom
import delensalot
from delensalot import utils
from delensalot.sims.sims_lib import Xunl, Xsky

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

lmax, mmax, phi_lmax = 1023, 1023, 1023+256
# lmax, mmax = 2047, 2047
# lmax, mmax = 4095, 4095
ll = np.arange(0,lmax)


Tunl_random = np.array([np.random.randn(hp.Alm.getsize(lmax)) + 1j*np.random.randn(hp.Alm.getsize(lmax))], dtype=np.complex128)[0]*1e-6
dlm_random = np.array([np.random.randn(hp.Alm.getsize(phi_lmax)) + 1j*np.random.randn(hp.Alm.getsize(phi_lmax))], dtype=np.complex128)[0]*1e-6
print(dlm_random.shape, Tunl_random.shape)

Tunl = Tunl_random
dlm = dlm_random
# geominfo = ('tgl',{'lmax': lmax, 'smax':3})
# geominfo = ('gl',{'lmax':lmax})
solver = 'shtns'
geominfo = ('gl',{'lmax':lmax})
lenjob_geominfo = ('gl',{'lmax':phi_lmax}) #('thingauss', {'lmax':phi_lmax, 'smax':3})


import time

# time.sleep(5)
"""
gclm2lenmap
"""
kwargs = {
    'geominfo': geominfo,
    'nthreads': 10,
    'epsilon': 1e-10,
    'verbosity': 1,
    'single_prec': False,
    'planned': False,
}
    
deflection_kwargs = {
    'dlm': dlm,
    'mmax_dlm': phi_lmax,
    'epsilon': 1e-10,
    'verbosity': 1,
    'single_prec': False,
    'geominfo': lenjob_geominfo,
    'nthreads': 10,
}

defres = {}
for backend in ['CPU', "GPU"]:
    if backend == 'GPU':
        solvers = ['cufinufft']
        sht_solver = 'shtns' # 'shtns'
    elif backend == 'CPU':
        solvers = ['duccnufft'] # duccnufft lenspyx
        sht_solver = 'ducc' # 'shtns'

    for solver in solvers:
        for mode in ['nuFFT']:
            print("Testing solver={} backend={} mode={}...".format(solver, backend, mode))
            t = pysht.get_transformer(solver, mode, backend)
            t = t(sht_solver, **kwargs, deflection_kwargs=deflection_kwargs)

            print("\n----Testing function gclm2lenmap...----")
            if backend == 'GPU':
                defres.update({
                    backend: t.gclm2lenmap_cupy(
                        Tunl.copy(), dlm=dlm, lmax=lmax, mmax=lmax, spin=0, nthreads=10, mode=2)})
                
            else:
                if solver == 'lenspyx':
                    defres.update({
                        backend: t.gclm2lenmap(
                            Tunl.copy(), dlm=dlm, lmax=lmax, mmax=lmax, spin=0, nthreads=10, mode=2)})
                else:
                    defres.update({
                        backend: t.gclm2lenmap(
                            Tunl.copy(), dlm=dlm, lmax=lmax, mmax=lmax, spin=0, nthreads=10, mode=2)})