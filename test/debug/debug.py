import shtns
import pysht.geometry as geometry
import ducc0
import os
lmax = 511

ntheta = (ducc0.fft.good_size(lmax + 2)+3)//4*4
nphihalf = ducc0.fft.good_size(lmax + 1)
nphi = 2 * nphihalf
print(ntheta, nphi)

os.environ['SHTNS_VERBOSE']="2"

constructor = shtns.sht(int(lmax), int(lmax))
constructor.set_grid(flags=shtns.SHT_ALLOW_GPU + shtns.sht_reg_poles + shtns.SHT_THETA_CONTIGUOUS, nlat=ntheta, nphi=nphi)

print(constructor.print_info())


# import os
# print(os.environ['LD_LIBRARY_PATH'])
# os.environ['LD_LIBRARY_PATH']="/mnt/sw/nix/store/zi2wc26znf75csf5hhz77p0d2bbz53ih-cuda-11.8.0/lib64:$LD_LIBRARY_PATH"

# import sys

# import numpy as np
# import time
# import healpy as hp
# import matplotlib.pyplot as plt
# import cupy as cp
# import lenspyx
# import pysht
# from pysht import get_geom
# import delensalot
# from delensalot import utils
# from delensalot.sims.sims_lib import Xunl, Xsky

# from mpl_toolkits.axes_grid1 import make_axes_locatable

# lmax, mmax = 511, 511
# # lmax, mmax = 2047, 2047
# # lmax, mmax = 4095, 4095
# ll = np.arange(0,lmax)
# lldlm = np.arange(0,lmax+1024)
# # geominfo = ('tgl',{'lmax': lmax, 'smax':3})
# # geominfo = ('gl',{'lmax':lmax})
# solver = 'shtns'
# geominfo = ('gl',{'lmax':lmax}) # if solver == 'shtns' else ('cc',{'nphi':4116, 'ntheta':2056})
# # geominfo = ('cc',{'nphi':1032, 'ntheta':2058})
# # geominfo = ('healpix',{'nside': 512})

# synunl = Xunl(lmax=lmax, geominfo=geominfo)
# synsky = Xsky(lmax=lmax, unl_lib=synunl, geominfo=geominfo, lenjob_geominfo=geominfo)

# philm = synunl.get_sim_phi(0, space='alm')
# dlm = hp.almxfl(philm, np.sqrt(lldlm*(lldlm+1)))

# Tunl = synunl.get_sim_unl(0, spin=0, space='alm', field='temperature')
# Tunlmap = synunl.get_sim_unl(0, spin=0, space='map', field='temperature')
# Tsky = synsky.get_sim_sky(0, spin=0, space='map', field='temperature')
# Tskyalm = synsky.get_sim_sky(0, spin=0, space='alm', field='temperature')



# """
# gclm2lenmap
# """
# deflection_kwargs = {
#     'nthreads': 10,
#     'dlm':dlm,
#     'mmax_dlm':len(ll),
#     'epsilon':1e-5,
#     'verbosity':0,
#     'single_prec':False,
#     'planned':False}

# defres = {}
# for backend in ['CPU', "GPU"]:
#     if backend == 'GPU':
#         solvers = ['cufinufft']
#         sht_solver = 'shtns' # 'shtns'
#     elif backend == 'CPU':
#         solvers = ['lenspyx'] #duccnufft
#         sht_solver = 'ducc' # 'shtns'

#     for solver in solvers:
#         for mode in ['nuFFT']:
#             print("Testing solver={} backend={} mode={}...".format(solver, backend, mode))
#             t = pysht.get_transformer(solver, mode, backend)
#             t = t(sht_solver, geominfo, deflection_kwargs)

#             print("\n----Testing function gclm2lenmap...----")
#             if backend == 'GPU':
#                 defres.update({
#                     backend: t.gclm2lenmap_cupy(
#                         Tunl.copy(), dlm=dlm, lmax=lmax, mmax=lmax, spin=0, nthreads=10, mode=2)})
                
#             else:
#                 if solver == 'lenspyx':
#                     defres.update({
#                         backend: t.gclm2lenmap(
#                             Tunl.copy(), dlm=dlm, lmax=lmax, mmax=lmax, spin=0, nthreads=10)})
#                 else:
#                     defres.update({
#                         backend: t.gclm2lenmap(
#                             Tunl.copy(), dlm=dlm, lmax=lmax, mmax=lmax, spin=0, nthreads=10, mode=2)})