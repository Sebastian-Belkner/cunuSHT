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
from ducc0.misc import get_deflected_angles
from delensalot.sims.sims_lib import Xunl, Xsky, Xobs


lmax, mmax = 2047, 2047
ll = np.arange(0,lmax)
lldlm = np.arange(0,lmax+1024)

solver = 'shtns'
geominfo = ('gl',{'lmax':lmax})

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


"""
GPU nuFFT
"""
sht_solver = 'shtns' # shtns
geominfo = ('gl',{'lmax':lmax}) #if sht_solver == 'shtns' else ('cc',{'nphi':4116, 'ntheta':2056})
for solver in ['cufinufft']:
    for backend in ['GPU']:
        for mode in ['nuFFT']:
            print("Testing solver={} backend={} mode={}...".format(solver, backend, mode))
            t = pysht.get_transformer(solver, mode, backend)
            t = t(sht_solver, geominfo, deflection_kwargs)
            t_gpu = t
            # t.set_geometry(geominfo)
            print("\n----Testing function gclm2lenmap...----")
            defres = t.gclm2lenmap(gclm=Tunl.copy(), dlm=dlm, lmax=lmax, mmax=lmax, spin=0, nthreads=4)

            print("\n----Testing function lenmap2gclm...----")
            defres = t.lenmap2gclm(points=Tsky.copy(), dlm=dlm, spin=0, lmax=lmax, mmax=lmax, nthreads=4)
            
            print("\n----Testing function lensgclm...----")
            defres = t.lensgclm(Tunl.copy(), spin=0, lmax_out=lmax, dlm=dlm, nthreads=4)

            # print("Testing function synthesis_general...")
            # defres = t.synthesis_general(Tunl.copy(), dlm=dlm, lmax=lmax, mmax=lmax, spin=0, nthreads=4)
            
            # print("Testing function adjoint_synthesis_general...")
            # defres = t.adjoint_synthesis_general(Tunl.copy(), dlm=dlm, lmax=lmax, mmax=lmax, spin=0, nthreads=4)
            print('\n\n')