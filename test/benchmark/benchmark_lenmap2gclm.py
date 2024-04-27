"""
Benchmark gclm2lenmap by scanning across different solvers, backends, and modes, and for different lmax values.
"""

import numpy as np
import time
import healpy as hp
import matplotlib.pyplot as plt
import pysht
import cupy as cp
import sys
from time import process_time

from delensalot.sims.sims_lib import Xunl, Xsky
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


runinfos = [
    # ("CPU", "lenspyx", 'ducc'),
    # ("CPU", "duccnufft", 'ducc'),
    ("GPU", "cufinufft", 'shtns')
    ]
epsilons = [1e-6]
# lmaxs = [256*n-1 for n in np.arange(int(sys.argv[1]), 24)]
lmaxs = [int(sys.argv[1])]
phi_lmaxs = [lmax for lmax in lmaxs]
defres = {}
Tsky = None
Tsky2 = None
for epsilon in epsilons:
    for lmax, phi_lmax in zip(lmaxs, phi_lmaxs):
        geominfo = ('gl',{'lmax': lmax})
        lenjob_geominfo = ('gl',{'lmax': phi_lmax})
        # geominfo = ('tgl',{'lmax': lmax, 'smax':2})
        # lenjob_geominfo = ('tgl',{'lmax': phi_lmax, 'smax':2})
        lldlm = np.arange(0, phi_lmax+1)
        synunl = Xunl(lmax=lmax, geominfo=geominfo, phi_lmax=phi_lmax)
        synsky = Xsky(lmax=lmax, unl_lib=synunl, geominfo=geominfo, lenjob_geominfo=geominfo, epsilon=epsilon)
        philm = synunl.get_sim_phi(0, space='alm')
        toydlm = hp.almxfl(philm, np.sqrt(np.arange(phi_lmax + 1, dtype=float) * np.arange(1, phi_lmax + 2)))
        toyunllm = synunl.get_sim_unl(0, spin=0, space='alm', field='temperature')
        Tsky = synsky.get_sim_sky(0, spin=0, space='map', field='temperature')
        Tsky2 = synsky.unl2len(toyunllm, philm, spin=0)

        for runinfo in runinfos:
            shttransformer_desc = 'shtns' if runinfo[0] == 'GPU' else 'ducc'
            kwargs = {
                'geominfo': geominfo,
                'nthreads': 10,
                'epsilon': epsilon,
                'verbosity': 1,
                'planned': False,
                'single_prec': False,
                'shttransformer_desc': shttransformer_desc
            }
            
            deflection_kwargs = {
                'geominfo': lenjob_geominfo,
                'nthreads': 10,
                'epsilon': epsilon,
                'verbosity': 1,
                'single_prec': False,
                'mmax_dlm': phi_lmax,
                'dlm': toydlm,
            } 
            print(runinfo)
            backend = runinfo[0]
            defres.update({backend: {}}) if backend not in defres.keys() else None
            solver = runinfo[1]
            defres[backend].update({solver : None}) if solver not in defres[backend].keys() else None
            for mode in ['nuFFT']:
                print("\nTesting:: solver = {} backend = {} mode = {} ...".format(solver, backend, mode))
                t = pysht.get_transformer(solver, mode, backend)
                t = t(**kwargs, deflection_kwargs=deflection_kwargs)
                print("\n----lmax: {}, phi_lmax: {}, dlm_lmax = {}, epsilon: {}----".format(lmax, phi_lmax, hp.Alm.getlmax(toydlm.size), deflection_kwargs['epsilon']))
                if backend == 'CPU':
                    if solver == 'lenspyx':
                        nalm = toyunllm.shape[-1]
                        gclm = np.array(np.zeros(shape=(1, nalm)), dtype=np.complex128) if not kwargs["single_prec"] else np.array(np.zeros(shape=(nalm)), dtype=np.complex64)
                        _ = Tsky2.copy()
                        print(_.shape)
                        wCPU = t.geom.weight
                        defres[backend][solver] = t.lenmap2gclm(
                            np.atleast_2d(_), dlm=toydlm, lmax=lmax, mmax=lmax, spin=0, gclm_out=gclm, nthreads=10, execmode='timing', ptg=None)
                    else:
                        wCPU = t.geom.weight
                        gclm = np.array(np.zeros(shape=(1, t.geom.nalm(lmax, lmax))), dtype=np.complex128) if not kwargs["single_prec"] else np.array(np.zeros(shape=(t.geom.nalm(lmax, lmax))), dtype=np.complex64)
                        _ = Tsky2.copy()
                        defres[backend][solver] = t.lenmap2gclm(
                                np.atleast_2d(_), dlm=toydlm, gclm_out=gclm, lmax=lmax, mmax=lmax, spin=0, nthreads=10, execmode='timing')
                elif backend == 'GPU':
                    wGPU = t.cc_transformer.constructor.gauss_wts()
                    lenmap = np.atleast_2d((Tsky2))#.reshape(t.geom.nph[0],-1).T.flatten()) # this only works for gl geom
                    lenmap = cp.array(lenmap, dtype=np.complex128) if not kwargs["single_prec"] else cp.array(lenmap.astype(np.complex64))
                    ll = np.arange(0,deflection_kwargs["mmax_dlm"]+1,1)
                    dlm_scaled = hp.almxfl(toydlm, np.nan_to_num(np.sqrt(1/(ll*(ll+1)))))
                    dlm_scaled = cp.array(np.atleast_2d(dlm_scaled), dtype=np.complex128) if not deflection_kwargs["single_prec"] else cp.array(np.atleast_2d(dlm_scaled).astype(np.complex64))
                    gclm = cp.array(np.zeros(shape=(1,t.geom.nalm(lmax, lmax))), dtype=np.complex128) if not kwargs["single_prec"] else cp.array(np.zeros(shape=(1,t.geom.nalm(lmax, lmax))), dtype=np.complex64)
                    defres[backend][solver] = t.lenmap2gclm(lenmap, dlm_scaled=dlm_scaled, lmax=lmax, mmax=lmax, gclm_out=gclm, nthreads=10, execmode='timing')