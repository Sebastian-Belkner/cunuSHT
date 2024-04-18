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
    # ("CPU", "lenspyx"),
    # ("CPU", "duccnufft"),
    ("GPU", "cufinufft")
]
epsilons = [1e-5]
lmaxs = [256*n-1 for n in np.arange(14, 20)]
phi_lmaxs = [lmax for lmax in lmaxs]
defres = {}
Tsky = None
Tsky2 = None
print(lmaxs)
for epsilon in epsilons:
    for lmax, phi_lmax in zip(lmaxs, phi_lmaxs):
        try:
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
            # toyunllm = (np.random.random(size=hp.Alm.getsize(lmax, lmax)) +  np.random.random(size=hp.Alm.getsize(lmax, lmax))*1j)*1e-6
            # toydlm = (np.random.random(size=hp.Alm.getsize(lmax, lmax)) +  np.random.random(size=hp.Alm.getsize(lmax, lmax))*1j)*1e-6
            for runinfo in runinfos:
                shttransformer_desc = 'shtns' if runinfo[0] == 'GPU' else 'ducc'
                kwargs = {
                    'geominfo': geominfo,
                    'nthreads': 10,
                    'epsilon': epsilon,
                    'verbosity': 1,
                    'planned': False,
                    'single_prec': True,
                    'shttransformer_desc': shttransformer_desc
                }
                
                deflection_kwargs = {
                    'geominfo': lenjob_geominfo,
                    'nthreads': 10,
                    'epsilon': epsilon,
                    'verbosity': 1,
                    'single_prec': True,
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
                    # if runinfo[0] == 'CPU':
                        # magnific = t.deflectionlib.dlm2A()
                    print(((lmax+1)*(lmax+1)), toyunllm.shape)
                    print("\n----lmax: {}, phi_lmax: {}, dlm_lmax = {}, epsilon: {}----".format(lmax, phi_lmax, hp.Alm.getlmax(toydlm.size), deflection_kwargs['epsilon']))
                    if backend == 'CPU':
                        if solver == 'lenspyx':
                            defres[backend][solver] = t.lenmap2gclm(
                                    np.atleast_2d(Tsky2.astype(np.float32)), dlm=toydlm, lmax=lmax, mmax=lmax, spin=0, nthreads=10, execmode='timing', ptg=None)
                        else:
                            defres[backend][solver] = t.lenmap2gclm(
                                    np.atleast_2d(Tsky2.astype(np.float32)), dlm=toydlm, lmax=lmax, mmax=lmax, spin=0, nthreads=10, execmode='timing')
                    elif backend == 'GPU':
                        defres[backend][solver] = t.lenmap2gclm(np.atleast_2d(Tsky2.astype(np.float32)), dlm=toydlm, lmax=lmax, mmax=lmax, spin=0, nthreads=10, execmode='timing')
        except Exception as e:
            print(e)
            sys.exit()
            # continue
    