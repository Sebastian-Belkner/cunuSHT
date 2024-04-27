"""
Benchmark gclm2lenmap by scanning across different solvers, backends, and modes, and for different lmax values.
"""
import numpy as np
import time
import healpy as hp
import pysht
import sys
from time import process_time
from delensalot.sims.sims_lib import Xunl, Xsky
import cupy as cp

runinfos = [
    # ("CPU", "lenspyx", 'ducc'),
    # ("CPU", "duccnufft", 'ducc'),
    ("GPU", "cufinufft", 'shtns')
    ]
epsilons = [1e-6]
# lmaxs = [256*n-1 for n in np.arange(int(sys.argv[1]), 24)]
lmaxs = [256*int(sys.argv[1])-1]
phi_lmaxs = [lmax for lmax in lmaxs]
defres = {}
Tsky = None
Tsky2 = None
for epsilon in epsilons:
    for runinfo in runinfos:
        for lmax, phi_lmax in zip(lmaxs, phi_lmaxs):
            geominfo = ('gl',{'lmax':lmax})
            lenjob_geominfo = ('gl',{'lmax':phi_lmax})
            lldlm = np.arange(0,phi_lmax+1)
            synunl = Xunl(lmax=lmax, geominfo=geominfo, phi_lmax=phi_lmax)
            philm = synunl.get_sim_phi(0, space='alm')
            toydlm = hp.almxfl(philm, np.sqrt(lldlm*(lldlm+1)))
            toyunllm = synunl.get_sim_unl(0, spin=0, space='alm', field='temperature')

            kwargs = {
                'geominfo': geominfo,
                'nthreads': 10,
                'epsilon': epsilon,
                'verbosity': 1,
                'planned': False,
                "plannednuFFT": False,
                'single_prec': False,
                'shttransformer_desc': runinfo[2]
            }
            
            deflection_kwargs = {
                'dlm': toydlm,
                'mmax_dlm': phi_lmax,
                'epsilon': epsilon,
                'verbosity': 1,
                'single_prec': False,
                'nthreads': 10,
                'geominfo': lenjob_geominfo,
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
                # print(t.constructor.spat_shape, lmax)
                print("\n----lmax: {}, phi_lmax: {}, dlm_lmax = {}, epsilon: {}----".format(lmax, phi_lmax, hp.Alm.getlmax(toydlm.size), deflection_kwargs['epsilon']))
                if backend == 'CPU':
                    if solver == 'lenspyx':
                        defres[backend][solver] = t.gclm2lenmap(
                                toyunllm.copy(), dlm=toydlm, lmax=lmax, mmax=lmax, spin=0, nthreads=10, execmode='timing', ptg=None)
                    else:
                        defres[backend][solver] = t.gclm2lenmap(
                                toyunllm.copy(), dlm=toydlm, lmax=lmax, mmax=lmax, spin=0, nthreads=10, execmode='timing', ptg=None)
                elif backend == 'GPU':
                    lenmap = cp.empty(t.constructor.spat_shape, dtype=cp.complex128)
                    ll = np.arange(0,deflection_kwargs["mmax_dlm"]+1,1)
                    dlm_scaled = hp.almxfl(toydlm, np.nan_to_num(np.sqrt(1/(ll*(ll+1)))))
                    dlm_scaled = cp.array(np.atleast_2d(dlm_scaled), dtype=np.complex128) if not deflection_kwargs["single_prec"] else cp.array(np.atleast_2d(dlm_scaled).astype(np.complex64))
                    defres[backend][solver] = t.gclm2lenmap(cp.array(toyunllm.copy()), dlm_scaled=dlm_scaled, lmax=lmax, mmax=lmax, nthreads=10, lenmap=lenmap, execmode='timing')
