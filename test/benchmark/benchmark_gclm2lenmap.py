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
    # ("CPU", "lenspyx"),
    # ("CPU", "duccnufft"),
    ("GPU", "cufinufft")
    ]
epsilons = [1e-10]
# lmaxs = [256*n-1 for n in np.arange(int(sys.argv[1]), 24)]
lmaxs = [256*int(sys.argv[1])-1]
phi_lmaxs = [lmax for lmax in lmaxs]
defres = {}
Tsky = None
Tsky2 = None
nthreads = 10

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

            backend = runinfo[0]
            defres.update({backend: {}}) if backend not in defres.keys() else None
            solver = runinfo[1]
            defres[backend].update({solver : None}) if solver not in defres[backend].keys() else None
            
            t = pysht.get_transformer(solver, backend)
            if backend == 'CPU':
                if solver == 'lenspyx':
                    kwargs = {
                        'geominfo_deflection': lenjob_geominfo,
                        'dglm': toydlm,
                        'mmax_dlm': lmax,
                        'nthreads': nthreads,
                        'verbosity': 1,
                        'epsilon': epsilon,
                        'single_prec': False,
                    }
                    t = t(**kwargs)
                    defres[backend][solver] = t.gclm2lenmap(
                            toyunllm.copy(), dlm=toydlm, lmax=lmax, mmax=lmax, spin=0, nthreads=10, execmode='timing', ptg=None)
                else:
                    kwargs = {
                        'geominfo_deflection': lenjob_geominfo,
                        'planned': False,
                    }
                    t = t(**kwargs)
                    defres[backend][solver] = t.gclm2lenmap(
                            gclm=toyunllm.copy(), dlm=toydlm, lmax=lmax, mmax=lmax, spin=0, nthreads=nthreads, epsilon=epsilon, execmode='timing', ptg=None)
            elif backend == 'GPU':
                kwargs = {
                    'geominfo_deflection': lenjob_geominfo,
                    'epsilon': epsilon,
                    'planned': True,
                }
                t = t(**kwargs)
                lenmap = cp.empty(t.deflectionlib.constructor.spat_shape, dtype=cp.complex128).flatten()
                ll = np.arange(0,lmax+1,1)
                dlm_scaled = hp.almxfl(toydlm, np.nan_to_num(np.sqrt(1/(ll*(ll+1)))))
                dlm_scaled = cp.array(np.atleast_2d(dlm_scaled), dtype=np.complex128) if kwargs['epsilon']<=1e-6 else cp.array(np.atleast_2d(dlm_scaled).astype(np.complex64))
                defres[backend][solver] = t.gclm2lenmap(cp.array(toyunllm.copy()), dlm_scaled=dlm_scaled, lmax=lmax, mmax=lmax, nthreads=nthreads, lenmap=lenmap, execmode='timing')
