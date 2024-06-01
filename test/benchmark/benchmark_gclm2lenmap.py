"""
Benchmark gclm2lenmap by scanning across different solvers, backends, and modes, and for different lmax values.
"""
import numpy as np
import time
import healpy as hp
import cunusht
import sys
from time import process_time

runinfos = [
    # ("CPU", "lenspyx"),
    # ("CPU", "duccnufft"),
    ("GPU", "cufinufft")
    ]
epsilons = [float(sys.argv[2])]
# lmaxs = [512*n-1 for n in np.arange(int(sys.argv[1]), 24)]
lmaxs = [512*int(sys.argv[1])-1]
runinfos = [("GPU", "cufinufft")] if sys.argv[3] == 'GPU' else [("CPU", "lenspyx")]
phi_lmaxs = [lmax for lmax in lmaxs]
defres = {}
Tsky = None
Tsky2 = None
nthreads = 20

for epsilon in epsilons:
    for runinfo in runinfos:
        for lmax, phi_lmax in zip(lmaxs, phi_lmaxs):
            geominfo = ('gl',{'lmax':lmax})
            lenjob_geominfo = ('gl',{'lmax':phi_lmax})
            lldlm = np.arange(0,phi_lmax+1)
            if False:
                from delensalot.sims.sims_lib import Xunl, Xsky
                synunl = Xunl(lmax=lmax, geominfo=geominfo, phi_lmax=phi_lmax)
                philm = synunl.get_sim_phi(0, space='alm')
                toydlm = hp.almxfl(philm, np.sqrt(lldlm*(lldlm+1)))
                toyunllm = synunl.get_sim_unl(0, spin=0, space='alm', field='temperature')
            else:
                nalm_unl = hp.Alm.getsize(lmax, mmax=lmax)
                toyunllm = np.array([np.random.rand(nalm_unl)*1e-6 + 1j*np.random.rand(nalm_unl)*1e-6])
                toydlm = np.random.rand(nalm_unl)*1e-6 + 1j*np.random.rand(nalm_unl)*1e-6

            backend = runinfo[0]
            defres.update({backend: {}}) if backend not in defres.keys() else None
            solver = runinfo[1]
            defres[backend].update({solver : None}) if solver not in defres[backend].keys() else None
            
            t = cunusht.get_transformer(backend, solver)
            if backend == 'CPU':
                if solver == 'lenspyx':
                    kwargs = {
                        'geominfo_deflection': lenjob_geominfo,
                        'dglm': toydlm,
                        'mmax_dlm': lmax,
                        'nthreads': nthreads,
                        'verbose': 1,
                        'epsilon': epsilon,
                        'single_prec': False if epsilon<1e-6 else True
                    }
                    t = t(**kwargs)
                    defres[backend][solver] = t.gclm2lenmap(
                            toyunllm.copy(), dlm=toydlm, lmax=lmax, mmax=lmax, spin=0, nthreads=nthreads, execmode='timing', ptg=None)
                else:
                    kwargs = {
                        'geominfo_deflection': lenjob_geominfo,
                        'nuFFTtype': 2,
                    }
                    t = t(**kwargs)
                    defres[backend][solver] = t.gclm2lenmap(
                            gclm=toyunllm.copy(), dlm=toydlm, lmax=lmax, mmax=lmax, spin=0, nthreads=nthreads, epsilon=epsilon, execmode='timing', ptg=None)
            elif backend == 'GPU':
                import cupy as cp
                kwargs = {
                    'geominfo_deflection': lenjob_geominfo,
                    'epsilon': epsilon,
                    'nuFFTtype': 2,
                }
                t = t(**kwargs)
                lenmap = cp.empty(t.deflectionlib.constructor.spat_shape, dtype=cp.complex128).flatten()
                ll = np.arange(0,lmax+1,1)
                dlm_scaled = hp.almxfl(toydlm, np.nan_to_num(np.sqrt(1/(ll*(ll+1)))))
                dlm_scaled = cp.array(np.atleast_2d(dlm_scaled), dtype=np.complex128) # must always be double precision since nuFFT is always run in double precision
                # defres[backend][solver] = t.gclm2lenmap(cp.array(toyunllm.copy()), dlm_scaled=dlm_scaled, lmax=lmax, mmax=lmax, lenmap=lenmap, ptg=None, execmode='timing', runid=int(sys.argv[4]))
                defres[backend][solver] = t.gclm2lenmap(toyunllm.copy(), dlm_scaled=dlm_scaled, lmax=lmax, mmax=lmax, lenmap=lenmap, ptg=None, execmode='timing', runid=int(sys.argv[4]))
