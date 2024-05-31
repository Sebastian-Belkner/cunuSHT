"""
Benchmark lenmap2gclm by scanning across different solvers, backends, and modes, and for different lmax values.
"""

import numpy as np
import healpy as hp
import cunusht
import cupy as cp
import sys
from lenspyx.lensing import get_geom as get_lenspyxgeom
from delensalot.sims.sims_lib import Xunl, Xsky

epsilons = [float(sys.argv[2])]
lmaxs = [512*int(sys.argv[1])-1]
runinfos = [("GPU", "cufinufft")] if sys.argv[3] == 'GPU' else [("CPU", "lenspyx")]
nthreads = 20
phi_lmaxs = [lmax for lmax in lmaxs]
defres = {}
for epsilon in epsilons:
    for runinfo in runinfos:
        for lmax, phi_lmax in zip(lmaxs, phi_lmaxs):
            lenjob_geominfo = ('gl',{'lmax': phi_lmax})
            lldlm = np.arange(0, phi_lmax+1)
            synunl = Xunl(lmax=lmax, geominfo=lenjob_geominfo, phi_lmax=phi_lmax)
            synsky = Xsky(lmax=lmax, unl_lib=synunl, geominfo=lenjob_geominfo, lenjob_geominfo=lenjob_geominfo, epsilon=epsilon)
            philm = synunl.get_sim_phi(0, space='alm')
            toydlm = hp.almxfl(philm, np.sqrt(np.arange(phi_lmax + 1, dtype=float) * np.arange(1, phi_lmax + 2)))
            toyunllm = synunl.get_sim_unl(0, spin=0, space='alm', field='temperature')
            Tsky2 = synsky.unl2len(toyunllm, philm, spin=0)
            for runinfo in runinfos:
                backend = runinfo[0]
                defres.update({backend: {}}) if backend not in defres.keys() else None
                solver = runinfo[1]

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
                        t = cunusht.get_transformer(backend, solver)
                        t = t(**kwargs)
                        
                        nalm = toyunllm.shape[-1]
                        gclm = np.array(np.zeros(shape=(1, nalm)), dtype=np.complex128) if not kwargs["single_prec"] else np.array(np.zeros(shape=(nalm)), dtype=np.complex64)
                        _ = Tsky2.copy()
                        defres[backend][solver] = t.lenmap2gclm(
                            np.atleast_2d(_), dlm=toydlm, lmax=lmax, mmax=lmax, spin=0, gclm_out=gclm, nthreads=nthreads, execmode='timing', epsilon=epsilon, ptg=None)
                    else:
                        kwargs = {
                            'geominfo_deflection': lenjob_geominfo,
                            'nuFFTtype': None,
                        }
                        t = cunusht.get_transformer(backend, solver)
                        t = t(**kwargs)
                        
                        gclm = np.array(np.zeros(shape=(1, t.deflectionlib.geom.nalm(lmax, lmax))), dtype=np.complex128) if epsilon<=1e-6 else np.array(np.zeros(shape=(t.deflectionlib.geom.nalm(lmax, lmax))), dtype=np.complex64)
                        _ = Tsky2.copy()
                        defres[backend][solver] = t.lenmap2gclm(
                                np.atleast_2d(_), dlm=toydlm, gclm_out=gclm, lmax=lmax, mmax=lmax, spin=0, nthreads=nthreads, epsilon=epsilon, execmode='timing')
                elif backend == 'GPU':
                    kwargs = {
                        'geominfo_deflection': lenjob_geominfo,
                        'epsilon': epsilon,
                        'nuFFTtype': 1,
                    }
                    t = cunusht.get_transformer(backend, solver)
                    t = t(**kwargs)
                    
                    lenmap = np.atleast_2d(Tsky2)
                    lenmap = cp.array(lenmap, dtype=np.complex128) if kwargs['epsilon']<=1e-6 else cp.array(lenmap.astype(np.complex64))
                    ll = np.arange(0,lmax+1,1)
                    dlm_scaled = hp.almxfl(toydlm, np.nan_to_num(np.sqrt(1/(ll*(ll+1)))))
                    dlm_scaled = cp.array(np.atleast_2d(dlm_scaled), dtype=np.complex128) if kwargs['epsilon']<=1e-6 else cp.array(np.atleast_2d(dlm_scaled).astype(np.complex64))
                    gclm = cp.array(np.zeros(shape=(1,t.geom.nalm(lmax, lmax))), dtype=np.complex128) if kwargs['epsilon']<=1e-6 else cp.array(np.zeros(shape=(1,t.geom.nalm(lmax, lmax))), dtype=np.complex64)
                    # defres[backend][solver] = t.lenmap2gclm(lenmap, dlm_scaled=dlm_scaled, lmax=lmax, mmax=lmax, gclm=gclm, epsilon=epsilon, execmode='timing', runid=int(sys.argv[4]))
                    defres[backend][solver] = t.lenmap2gclm(lenmap.get(), dlm_scaled=dlm_scaled, lmax=lmax, mmax=lmax, gclm=gclm, epsilon=epsilon, execmode='timing', runid=int(sys.argv[4]))