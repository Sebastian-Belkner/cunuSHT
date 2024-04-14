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

# backends = ['{}'.format(sys.argv[1]) if len(sys.argv) > 1 else 'CPU']
# lmaxs = [int(sys.argv[2])]


runinfos = [
    # ("CPU", "lenspyx"),
    # ("CPU", "duccnufft"),
    ("GPU", "cufinufft")
]

lmaxs = [256*n-1 for n in np.arange(12, 15)]
phi_lmaxs = [lmax for lmax in lmaxs]
defres = {}
Tsky = None
Tsky2 = None
for runinfo in runinfos:
    for lmax, phi_lmax in zip(lmaxs, phi_lmaxs):
        geominfo = ('gl',{'lmax':lmax})
        lenjob_geominfo = ('gl',{'lmax':phi_lmax})
        lldlm = np.arange(0,phi_lmax+1)
        synunl = Xunl(lmax=lmax, geominfo=geominfo, phi_lmax=phi_lmax)
        synsky = Xsky(lmax=lmax, unl_lib=synunl, geominfo=geominfo, lenjob_geominfo=geominfo, epsilon=1e-10)
        philm = synunl.get_sim_phi(0, space='alm')
        toydlm = hp.almxfl(philm, np.sqrt(lldlm*(lldlm+1)))
        toyunllm = synunl.get_sim_unl(0, spin=0, space='alm', field='temperature')

        shttransformer_desc = 'shtns' if runinfo[0] == 'GPU' else 'ducc'
        kwargs = {
            'geominfo': geominfo,
            'nthreads': 10,
            'epsilon': 1e-10,
            'verbosity': 0,
            'planned': False,
            'single_prec': False,
            'shttransformer_desc': shttransformer_desc
        }
        
        deflection_kwargs = {
            'dlm': toydlm,
            'mmax_dlm': phi_lmax,
            'epsilon': 1e-10,
            'verbosity': 0,  
            'single_prec': False,
            'nthreads': 10,
            'geominfo': lenjob_geominfo,
        }
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
                    defres[backend][solver] = t.gclm2lenmap(
                            toyunllm.copy(), dlm=toydlm, lmax=lmax, mmax=lmax, spin=0, nthreads=10, execmode='timing', ptg=None)
                else:
                    defres[backend][solver] = t.gclm2lenmap(
                            toyunllm.copy(), dlm=toydlm, lmax=lmax, mmax=lmax, spin=0, nthreads=10, execmode='timing')
            elif backend == 'GPU':
                defres[backend][solver] = t.gclm2lenmap_cupy(toyunllm.copy(), dlm=toydlm, lmax=lmax, mmax=lmax, spin=0, nthreads=10, execmode='timing')