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

backends = ['{}'.format(sys.argv[1]) if len(sys.argv) > 1 else 'CPU']
print('backends: {}'.format(backends))
if backends == ['GPU']:
    solvers = ['cufinufft']
else:
    solvers = ['lenspyx']

sht_solver = 'shtns'
# lmaxs = np.array([n*256-1 for n in np.arange(1,15)])
lmaxs = np.array([2**n-1 for n in np.arange(int(sys.argv[2]),int(sys.argv[2])+1)])
for lmax in lmaxs:
    geominfo = ('gl',{'lmax':lmax})
    lldlm = np.arange(0,lmax+1024)
    synunl = Xunl(lmax=lmax, geominfo=geominfo)
    synsky = Xsky(lmax=lmax, unl_lib=synunl, geominfo=geominfo, lenjob_geominfo=geominfo)

    philm = synunl.get_sim_phi(0, space='alm')
    toydlm = hp.almxfl(philm, np.sqrt(lldlm*(lldlm+1)))

    toyunllm = synunl.get_sim_unl(0, spin=0, space='alm', field='temperature')
    # toyunllm = (np.random.random(size=hp.Alm.getsize(lmax, lmax)) +  np.random.random(size=hp.Alm.getsize(lmax, lmax))*1j)*1e-6
    # toydlm = (np.random.random(size=hp.Alm.getsize(lmax, lmax)) +  np.random.random(size=hp.Alm.getsize(lmax, lmax))*1j)*1e6
    
    deflection_kwargs = {
        'nthreads': 10,
        'dlm':toydlm,
        'mmax_dlm':lmax,
        'epsilon':1e-10,
        'verbosity':0,
        'single_prec':False,
        'planned':False,
        'lenjob_geominfo': geominfo,
        
    }
    geominfo_CAR = ('cc',{'nphi':2*(lmax+1), 'ntheta':lmax+1})
    defres = {}
    for backend in backends:
        if backend == 'GPU':
            solvers = ['cufinufft']
            sht_solver = 'shtns' # 'shtns'
        elif backend == 'CPU':
            solvers = ['duccnufft'] # duccnufft #lenspyx
            sht_solver = 'ducc' # 'shtns'
    for solver in solvers:
        for mode in ['nuFFT']:
            print("Testing solver={} backend={} mode={}...".format(solver, backend, mode))
            t = pysht.get_transformer(solver, mode, backend)
            t = t(sht_solver, geominfo, deflection_kwargs)
            # t.set_geometry(geominfo)
            print("\n----Testing function gclm2lenmap...----")
            print("\n----lmax: {}, epsilon: {}----".format(lmax, deflection_kwargs['epsilon']))
            if backend == 'CPU':
                if solver == 'lenspyx':
                    defres.update({
                        backend: t.gclm2lenmap(
                            toyunllm.copy(), dlm=toydlm, lmax=lmax, mmax=lmax, spin=0, nthreads=10, mode=1, ptg=None)})
                else:
                    defres.update({
                        backend: t.gclm2lenmap(
                            toyunllm.copy(), dlm=toydlm, lmax=lmax, mmax=lmax, spin=0, nthreads=10, mode=1)})
            else:
                # defres = t.gclm2lenmap(toyunllm.copy(), dlm=toydlm, lmax=lmax, mmax=lmax, spin=0, nthreads=10, mode=1)
                defres = t.gclm2lenmap_cupy(toyunllm.copy(), dlm=toydlm, lmax=lmax, mmax=lmax, spin=0, nthreads=10, mode=1)
            # FIXME after return, sometimes segmentation fault. Perhaps GPU not properly released
            # print('\n{} gclm2lenmap() time is: {:.3f} ms'.format(backends[0], (t2-t1)*100))