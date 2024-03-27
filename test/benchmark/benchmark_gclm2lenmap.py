"""
Benchmark gclm2lenmap by scanning across different solvers, backends, and modes, and for different lmax values.
"""
import numpy as np
import time
import healpy as hp
import pysht
import sys
from time import process_time

backends = ['{}'.format(sys.argv[1]) if len(sys.argv) > 1 else 'CPU']
print('backends: {}'.format(backends))
if backends == ['GPU']:
    solvers = ['cufinufft']
else:
    solvers = ['finufft']

sht_solver = 'shtns' # shtns
# lmaxs = np.array([n*256-1 for n in np.arange(1,15)])
lmaxs = np.array([2**n-1 for n in np.arange(5,13)])
# lmaxs = [2*2048-1]
for lmax in lmaxs:
    toyskylm = np.random.random(size=hp.Alm.getsize(lmax, lmax)) +  np.random.random(size=hp.Alm.getsize(lmax, lmax))*1j
    toydlm = np.random.random(size=hp.Alm.getsize(lmax, lmax)) +  np.random.random(size=hp.Alm.getsize(lmax, lmax))*1j
    
    deflection_kwargs = {
        'nthreads': 4,
        'dlm':toydlm,
        'mmax_dlm':lmax,
        'epsilon':1e-7,
        'verbosity':0,
        'single_prec':False,
        'planned':False
    }
    geominfo = ('gl',{'lmax':lmax}) #if sht_solver == 'shtns' else ('cc',{'nphi':4116, 'ntheta':2056})
    for solver in solvers:
        for backend in backends:
            for mode in ['nuFFT']:
                print("Testing solver={} backend={} mode={}...".format(solver, backend, mode))
                t = pysht.get_transformer(solver, mode, backend)
                t = t(sht_solver, geominfo, deflection_kwargs)
                t_gpu = t
                # t.set_geometry(geominfo)
                print("\n----Testing function gclm2lenmap...----")
                # t1 = process_time()
                geominfo_cc = ('cc',{'nphi':2*(lmax+1), 'ntheta':lmax+1})
                tcc = pysht.get_transformer('shtns', 'SHT', 'GPU')(geominfo_cc)
                defres = t.gclm2lenmap(gclm=toyskylm, dlm=toyskylm, lmax=lmax, mmax=lmax, spin=0, nthreads=4, cc_transformer=tcc)
                # t2 = process_time()
                # FIXME after return, sometimes segmentation fault. Perhaps GPU not properly released
                print(defres)
                # print('\n{} gclm2lenmap() time is: {:.3f} ms'.format(backends[0], (t2-t1)*100))