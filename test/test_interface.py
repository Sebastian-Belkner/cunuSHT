import numpy as np
import cunusht
from cunusht.utils import good_lmax_array, Alm

lmax = good_lmax_array[15]
print(f"lmax is {lmax}")
nalm_unl = Alm.getsize(lmax, mmax=lmax)
toyunllm = np.array([np.random.rand(nalm_unl)*1e-6 + 1j*np.random.rand(nalm_unl)*1e-6])
toydlm = np.random.rand(nalm_unl)*1e-6 + 1j*np.random.rand(nalm_unl)*1e-6
kwargs = {
    'geominfo_deflection': ('gl',{'lmax': lmax}),
    'dglm': toydlm,
    'mmax_dlm': lmax,
    'nthreads': 20,
    'verbose': 0,
    'epsilon': 1e-08,
    'single_prec': False
}
t = cunusht.get_transformer(backend="CPU")(**kwargs)

lenmap = t.gclm2lenmap(toyunllm.copy(), dlm=toydlm, lmax=lmax, mmax=lmax, spin=0, nthreads=kwargs["nthreads"])
print(f"This is your lenmap: {lenmap}")

gclm = np.zeros_like(toyunllm)
gclm = t.lenmap2gclm(np.atleast_2d(lenmap.copy()), dlm=toydlm, lmax=lmax, mmax=lmax, spin=0, gclm_out=gclm, nthreads=kwargs["nthreads"])
print(f"This is your gclm: {gclm}")