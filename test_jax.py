import numpy as np
import cunusht
from cunusht.utils import good_lmax_array, Alm
import jaxbind
from jax.test_util import check_grads


def _alm2realalm(alm, lmax, dtype, out=None):
    if out is None:
        out = np.empty((alm.shape[0], alm.shape[1] * 2 - lmax - 1), dtype=dtype)
    out[:, 0 : lmax + 1] = alm[:, 0 : lmax + 1].real
    out[:, lmax + 1 :] = alm[:, lmax + 1 :].view(dtype)
    out[:, lmax + 1 :] *= np.sqrt(2.0)
    return out


def _realalm2alm(alm, lmax, dtype, out=None):
    if out is None:
        out = np.empty((alm.shape[0], (alm.shape[1] + lmax + 1) // 2), dtype=dtype)
    out[:, 0 : lmax + 1] = alm[:, 0 : lmax + 1]
    out[:, lmax + 1 :] = alm[:, lmax + 1 :].view(dtype)
    out[:, lmax + 1 :] *= np.sqrt(2.0) / 2
    return out


def build_op(lmax, epsilon=1e-8, nthreads=1):
    nalm_unl = Alm.getsize(lmax, mmax=lmax)
    toydlm = np.random.rand(nalm_unl)*1e-6 + 1j*np.random.rand(nalm_unl)*1e-6
    kwargs = {
        'geominfo_deflection': ('gl',{'lmax': lmax}),
        'dglm': toydlm,
        'mmax_dlm': lmax,
        'nthreads': nthreads,
        'verbose': 0,
        'epsilon': epsilon,
        'single_prec': False
    }
    t = cunusht.get_transformer(backend="CPU")(**kwargs)

    # FIXME: the two lines below are simply to obtain the shape of the lenmap.
    # If there is a more direct way, let's use that instead!
    toyunllm = np.array([np.random.rand(nalm_unl)*1e-6 + 1j*np.random.rand(nalm_unl)*1e-6])
    mapshape = t.gclm2lenmap(toyunllm.copy(), dlm=toydlm, lmax=lmax, mmax=lmax, spin=0, nthreads=kwargs["nthreads"]).shape

    def fwd(out, args, kwargs_dump):
        out=out[0]
        (gclm,) = args
        gclm = np.atleast_2d(gclm.astype(np.float64))
        out[()] = t.gclm2lenmap(_realalm2alm(gclm,lmax,np.complex128), dlm=toydlm, lmax=lmax, mmax=lmax, spin=0, nthreads=kwargs["nthreads"])
    def adj(out, args, kwargs_dump):
        out=out[0]
        (lenmap,) = args
        lenmap = lenmap.astype(np.float64)
        gclm = np.zeros((nalm_unl,), dtype=np.complex128)
        out[()] = _alm2realalm(t.lenmap2gclm(np.atleast_2d(lenmap.copy()), dlm=toydlm, lmax=lmax, mmax=lmax, spin=0, gclm_out=gclm, nthreads=kwargs["nthreads"]),lmax,np.float64)
    def fwd_abstract(*args, **kwargs):
        return ((mapshape, np.float64),)
    def adj_abstract(*args, **kwargs):
        return (((nalm_unl*2-lmax-1,), np.float64),)

    return jaxbind.get_linear_call(
        fwd,
        adj,
        fwd_abstract,
        adj_abstract,
    )


lmax=1280
max_order=2
op = build_op(lmax)
nalm = Alm.getsize(lmax, mmax=lmax)
alm = _alm2realalm(np.array([np.random.rand(nalm)*1e-6 + 1j*np.random.rand(nalm)*1e-6]),lmax,np.float64)
check_grads(op, (alm,), order=max_order, modes=("fwd", "rev"), eps=1.0)

