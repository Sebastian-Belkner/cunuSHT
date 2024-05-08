"""
Follows https://sepwww.stanford.edu/sep/prof/pvi/conj/paper_html/node9.html#:~:text=The%20dot%20product%20test%20is,are%20adjoint%20to%20each%20other.&text=your%20program%20for-,.,scalars%20that%20should%20be%20equal.&text=(unless%20the%20random%20numbers%20d

python3 -m unittest test_adjointOperator.py
"""
import unittest

import numpy as np
import cupy as cp
import shtns
import healpy as hp
import cunusht
import ducc0
from numpy.testing import assert_allclose

from lenspyx.utils_hp import alm_copy
from cunusht.geometry import get_geom
from cunusht.deflection.CPU_nufft_transformer import deflection

from ducc0.sht.experimental import adjoint_synthesis_general, synthesis_general
from delensalot.sims.sims_lib import Xunl, Xsky


def nalm(lmax, mmax):
    return ((mmax+1)*(mmax+2))//2 + (mmax+1)*(lmax-mmax)


def random_alm(lmax, mmax, spin, ncomp, rng):
    res = rng.uniform(-1., 1., (ncomp, nalm(lmax, mmax))) \
     + 1j*rng.uniform(-1., 1., (ncomp, nalm(lmax, mmax)))
    # make a_lm with m==0 real-valued
    res[:, 0:lmax+1].imag = 0.
    ofs=0
    for s in range(spin):
        res[:, ofs:ofs+spin-s] = 0.
        ofs += lmax+1-s
    return res


def compress_alm(alm, lmax):
    res = np.empty(2*len(alm)-lmax-1, dtype=np.float64)
    res[0:lmax+1] = alm[0:lmax+1].real
    res[lmax+1::2] = np.sqrt(2)*alm[lmax+1:].real
    res[lmax+2::2] = np.sqrt(2)*alm[lmax+1:].imag
    return res


def myalmdot(a1, a2, lmax):
    return ducc0.misc.vdot(compress_alm(a1, lmax), compress_alm((a2), lmax))

test_cases = [ 
    (lmax, lmax) for lmax in [2**n-1 for n in np.arange(6, 8)]
]

def contract_almxblm(alm, blm):
    '''
    Return sum_lm alm x conj(blm), i.e. the sum of the Hadamard product of two
    sets of spherical harmonic coefficients corresponding to real fields.

    Parameters
    ----------
    alm : (..., nelem) complex array
        Healpix-ordered (m-major) alm array.
    blm : (..., nelem) complex array
        Healpix ordered (m-major) alm array.

    Returns
    -------
    had_sum : float
        Sum of Hadamard product (real valued).

    Raises
    ------
    ValueError
        If input arrays have different shapes.
    '''

    if blm.shape != alm.shape:
        raise ValueError('Shape alm ({}) != shape blm ({})'.
                         format(alm.shape, blm.shape))

    lmax = hp.Alm.getlmax(alm.shape[-1])
    blm = np.conj(blm)
    csum = complex(np.tensordot(alm, blm, axes=alm.ndim))
    had_sum = 2 * np.real(csum)

    # We need to subtract the m=0 elements once.
    had_sum -= np.real(np.sum(alm[...,:lmax+1] * blm[...,:lmax+1]))

    return had_sum

runinfos = [
    # ("CPU", "lenspyx", 'ducc'),
    # ("CPU", "duccnufft", 'ducc'),
    ("GPU", "cufinufft", 'shtns')
    ]

class TestUnit(unittest.TestCase):
    
    @unittest.skip("Skipping this test method for now")
    def test_unit_g2l(self):
        """Compares the dot product of two arrays
            y'(A x) = (A' y)' x
        """
        for test_case in test_cases:
            with self.subTest(input_value=test_case):
                solver = runinfos[0][1]
                mode = 'nuFFT'
                backend = runinfos[0][0]
                
                lmax = test_case[0]
                mmax = lmax
                phi_lmax = lmax
                geominfo = ('gl',{'lmax': lmax})
                lenjob_geominfo = ('gl',{'lmax': phi_lmax})
                epsilon = 1e-8
                deflection_epsilon = epsilon
                shttransformer_desc = runinfos[0][2]

                lldlm = np.arange(0, phi_lmax+1)
                synunl = Xunl(lmax=lmax, geominfo=geominfo, phi_lmax=phi_lmax)
                synsky = Xsky(lmax=lmax, unl_lib=synunl, geominfo=geominfo, lenjob_geominfo=geominfo, epsilon=epsilon)
                philm = synunl.get_sim_phi(0, space='alm')
                toydlm = hp.almxfl(philm, np.sqrt(np.arange(phi_lmax + 1, dtype=float) * np.arange(1, phi_lmax + 2)))
                
                kwargs = {
                    'geominfo': geominfo,
                    'nthreads': 10,
                    'epsilon': epsilon,
                    'verbosity': 1,
                    'planned': False,
                    'single_prec': False,
                    'shttransformer_desc': shttransformer_desc
                }
                
                deflection_kwargs = {
                    'geominfo': lenjob_geominfo,
                    'nthreads': 10,
                    'epsilon': deflection_epsilon,
                    'verbosity': 1,
                    'single_prec': False,
                    'mmax_dlm': phi_lmax, # this is merely to cut out the m-modes.
                    'dlm': toydlm,
                }
                # This is (A x)
                toyunllm = synunl.get_sim_unl(0, spin=0, space='alm', field='temperature')
                Tsky = np.atleast_2d(synsky.unl2len(toyunllm, philm, spin=0))
                
                
                # This is for (A' y): generate random data
                toyunllm_other = synunl.get_sim_unl(1, spin=0, space='alm', field='temperature')
                Tsky_other = np.atleast_2d(synsky.unl2len(toyunllm_other, philm, spin=0))
                Tsky_other = Tsky_other.astype(np.complex128) if not kwargs["single_prec"] else Tsky_other.astype(np.complex64)

                ll = np.arange(0,deflection_kwargs["mmax_dlm"]+1,1)
                dlm_scaled = hp.almxfl(toydlm, np.nan_to_num(np.sqrt(1/(ll*(ll+1)))))
                dlm_scaled = cp.array(np.atleast_2d(dlm_scaled), dtype=np.complex128) if not deflection_kwargs["single_prec"] else cp.array(np.atleast_2d(dlm_scaled).astype(np.complex64))
                t = cunusht.get_transformer(solver, mode, backend)
                t = t(**kwargs, deflection_kwargs=deflection_kwargs)
                
                nalm = ((mmax+1)*(mmax+2))//2 + (mmax+1)*(lmax-mmax)
                # This is A' y
                if runinfos[0][0] == 'CPU':
                    gclm = np.zeros(shape=(1,nalm), dtype=np.complex128) if not kwargs["single_prec"] else np.zeros(shape=(1,nalm), dtype=np.complex64)
                    gclm = t.lenmap2gclm(Tsky_other.astype(np.float64), dlm=toydlm, lmax=lmax, mmax=lmax, spin=0, gclm_out=gclm, nthreads=10, execmode='normal')
                else:
                    gclm = cp.array(np.zeros(shape=(1,nalm)), dtype=np.complex128) if not kwargs["single_prec"] else cp.array(np.zeros(shape=(1,nalm)), dtype=np.complex64)
                    t.lenmap2gclm(cp.array(Tsky_other), dlm_scaled=dlm_scaled, lmax=lmax, mmax=lmax, spin=0, gclm_out=gclm, nthreads=10, execmode='normal')
                    gclm = gclm.get()
                
                # Now, compare y'(A x) = (A' y)'x
                lhs = np.dot(Tsky[0], Tsky_other[0])
                rhs = contract_almxblm(gclm[0], toyunllm)
                
                print(lhs, rhs)
                
    @unittest.skip("Skipping this test method for now")     
    def test_unit_synthgen(self):
        """Compares the dot product of two arrays, here for DUCC's synthesis general
            y'(A x) = (A' y)' x
        """
        
        # settings
        sht_mode = 'STANDARD'
        epsilon = 1e-14
        verbosity = 1
        nthreads = 10
        spin = 0
        
        lmax_unl = 2048
        mmax_unl = lmax_unl
        geominfo = ('gl',{'lmax': lmax_unl})
        geom = get_geom(geominfo)
        npix = geom.npix()
        # ptg = np.random.rand(npix, 2)*1e-6
        nalm_unl = hp.Alm.getsize(lmax_unl, mmax=mmax_unl)
        dlm = np.random.rand(nalm_unl)*1e-32 + 1j*np.random.rand(nalm_unl)*1e-32
        dlm *= 0.
        
        deflection_kwargs = {
            'geominfo': geominfo,
            'nthreads': 10,
            'epsilon': epsilon,
            'verbosity': 1,
            'single_prec': False,
            'mmax_dlm': lmax_unl, # this is merely to cut out the m-modes.
            'dlm': dlm,
            "shttransformer_desc": 'ducc'
        }
        ptg = np.array(deflection(**deflection_kwargs).dlm2pointing(dlm)).T
        
        x = np.atleast_2d(np.random.rand((nalm_unl)) + 1j*np.random.rand((nalm_unl)))
        ll = np.ones(shape=(lmax_unl+1))
        x = np.atleast_2d(hp.synalm(ll, lmax=lmax_unl, mmax=mmax_unl))
        # TODO change to random_alms
        y_tilde = np.empty((1,npix))
        print("shapes: ", x.shape, y_tilde.shape, ptg.shape)
        synthesis_general(
            map=y_tilde, lmax=lmax_unl, mmax=mmax_unl, alm=x, loc=ptg,
            spin=spin, epsilon=epsilon, nthreads=nthreads, mode=sht_mode,
            verbose=verbosity)
        
        lmax_len = lmax_unl + 0
        mmax_len = lmax_unl + 0
        y = np.atleast_2d(np.random.rand(npix))
        y_prime = np.conj(y)
        nalm_len = hp.Alm.getsize(lmax_len, mmax=mmax_len)
        x_tilde = np.empty((1,nalm_len), dtype=complex)
        print("shapes: ", x_tilde.shape, y_prime.shape, ptg.shape)
        adjoint_synthesis_general(
            lmax=lmax_len, mmax=mmax_len, map=y, loc=ptg, 
            spin=spin, epsilon=epsilon, nthreads=nthreads, mode=sht_mode,
            alm=x_tilde, verbose=verbosity)
        
        ncomp = 1
        v2 = ducc0.misc.vdot(y.real, y_tilde.real) + ducc0.misc.vdot(y.imag, y_tilde.imag) 
        v1 = np.sum([myalmdot(x[c, :], x_tilde[c, :], lmax_len)
                    for c in range(ncomp)])
        assert_allclose(v1, v2, rtol=1e-9)
        
        print(v2, v1)
        lhs = np.dot(y_prime[0], y_tilde[0])
        rhs = contract_almxblm(x_tilde[0], x[0])
        
        print(lhs, rhs)
   
    # @unittest.skip("Skipping this test method for now")
    def test_adjointness_general(self):
        lmmax, npix, spin, nthreads = (1024, 1024), 20000, 0, 10
        rng = np.random.default_rng(48)

        lmax, mmax = lmmax
        epsilon = 1e-8
        ncomp = 1 if spin == 0 else 2
        slm1 = random_alm(lmax, mmax, spin, ncomp, rng)
        loc = rng.uniform(0., 1., (npix,2))
        loc[:, 0] *= np.pi
        loc[:, 1] *= 2*np.pi
        points2 = rng.uniform(-0.5, 0.5, (loc.shape[0],ncomp)).T
        
        print("parameters: ", lmax, mmax, epsilon, ncomp, spin, nthreads)
        points1 = ducc0.sht.synthesis_general(lmax=lmax, mmax=mmax, alm=slm1, loc=loc, spin=spin, epsilon=epsilon, nthreads=nthreads)
        
        print("adjoint parameters: ", lmax, mmax, epsilon, ncomp, spin, nthreads)
        slm2 = ducc0.sht.adjoint_synthesis_general(lmax=lmax, mmax=mmax, map=points2, loc=loc, spin=spin, epsilon=epsilon, nthreads=nthreads)
        v1 = np.sum([myalmdot(slm1[c, :], slm2[c, :], lmax)
                    for c in range(ncomp)])
        v2 = ducc0.misc.vdot(points2.real, points1.real) + ducc0.misc.vdot(points2.imag, points1.imag) 
        assert_allclose(v1, v2, rtol=1e-9)
        print(v1, v2)

        if spin > 0:
            points1 = ducc0.sht.synthesis_general(lmax=lmax, mmax=mmax, alm=slm1[:1], loc=loc, spin=spin, epsilon=epsilon, nthreads=nthreads, mode="GRAD_ONLY")
            slm2 = ducc0.sht.adjoint_synthesis_general(lmax=lmax, mmax=mmax, map=points2, loc=loc, spin=spin, epsilon=epsilon, nthreads=nthreads, mode="GRAD_ONLY")
            v1 = np.sum([myalmdot(slm1[c, :], slm2[c, :], lmax)
                        for c in range(1)])
            v2 = ducc0.misc.vdot(points2.real, points1.real) + ducc0.misc.vdot(points2.imag, points1.imag) 
            assert_allclose(v1, v2, rtol=1e-9)

class TestIntegration(unittest.TestCase):
    
    @unittest.skip("Skipping this test method for now")
    def test_integration_C2Candpointing2nuFFT(self):
        for test_case in test_cases:
            with self.subTest(input_value=test_case):
                pass

if __name__ == '__main__':
    unittest.main()