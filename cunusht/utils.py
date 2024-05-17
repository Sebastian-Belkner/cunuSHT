import time, os
from datetime import timedelta
import numpy as np
import sys
import json

from numpy.random import default_rng
rng = default_rng()

class Alm:
    """alm arrays useful statics. Directly from healpy but excluding keywords


    """
    @staticmethod
    def getsize(lmax:int, mmax:int):
        """Number of entries in alm array with lmax and mmax parameters

        Parameters
        ----------
        lmax : int
          The maximum multipole l, defines the alm layout
        mmax : int
          The maximum quantum number m, defines the alm layout

        Returns
        -------
        nalm : int
            The size of a alm array with these lmax, mmax parameters

        """
        return ((mmax+1) * (mmax+2)) // 2 + (mmax+1) * (lmax-mmax)

    @staticmethod
    def getidx(lmax:int, l:int or np.ndarray, m:int or np.ndarray):
        """Returns index corresponding to (l,m) in an array describing alm up to lmax.

        In HEALPix C++ and healpy, :math:`a_{lm}` coefficients are stored ordered by
        :math:`m`. I.e. if :math:`\ell_{max}` is 16, the first 16 elements are
        :math:`m=0, \ell=0-16`, then the following 15 elements are :math:`m=1, \ell=1-16`,
        then :math:`m=2, \ell=2-16` and so on until the last element, the 153th, is
        :math:`m=16, \ell=16`.

        Parameters
        ----------
        lmax : int
          The maximum l, defines the alm layout
        l : int
          The l for which to get the index
        m : int
          The m for which to get the index

        Returns
        -------
        idx : int
          The index corresponding to (l,m)
        """
        return m * (2 * lmax + 1 - m) // 2 + l

    @staticmethod
    def getlmax(s:int, mmax:int or None):
        """Returns the lmax corresponding to a given healpy array size.

        Parameters
        ----------
        s : int
          Size of the array
        mmax : int
          The maximum m, defines the alm layout

        Returns
        -------
        lmax : int
          The maximum l of the array, or -1 if it is not a valid size.
        """
        if mmax is not None and mmax >= 0:
            x = (2 * s + mmax ** 2 - mmax - 2) / (2 * mmax + 2)
        else:
            x = (-3 + np.sqrt(1 + 8 * s)) / 2
        if x != np.floor(x):
            return -1
        else:
            return int(x)

def almxfl(alm:np.ndarray, fl:np.ndarray, mmax:int or None, inplace:bool):
    """Multiply alm by a function of l.

    Parameters
    ----------
    alm : array
      The alm to multiply
    fl : array
      The function (at l=0..fl.size-1) by which alm must be multiplied.
    mmax : None or int
      The maximum m defining the alm layout. Default: lmax.
    inplace : bool
      If True, modify the given alm, otherwise make a copy before multiplying.

    Returns
    -------
    alm : array
      The modified alm, either a new array or a reference to input alm,
      if inplace is True.

    """
    lmax = Alm.getlmax(alm.size, mmax)
    if mmax is None or mmax < 0:
        mmax = lmax
    assert fl.size > lmax, (fl.size, lmax)
    if inplace:
        for m in range(mmax + 1):
            b = m * (2 * lmax + 1 - m) // 2 + m
            alm[b:b + lmax - m + 1] *= fl[m:lmax+1]
        return
    else:
        ret = np.empty_like(alm)
        for m in range(mmax + 1):
            b = m * (2 * lmax + 1 - m) // 2 + m
            ret[b:b + lmax - m + 1] = alm[b:b + lmax - m + 1] * fl[m:lmax+1]
        return ret

def synalm(cl:np.ndarray, lmax:int, mmax:int or None, rlm_dtype=np.float64):
    """Creates a Gaussian field alm from input cl array

    Parameters
    ----------
    cl : ndarray
        The power spectrum of the map
    lmax : int
        Maximum multipole simulated
    mmax: int
        Maximum m defining the alm layout, defaults to lmax if None or < 0
    rlm_dtype(optional, defaults to np.float64):
        Precision of real components of the array (e.g. np.float32 for single precision output array)

    Returns
    -------
    alm: ndarray
        harmonic coefficients of Gaussian field with lmax, mmax parameters

    """
    assert lmax + 1 <= cl.size
    if mmax is None or mmax < 0:
        mmax = lmax
    alm_size = Alm.getsize(lmax, mmax)
    alm = rng.standard_normal(alm_size, dtype=rlm_dtype) + 1j * rng.standard_normal(alm_size, dtype=rlm_dtype)
    almxfl(alm, np.sqrt(cl[:lmax+1] * 0.5), mmax, True)
    real_idcs = Alm.getidx(lmax, np.arange(lmax + 1, dtype=int), 0)
    alm[real_idcs] = alm[real_idcs].real * np.sqrt(2.)
    return alm

class timer:
    def __init__(self, verbose, prefix='', suffix=''):
        self.t0 = time.time()
        self.ti = self.t0
        self.verbose = verbose
        self.prefix = prefix
        self.suffix = suffix
        self.keys = {}
        self.t0s = {}

    def __iadd__(self, othertimer):
        for k in othertimer.keys:
            if not k in self.keys:
                self.keys[k] = othertimer.keys[k]
            else:
                self.keys[k] += othertimer.keys[k]
        return self

    def reset(self):
        t0_, ti_ = self.t0, self.ti
        self.t0 = time.time()
        return t0_, ti_
        
    def set(self, t0, ti):
        self.t0, self.ti = t0, ti

    def reset_ti(self):
        self.ti = time.time()
        self.t0 = time.time()

    def start(self, key):
        """Starts the time tracker
        Args:
            key (_type_): _description_
        """        
        assert key not in self.t0s.keys(), f"{key} already in timer dict t0s"
        self.t0s[key] = time.time()
        
    def delete(self, key):
        if key in self.t0s.keys():
            self.t0s.pop(key) 

    def close(self, key):
        """Closes the time tracker and adds the elapsed time to the key
        Args:
            key (_type_): _description_
        """        
        assert key in self.t0s.keys()
        if key not in self.keys.keys():
            self.keys[key]  = time.time() - self.t0s[key]
        else:
            self.keys[key] += time.time() - self.t0s[key]
        del self.t0s[key]

    def __str__(self):
        if len(self.keys) == 0:
            return r""
        maxlen = np.max([len(k) for k in self.keys])
        dt_tot = time.time() - self.ti
        s = "\n"
        s += "  " + 27*('--') + '\n'
        ts = "\r | {0:%s}" % (str(maxlen) + "s")
        for k in self.keys:
            _ = str(timedelta(seconds=self.keys[k], ))
            _ = ':'.join(str(_).split(':')[2:])
            s += ts.format(k) + ":  [" + _ + "] " + "(%.1f%%)  \t|\n"%(100 * self.keys[k]/dt_tot)
        _ = str(timedelta(seconds=dt_tot))
        _ = ':'.join(str(_).split(':')[2:])
        s += "  " + 27*('- ') + '\n'
        s += ts.format("Total") + ":  [" + _ + "] " + "sec.mus  \t|"
        s += "\n  " + 27*('--') + '\n'
        return s

    def add(self, label):
        """Add the elapsed time since the last call to the label

        Args:
            label (_type_): _description_
        """        
        if label not in self.keys:
            self.keys[label] = 0.
        t0 = time.time()
        self.keys[label] += t0 - self.t0
        self.t0 = t0

    def add_elapsed(self, label):
        """Add the elapsed time since the start of the timer to the label

        Args:
            label (_type_): _description_
        """        
        if label not in self.keys:
            self.keys[label] = 0.
        t0 = time.time()
        self.keys[label] += t0 - self.ti

    def dumpjson(self, fn):
        """Dump the timer keys to a json file

        Args:
            fn (function): _description_
        """        
        json.dump(self.keys, open(fn, 'w'))

    def checkpoint(self, msg):
        """Prints a message and the time elapsed since the last checkpoint

        Args:
            msg (_type_): _description_
        """        
        dt = time.time() - self.t0
        self.t0 = time.time()

        if self.verbose:
            dh = np.floor(dt / 3600.)
            dm = np.floor(np.mod(dt, 3600.) / 60.)
            ds = np.floor(np.mod(dt, 60))
            dhi = np.floor((self.t0 - self.ti) / 3600.)
            dmi = np.floor(np.mod((self.t0 - self.ti), 3600.) / 60.)
            dsi = np.floor(np.mod((self.t0 - self.ti), 60))
            sys.stdout.write("\r  %s   [" % self.prefix + ('%02d:%02d:%02d' % (dh, dm, ds)) + "] "
                             + " (total [" + (
                                 '%02d:%02d:%02d' % (dhi, dmi, dsi)) + "]) " + msg + ' %s \n' % self.suffix)