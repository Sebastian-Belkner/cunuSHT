import os, sys
sys.path.append('/mnt/home/sbelkner/git/pySHT/build')
import numpy as np

import cupy as cp
import pysht_interface.popy as popy
import pysht_interface.dopy as dopy
from cupy.cuda.memory import MemoryPointer, UnownedMemory

def memaddress2cparr(memaddress:int, size:int, dtype:type, owner=None):
    sizeByte = size * cp.dtype(dtype).itemsize
    mem = UnownedMemory(memaddress, sizeByte, owner)
    memptr = MemoryPointer(mem, 0)
    return cp.ndarray(size, dtype=dtype, memptr=memptr)

def Cpointing_ptrs(thetas, phi0, nphis, ringstarts, synthmaps, nring, npix, host_result):
    mas = popy.Cpointing_ptrs(thetas.data.ptr, phi0.data.ptr, nphis.data.ptr, ringstarts.data.ptr, synthmaps.data.ptr, nring, npix, host_result)
    cparr_theta = memaddress2cparr(mas[0], npix, cp.double)
    cparr_phi = memaddress2cparr(mas[1], npix, cp.double)
    return [cparr_theta, cparr_phi]

def Cpointing_cparr(thetas, phi0, nphis, ringstarts, synthmap, nring, npix, outarr_pt, outarr_pp):
    popy.CUpointing_cparr(thetas, phi0, nphis, ringstarts, synthmap, outarr_pt, outarr_pp)

def Cpointing_2Dto1D(thetas, phi0, nphis, ringstarts, spin1_theta, spin1_phi, out_ptheta, out_pphi):
    popy.CUpointing_2Dto1D(thetas, phi0, nphis, ringstarts, spin1_theta, spin1_phi, out_ptheta, out_pphi)
    
def Cpointing_1Dto1D(thetas, phi0, nphis, ringstarts, spin1_theta, spin1_phi, out_ptheta, out_pphi):
    popy.CUpointing_1Dto1D(thetas, phi0, nphis, ringstarts, spin1_theta, spin1_phi, out_ptheta, out_pphi)

def Cdoubling_ptrs(pointings, nring, nphi):
    memaddress = dopy.Cdoubling_ptrs(pointings.data.ptr, nring, nphi)
    cparr = memaddress2cparr(memaddress, 4*nphi*nring, cp.double)
    return cparr

def Cdoubling_1D(synth1D, nring, nphi, out):
    dopy.CUdoubling_cparr1D(synth1D, nring, nphi, out)
    
def Cdoubling_cparr2D(synth2D, nring, nphi, out):
    dopy.CUdoubling_cparr2D(synth2D, nring, nphi, out)