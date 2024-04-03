import os, sys
sys.path.append('/mnt/home/sbelkner/git/pySHT/build')
import numpy as np

import cupy as cp
import popy
import dopy
from cupy.cuda.memory import MemoryPointer, UnownedMemory

def memaddress2cparr(memaddress:int, size:int, dtype:type, owner=None):
    sizeByte = size * cp.dtype(dtype).itemsize
    mem = UnownedMemory(memaddress, sizeByte, owner)
    memptr = MemoryPointer(mem, 0)
    return cp.ndarray(size, dtype=dtype, memptr=memptr)

def Cpointing_ptrs(thetas_, phi0_, nphis_, ringstarts_, synthmaps_, nrings, npix, host_result):
    mas = popy.Cpointing_ptrs(thetas_.data.ptr, phi0_.data.ptr, nphis_.data.ptr, ringstarts_.data.ptr, synthmaps_.data.ptr, nrings, npix, host_result)
    print("I am Python: memory address from pointing: {}".format(mas))
    cparr_theta = memaddress2cparr(mas[0], npix, cp.double)
    cparr_phi = memaddress2cparr(mas[1], npix, cp.double)
    # print("Shape_theta:", cparr_theta.shape)
    # print("Data type_theta:", cparr_theta.dtype)
    # print("Strides_theta:", cparr_theta.strides)
    # print("Device pointer_theta:", cparr_theta.data.ptr) 
    return [cparr_theta, cparr_phi]

def Cdoubling_ptrs(pointings_memaddress, nring, nphi):
    memaddress = dopy.Cdoubling_ptrs(pointings_memaddress.data.ptr, nring, nphi)
    cparr = memaddress2cparr(memaddress, 4*nphi*nring, cp.double)
    print("Shape:", cparr.shape)
    print("Data type:", cparr.dtype)
    print("Strides:", cparr.strides)
    print("Device pointer:", cparr.data.ptr) 
    return cparr