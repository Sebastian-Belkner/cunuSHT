import os, sys
sys.path.append('/mnt/home/sbelkner/git/pySHT/build')
import numpy as np

import cupy as cp
import popy
import dopy

def Cdoubling(device_pointing_ptr, nphi, nring, host_result):
     
    memaddress = dopy.Cdoubling(device_pointing_ptr, nphi, nring, host_result)
    # mem_pointer = cp.cuda.MemoryPointer(memaddress, size)
    # mem_pointer = cp.cuda.memory.malloc_managed(size * cp.dtype(cp.float64).itemsize)

    # arr = cp.ndarray(2 * npix, dtype=cp.float64, memptr=mem_pointer)
    #TODO I want a cupy.array here
    return memaddress