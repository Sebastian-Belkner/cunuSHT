import os, sys
sys.path.append('/mnt/home/sbelkner/git/pySHT/test/c/build')
import numpy as np

import cupy as cp
import popy

from cupy.cuda.memory import MemoryPointer, UnownedMemory

def memaddress2cparr(memaddress:int, size:int, dtype:type, owner=None):
    sizeByte = size * cp.dtype(dtype).itemsize
    mem = UnownedMemory(memaddress, sizeByte, owner)
    memptr = MemoryPointer(mem, 0)
    return cp.ndarray(size, dtype=dtype, memptr=memptr)


def test_Cpointing(thetasCPU, nring, npix, red, imd, redGPU, host_result):
    print("Testing Cpointing_devres3")
    data = popy.Cpointing_devres3(npix, host_result)
    print("I am Python: host_result = {}".format(host_result))
    print(20*"~")
    
    
    print("\nTesting Cpointing_devres1")
    data = popy.Cpointing_devres1(npix, host_result)
    print("I am Python: host_result = {}".format(host_result))
    print(20*"~")
    
    
    print("\nTesting Cpointing_arrdevres3")
    data = popy.Cpointing_arrdevres3(thetasCPU, nring, npix, host_result)
    print("I am Python: host_result = {}".format(host_result))
    print(20*"~")


    print("\nTesting Cpointing_garrdevres3")
    rGPUptr = redGPU.data.ptr
    print('Pointer in GPU memory: ', rGPUptr) 
    data = popy.Cpointing_garrdevres3(rGPUptr, nring, npix, host_result)
    print("I am Python: host_result = {}".format(host_result))
    print(20*"~")


    print("\nTesting Cpointing_structdevres3")
    data = popy.Cpointing_structdevres3(red, imd, nring, npix, host_result)
    print("I am Python: host_result = {}".format(host_result))
    print(20*"~")
    
    
    print("\nTesting Cpointing_structdevres1")
    data = popy.Cpointing_structdevres1(red, imd, nring, npix, host_result)
    print("I am Python: host_result = {}".format(host_result))
    print(20*"~")
    
    
    print("\nTesting Cpointing_structdevres3_retptr")
    memaddress = popy.Cpointing_structdevres3_retptr(red, imd, nring, npix)
    cparr = memaddress2cparr(memaddress, npix, cp.double)
    print("Shape:", cparr.shape)
    print("Data type:", cparr.dtype)
    print("Strides:", cparr.strides)
    print("Device pointer:", cparr.data.ptr) 
    data = popy.Cpointing_garrdevres3(cparr.data.ptr, nring, npix, host_result)
    print(data)
    print(20*"~")
    
    return cparr