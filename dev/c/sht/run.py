# Python function calling the compiled C++/CUDA function

import ctypes
import os
import numpy as np

import time

import healpy as hp
import matplotlib.pyplot as plt
import cunusht

# Define a complex structure
class c_complex(ctypes.Structure):
    _fields_ = [("real", ctypes.c_double),
                ("imag", ctypes.c_double)]

def synthesis_ring():
    cuda_lib = ctypes.CDLL(cunusht.__file__+'/c/assocLeg.so')
    # Define the function prototype
    cuda_lib.synthesis_ring.argtypes = [
        # ctypes.POINTER(c_complex),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_double),
    ]
    cuda_lib.synthesis_ring.restype = None

    # Prepare data
    lmax_ = 100
    Nlat, Nlon = 1000, 1000
    size_ = Nlat*Nlon
    output_array = np.zeros(shape=size_, dtype=np.double)
    ll = np.arange(lmax_+1)
    alm_ = hp.synalm(1.*np.exp(-ll/100.))
    
    alm_ = [c_complex(np.real(a), np.imag(a)) for a in alm_]
    print(len(alm_))
    # Convert Python lists to ctypes arrays
    lmax = ctypes.c_int(lmax_)
    size = ctypes.c_int(output_array.shape[0])
    output_x = (ctypes.c_double * size_)(*output_array)
    # alm = (c_complex * len(alm_))(*alm_)
    result = np.zeros_like(output_x)
    
    
    f, axarr = plt.subplots(1,1,figsize=(12,12))
    output_array = np.zeros(shape=size_, dtype=np.double)
    output_x = (ctypes.c_double * size_)(*output_array)
    cuda_lib.synthesis_ring(len(alm_), Nlat, Nlon, output_x)
    result = np.array(output_x)
    axarr.imshow(result.reshape(Nlat,Nlon), cmap='YlGnBu')
    print(result)
    plt.savefig(cunusht.__file__+'/c/result.png')
    
    # loffset = lmax_-5
    # f, axarr = plt.subplots(lmax_-loffset,2*lmax_-1-2*loffset,figsize=(18,18))
    # for l_ in range(loffset,lmax_):
    #     for m_ in range(-l_+loffset,l_+1-loffset):
    #         l = ctypes.c_int(l_)
    #         m = ctypes.c_int(m_)

    #         output_array = np.zeros(shape=size_, dtype=np.double)
    #         output_x = (ctypes.c_double * size_)(*output_array)
    #         print(l,m)
    #         cuda_lib.synthesis_ring(l, m, Nlat, Nlon, output_x, size)
    #         result = np.array(output_x)
    #         print(result)
    #         axarr[l_-loffset][int(lmax_)-1+m_-loffset].imshow(result.reshape(Nlat,Nlon), cmap='YlGnBu')
    # plt.savefig(cunusht.__file__+'/c/resultall.png')

def synthesis_ringl():
    cuda_lib = ctypes.CDLL(cunusht.__file__+'/c/assocLeg.so')
    # Define the function prototype
    cuda_lib.synthesis_ringl.argtypes = [
        # ctypes.POINTER(c_complex),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int,
        ctypes.c_int
    ]
    cuda_lib.synthesis_ringl.restype = None

    # Prepare data
    lmax_ = 40
    mmax_ = 0
    Nlat, Nlon = 400, 400
    size_ = Nlat*Nlon
    output_array = np.zeros(shape=size_, dtype=np.double)
    ll = np.arange(lmax_+1)
    alm_ = hp.synalm(1.*np.exp(-ll/100.))
    
    
    alm_ = [c_complex(np.real(a), np.imag(a)) for a in alm_]
    # Convert Python lists to ctypes arrays
    lmax = ctypes.c_int(lmax_)
    size = ctypes.c_int(output_array.shape[0])
    output_x = (ctypes.c_double * size_)(*output_array)
    # alm = (c_complex * len(alm_))(*alm_)
    result = np.zeros_like(output_x)
    
    
    # f, axarr = plt.subplots(1,1,figsize=(12,12))
    # output_array = np.zeros(shape=size_, dtype=np.double)
    # output_x = (ctypes.c_double * size_)(*output_array)
    # l = ctypes.c_int(lmax_)
    # m = ctypes.c_int(mmax_)
    # cuda_lib.synthesis_ringl(len(alm_), Nlat, Nlon, output_x, l, m)
    # result = np.array(output_x)
    # axarr.imshow(result.reshape(Nlat,Nlon), cmap='YlGnBu')
    # # print(result)
    # plt.savefig(cunusht.__file__+'/c/result.png')
    
    loffset = 30 #lmax_-5
    # f, axarr = plt.subplots(lmax_-loffset,2*lmax_-1-2*loffset,figsize=(18,18))
    f, axarr = plt.subplots(lmax_-loffset,lmax_-loffset,figsize=(18,18))
    for l_ in range(loffset,lmax_):
        for m_ in range(0,l_+1-loffset): #-l_+loffset
            l = ctypes.c_int(l_)
            m = ctypes.c_int(m_)

            output_array = np.zeros(shape=size_, dtype=np.double)
            output_x = (ctypes.c_double * size_)(*output_array)
            print(l_,m_)
            cuda_lib.synthesis_ringl(len(alm_), Nlat, Nlon, output_x, l, m)
            result = np.array(output_x)
            # axarr[l_-loffset][m_-loffset].imshow(result.reshape(Nlat,Nlon), cmap='YlGnBu')
            axarr[l_-loffset][m_].imshow(result.reshape(Nlat,Nlon), cmap='YlGnBu')
            
            ctypes.CDLL(None).dlclose(cuda_lib._handle)
            cuda_lib = ctypes.CDLL(cunusht.__file__+'/c/assocLeg.so')
            # Define the function prototype
            cuda_lib.synthesis_ringl.argtypes = [
                # ctypes.POINTER(c_complex),
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.POINTER(ctypes.c_double),
                ctypes.c_int,
                ctypes.c_int
            ]
            cuda_lib.synthesis_ringl.restype = None
    plt.savefig(cunusht.__file__+'/c/resultall.png')


def synthesis_NlonNlat():
    cuda_lib = ctypes.CDLL(cunusht.__file__+'/c/assocLeg.so')
    # Define the function prototype
    cuda_lib.synthesis_NlonNlat.argtypes = [
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_double),
    ]
    cuda_lib.synthesis_NlonNlat.restype = None

    # Prepare data
    lmax_ = 160
    mmax_ = 4
    Nlat, Nlon = 500, 500
    size_ = Nlat * Nlon
    output_array = np.zeros(shape=size_, dtype=np.double)

    # Convert Python lists to ctypes arrays
    lmax = ctypes.c_int(lmax_)
    mmax = ctypes.c_int(mmax_)
    size = ctypes.c_int(output_array.shape[0])
    output_x = (ctypes.c_double * size_)(*output_array)
    result = np.zeros_like(output_x)

    # f, axarr = plt.subplots(1,1,figsize=(12,12))
    # for l_ in range(lmax_,lmax_+1):
    #     for m_ in range(mmax_,mmax_+1):
    #         l = ctypes.c_int(l_)
    #         m = ctypes.c_int(m_)
    #         print(l_, m_)
    #         output_array = np.zeros(shape=size_, dtype=np.double)
    #         output_x = (ctypes.c_double * size_)(*output_array)
    #         cuda_lib.synthesis_NlonNlat(l, m, Nlat, Nlon, output_x, size)
    #         result = np.array(output_x)
    #         axarr.imshow(result.reshape(Nlat,Nlon), cmap='YlGnBu')
    #         print(result)
    # plt.savefig(cunusht.__file__+'/c/result.png')
    
    loffset = lmax_-10
    f, axarr = plt.subplots(lmax_-loffset,2*lmax_-1-2*loffset,figsize=(18,18))
    for l_ in range(loffset,lmax_):
        for m_ in range(-l_+loffset,l_+1-loffset):
            l = ctypes.c_int(l_)
            m = ctypes.c_int(m_)

            output_array = np.zeros(shape=size_, dtype=np.double)
            output_x = (ctypes.c_double * size_)(*output_array)
            cuda_lib.synthesis_NlonNlat(l, m, Nlat, Nlon, output_x, size)
            result = np.array(output_x)
            print(result)
            axarr[l_-loffset][int(lmax_)-1+m_-loffset].imshow(result.reshape(Nlat,Nlon), cmap='YlGnBu')
    plt.savefig(cunusht.__file__+'/c/resultall.png')

def legendre():
    cuda_lib = ctypes.CDLL(cunusht.__file__+'/c/assocLeg.so')
    # Define the function prototype
    cuda_lib.Legendreup.argtypes = [
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int]
    cuda_lib.Legendreup.restype = None

    cuda_lib.Legendredown.argtypes = [
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int]
    cuda_lib.Legendredown.restype = None
    
    # Prepare data
    lmax_ = 4096
    input_array = np.arange(-1, 1, 0.002)
    output_array = np.zeros_like(input_array)
    size_ = len(input_array)
    s_ = (lmax_+1)*size_

    # Convert Python lists to ctypes arrays
    lmax = ctypes.c_int(lmax_)
    size = ctypes.c_int(size_)
    input_x = (ctypes.c_double * size_)(*input_array)
    output_x = (ctypes.c_double * s_)(*output_array)

    cuda_lib.Legendreup(lmax, input_x, output_x, size)
    # start = time.process_time()
    
    cuda_lib.Legendredown(lmax, input_x, output_x, size)
 
def associatedlegendre():
    cuda_lib = ctypes.CDLL(cunusht.__file__+'/c/assocLeg.so')
    # Define the function prototype
    cuda_lib.associated_legendre.argtypes = [
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int]
    cuda_lib.associated_legendre.restype = None

    # Prepare data
    lmax_ = 10
    mmax_ = 5
    dx = 0.002
    input_array = np.arange(-1, 1+dx, dx)
    output_array = np.zeros_like(input_array)
    size_ = len(input_array)

    # Convert Python lists to ctypes arrays
    lmax = ctypes.c_int(lmax_)
    mmax = ctypes.c_int(mmax_)
    size = ctypes.c_int(size_)
    input_x = (ctypes.c_double * size_)(*input_array)
    output_x = (ctypes.c_double * size_)(*output_array)

    for lmax in range(mmax_, lmax_+1):
        # for mmax in range(0, lmax+1):
        for mmax in range(mmax_, mmax_+1):
            cuda_lib.associated_legendre(ctypes.c_int(lmax), ctypes.c_int(mmax), input_x, output_x, size)
            result = np.array(output_x)
            # if mmax==mmax_:
            plt.plot(input_array, result, label='P_%d^%d(x)' % (lmax, mmax))
                # print(result)
    plt.legend()
    plt.title('P_l^m(x)')
    plt.xlabel('x')
    plt.ylabel('associated Legendre')
    # plt.ylim(-2.2,2.2)
    plt.savefig(cunusht.__file__+'/c/result.png')
      
def multiply2():
    # Define the function prototype
    # cuda_lib.multiply.argtypes = [
    #     ctypes.POINTER(ctypes.c_double),
    #     ctypes.POINTER(ctypes.c_double),
    #     ctypes.c_int]
    # # cuda_lib.multiply.restype = None

    # Prepare data
    input_array = np.arange(1, 2, 0.01)
    output_array = np.zeros_like(input_array)
    size_ = len(input_array)

    # Convert Python lists to ctypes arrays
    size = ctypes.c_int(size_)
    input_x = (ctypes.c_double * size_)(*input_array)
    output_x = (ctypes.c_double * size_)(*output_array)

    # Call the CUDA function
    cuda_lib.multiply()

    # Print the result
    result = np.array(output_x)
    import matplotlib.pyplot as plt
    # plt.plot(result[:10])
    # plt.savefig(cunusht.__file__+'/c/result.png')
    print("Result:", result)
    
def fibonacci():
    cuda_lib = ctypes.CDLL(cunusht.__file__+'/c/kernel.so')
    cuda_lib.multiply.argtypes = [
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_int)
        ]
    cuda_lib.multiply.restype = None

    size_ = 40
    output_array = np.zeros(shape=size_, dtype=int)
    size = ctypes.c_int(size_)
    output_x = (ctypes.c_int * size_)(*output_array)

    # Call the CUDA function
    cuda_lib.Fibonacci(size, output_x)

    result = np.array(output_x)
    print("Result:", result)
    
if __name__ == "__main__":
    # legendre()
    associatedlegendre()
    # synthesis_NlonNlat()
    # synthesis_ring()
    # synthesis_ringl()
    # multiply()
    # fibonacci()
    
    
# for P_l^m(x):
#     Initialize P_m^m(x): Pmm and Pmp1m
#     for ll = [m:l]:
#         P_(ll+1) = a Pmp1m + b Pmm