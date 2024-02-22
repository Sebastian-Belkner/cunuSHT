# Python function calling the compiled C++/CUDA function

import ctypes
import os
import numpy as np

import time

import matplotlib.pyplot as plt


def synthesis():
    cuda_lib = ctypes.CDLL('/mnt/home/sbelkner/git/pySHT/pysht/c/assocLeg.so')
    # Define the function prototype
    cuda_lib.synthesis.argtypes = [
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_float),
    ]
    cuda_lib.synthesis.restype = None

    # Prepare data
    lmax_ = 8
    mmax_ = 8
    Nlat, Nlon = 500, 500
    size_ = Nlat * Nlon
    output_array = np.zeros(shape=size_, dtype=float)

    # Convert Python lists to ctypes arrays
    lmax = ctypes.c_int(lmax_)
    mmax = ctypes.c_int(mmax_)
    size = ctypes.c_int(output_array.shape[0])
    output_x = (ctypes.c_float * size_)(*output_array)
    result = np.zeros_like(output_x)


    f, axarr = plt.subplots(lmax_,2*lmax_-1,figsize=(12,12))
    for l_ in range(0,lmax_):
        for m_ in range(-l_,l_+1):
            l = ctypes.c_int(l_)
            m = ctypes.c_int(m_)
        # for mmax in range(1,lmax+1,2):
            print(l_, m_)
            output_array = np.zeros(shape=size_, dtype=float)
            output_x = (ctypes.c_float * size_)(*output_array)
            cuda_lib.synthesis(l, m, Nlat, Nlon, output_x, size)
            result = np.array(output_x)
            axarr[l_][int(lmax_)-1+m_].imshow(result.reshape(Nlat,Nlon), cmap='YlGnBu')
    plt.savefig('/mnt/home/sbelkner/git/pySHT/pysht/c/resultall.png')


def legendre():
    cuda_lib = ctypes.CDLL('/mnt/home/sbelkner/git/pySHT/pysht/c/assocLeg.so')
    # Define the function prototype
    cuda_lib.Legendreup.argtypes = [
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int]
    cuda_lib.Legendreup.restype = None

    cuda_lib.Legendredown.argtypes = [
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
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
    input_x = (ctypes.c_float * size_)(*input_array)
    output_x = (ctypes.c_float * s_)(*output_array)

    cuda_lib.Legendreup(lmax, input_x, output_x, size)
    # start = time.process_time()
    
    cuda_lib.Legendredown(lmax, input_x, output_x, size)
 
def associatedlegendre():
    cuda_lib = ctypes.CDLL('/mnt/home/sbelkner/git/pySHT/pysht/c/assocLeg.so')
    # Define the function prototype
    cuda_lib.associated_legendre.argtypes = [
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int]
    cuda_lib.associated_legendre.restype = None

    # Prepare data
    lmax_ = 3
    mmax_ = 3
    dx = 0.002
    input_array = np.arange(-1, 1+dx, dx)
    output_array = np.zeros_like(input_array)
    size_ = len(input_array)

    # Convert Python lists to ctypes arrays
    lmax = ctypes.c_int(lmax_)
    mmax = ctypes.c_int(mmax_)
    size = ctypes.c_int(size_)
    input_x = (ctypes.c_float * size_)(*input_array)
    output_x = (ctypes.c_float * size_)(*output_array)

    for lmax in range(0, lmax_+1):
        # for mmax in range(0, lmax+1):
        for mmax in range(-lmax, lmax+1):
            cuda_lib.associated_legendre(lmax, mmax, input_x, output_x, size)
            result = np.array(output_x)
            if mmax==0:
                plt.plot(input_array, result, label='P_%d^%d(x)' % (lmax, mmax))
                # print(result)
    plt.legend()
    plt.title('P_l^m(x)')
    plt.xlabel('x')
    plt.ylabel('associated Legendre')
    plt.ylim(-2.2,2.2)
    plt.savefig('/mnt/home/sbelkner/git/pySHT/pysht/c/result.png')
      
def multiply2():
    # Define the function prototype
    # cuda_lib.multiply.argtypes = [
    #     ctypes.POINTER(ctypes.c_float),
    #     ctypes.POINTER(ctypes.c_float),
    #     ctypes.c_int]
    # # cuda_lib.multiply.restype = None

    # Prepare data
    input_array = np.arange(1, 2, 0.01)
    output_array = np.zeros_like(input_array)
    size_ = len(input_array)

    # Convert Python lists to ctypes arrays
    size = ctypes.c_int(size_)
    input_x = (ctypes.c_float * size_)(*input_array)
    output_x = (ctypes.c_float * size_)(*output_array)

    # Call the CUDA function
    cuda_lib.multiply()

    # Print the result
    result = np.array(output_x)
    import matplotlib.pyplot as plt
    # plt.plot(result[:10])
    # plt.savefig('/mnt/home/sbelkner/git/pySHT/pysht/c/result.png')
    print("Result:", result)
    
def fibonacci():
    cuda_lib = ctypes.CDLL('/mnt/home/sbelkner/git/pySHT/pysht/c/kernel.so')
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
    # associatedlegendre()
    synthesis()
    # multiply()
    # fibonacci()