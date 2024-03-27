# Python function calling the compiled C++/CUDA function

import ctypes
import os
import numpy as np

# Load the CUDA library
cuda_lib = ctypes.CDLL('/mnt/home/sbelkner/git/pySHT/pysht/c/assocLeg.so')  # Update with the correct path

    
def assLegendre():
    # Define the function prototype
    cuda_lib.assLegendre.argtypes = [
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int]
    cuda_lib.assLegendre.restype = None

    # Prepare data
    lmax_ = 1
    mmax_ = 1
    input_array = np.arange(0, 1, 0.1)
    output_array = np.zeros_like(input_array)
    size_ = len(input_array)

    # Convert Python lists to ctypes arrays
    lmax = ctypes.c_int(lmax_)
    mmax = ctypes.c_int(mmax_)
    size = ctypes.c_int(size_)
    input_x = (ctypes.c_float * size_)(*input_array)
    output_x = (ctypes.c_float * size_)(*output_array)

    # Call the CUDA function
    cuda_lib.assLegendre(lmax, mmax, input_x, output_x, size)

    # Print the result
    result = np.array(output_x)
    print(result.shape)
    import matplotlib.pyplot as plt
    plt.plot(result[:10])
    plt.savefig('/mnt/home/sbelkner/git/pySHT/pysht/c/result.png')
    print("Result:", result)
    

    
if __name__ == "__main__":
    assLegendre()