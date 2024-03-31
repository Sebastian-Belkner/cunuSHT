#include <stdio.h>
#include <stdlib.h>
// #include <cuda_runtime.h>


__device__ int fibonaccidown(int n) {
    if (n <= 1) {
        return n;
    }
    return fibonaccidown(n - 1) + fibonaccidown(n - 2);
}


__global__ void fibonaccidown_kernel(int n, int *result) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        result[tid] = fibonaccidown(tid);
    }
}

__device__ int fibonacciup(int nmax, int counter, int fib0, int fib1) {
    if (counter==nmax) {
        if (nmax==0) {
            return 1;
        } else if (nmax==1) {
            return 1;
        }
        return fib1;
     } else if (counter < nmax) {
            return fibonacciup(nmax, counter+1, fib1, fib0+fib1);  
     }
}


__global__ void fibonacciup_kernel(int n, int *result) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int counter = 1;
    if (tid < n) {
        result[tid] = fibonacciup(tid, counter, 1, 1);
    }
}


// # Wrapper function to call the CUDA kernel
extern "C" void Fibonacci(int n, int *host_result) {
    const int threadsPerBlock = 1;
    const int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    int *device_result;

    // Allocate memory on host and device
    cudaMalloc((void **)&device_result, n * sizeof(int));

    // Launch kernel
    printf(" %d %d %d\n", n, threadsPerBlock, blocksPerGrid);
    // fibonaccidown_kernel<<<blocksPerGrid, threadsPerBlock>>>(n, device_result);
    fibonacciup_kernel<<<1, n>>>(n, device_result);
    cudaDeviceSynchronize();

    // Copy result from device to host
    cudaMemcpy(host_result, device_result, n * sizeof(int), cudaMemcpyDeviceToHost);

    cudaError_t errSync  = cudaGetLastError();
    cudaError_t errAsync = cudaDeviceSynchronize();
    if (errSync != cudaSuccess) 
    printf("Sync kernel error: %s\n", cudaGetErrorString(errSync));
    if (errAsync != cudaSuccess)
    printf("Async kernel error: %s\n", cudaGetErrorString(errAsync));

    cudaError_t cuda_error = cudaGetLastError();
    if (cuda_error != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(cuda_error));
    }


    printf("Fibonacci Sequence:\n");
    for (int i = 0; i < n; ++i) {
        printf("%d ", host_result[i]);
    }
    printf("\n");

    cudaFree(device_result);
}


__device__ float d_legendre(int l, float x) {
    if (l == 0) {
        return 1.0f;
    } else if (l == 1) {
        return x;
    } else {
        return ((2.0f * l - 1.0f) * x * d_legendre(l - 1, x) - (l - 1) * d_legendre(l - 2, x)) / l;
    }
}
__global__ void kernel_legendre(int n, float *x, int lmax, float *result) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    if (idx < n) {
        float x_ = x[threadIdx.x];
        result[idx] = d_legendre(blockIdx.x, x_);
    }
}


// # Wrapper function to call the CUDA kernel
extern "C" void Legendre(int lmax, float *host_x, float *host_result, int size_x) {
    float *device_x, *device_result;

    // Allocate device memory
    cudaMalloc((void **)&device_x, size_x * sizeof(float));
    cudaMalloc((void **)&device_result, size_x * (lmax + 1) * sizeof(float));

    // Copy data from host to device
    cudaMemcpy(device_x, host_x, size_x * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    // int threadsPerBlock = 256;
    // int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
    printf(" %d %d\n", lmax, size_x);
    // kernel_legendre<<<blocksPerGrid, threadsPerBlock>>>(size* (lmax + 1), device_x, lmax, device_result);
    kernel_legendre<<<lmax+1, size_x>>>(size_x * (lmax + 1), device_x, lmax, device_result);
    cudaDeviceSynchronize();

    // Copy result from device to host
    cudaMemcpy(host_result, device_result, size_x * (lmax + 1) * sizeof(float), cudaMemcpyDeviceToHost);


    // Print results
    // for (int l = 0; l <= lmax; ++l) {
    //     printf("P_%d(x) = ", l);
    //     for (int i = 0; i < size_x; ++i) {
    //         printf("%.2f ", host_result[l*size_x+i]);
    //     }
    //     printf("\n");
    // }
    // printf("done");

    // Free device memory
    cudaFree(device_x);
    cudaFree(device_result);
}



__global__ void multiplyByTwo(float *array, int size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    // Ensure we stay within the array bounds
    if (tid < size) {
        array[tid] *= 2;
    }
}

// # Wrapper function to call the CUDA kernel
extern "C" void multiply(float *hostArray, float *host_output, int arraySize) {
    const int threadsPerBlock = 1024;
    float *deviceArray;

    // Allocate device memory
    cudaMalloc((void **)&deviceArray, arraySize * sizeof(float));
    // Copy data from host to device
    cudaMemcpy((void **)deviceArray, hostArray, arraySize * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    int blocksPerGrid = (arraySize + threadsPerBlock - 1) / threadsPerBlock;
    printf("%d %d %d\n", arraySize, threadsPerBlock, blocksPerGrid);
    multiplyByTwo<<<blocksPerGrid, threadsPerBlock>>>(deviceArray, arraySize);
    // mult<<<1, 1>>>(hostArray, arraySize);
    cudaDeviceSynchronize(); // Wait for kernel to finish

    // Copy result from device to host
    cudaMemcpy(host_output, deviceArray, arraySize * sizeof(float), cudaMemcpyDeviceToHost);

    cudaError_t errSync  = cudaGetLastError();
    cudaError_t errAsync = cudaDeviceSynchronize();
    if (errSync != cudaSuccess) 
    printf("Sync kernel error: %s\n", cudaGetErrorString(errSync));
    if (errAsync != cudaSuccess)
    printf("Async kernel error: %s\n", cudaGetErrorString(errAsync));

    // Print the updated array
    printf("input \n");
    for (int i = 0; i < arraySize; ++i) {
        printf("%.2f ", hostArray[i]);
    }
    printf("\noutput \n");
    for (int i = 0; i < arraySize; ++i) {
        printf("%.2f ", host_output[i]);
    }
    printf("\n");

    // Free device memory
    cudaFree(deviceArray);
}