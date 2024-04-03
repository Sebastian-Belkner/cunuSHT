#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <cmath>
#include <chrono>
#include <time.h>

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_device_runtime_api.h>

#include "doubling.cuh"


__global__ void compute_doubling_spin0(double* pointings, int nring, int nphi, double *doublings) {
    // map_dfs = np.empty((2 * ntheta - 2, nphi), dtype=np.complex128 if spin == 0 else ctype[map.dtype])
    //idx is nrings
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx <= nring) {
        // map_dfs[:ntheta, :] = map[0]
        // map_dfs[ntheta:, :nphihalf] = map_dfs[ntheta - 2:0:-1, nphihalf:]
        // map_dfs[ntheta:, nphihalf:] = map_dfs[ntheta - 2:0:-1, :nphihalf]
        for (int i = 0; i < nphi; i++) {
            doublings[i] = pointings[i];
            doublings[i + nphi] = pointings[i + nphi];
        }
        for (int i = nphi; i < 2*nphi; i++) {
            doublings[i] = pointings[2*nphi - i];
            doublings[i + nphi] = pointings[nphi - i];
        }
    }
}

double* CUdoubling(double* devpointings, int nring, int nphi) {
    printf("starting allocation\n");
    clock_t start, stop;
    double cpu_time_used;
    start = clock();
    double *device_result;
    cudaMalloc((void**)&device_result, 4*nring*nphi * sizeof(double));

    printf("Calling kernel\n ");
    start = clock();
    const int threadsPerBlock = 128;
    int blocksPerGrid = (nring + threadsPerBlock - 1) / threadsPerBlock;
    compute_doubling_spin0<<<blocksPerGrid, threadsPerBlock>>>(devpointings, nring, nphi, device_result);
    cudaDeviceSynchronize();
    cudaError_t errSync  = cudaGetLastError();
    cudaError_t errAsync = cudaDeviceSynchronize();
    if (errSync != cudaSuccess) 
    printf("Sync kernel error: %s\n", cudaGetErrorString(errSync));
    if (errAsync != cudaSuccess)
    printf("Async kernel error: %s\n", cudaGetErrorString(errAsync));
    stop = clock();
    printf("Kernel done: duration = %f\n sec", (double)(stop - start)/CLOCKS_PER_SEC);

    cudaFree(devpointings);

    return device_result;
}