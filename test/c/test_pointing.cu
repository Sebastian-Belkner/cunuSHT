#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <cmath>
#include <chrono>
#include <time.h>
#include <tuple>

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_device_runtime_api.h>

#include "kernel_params.h"
#include "test_pointing.cuh"


__global__ void compute_devres(int size, double *pointings) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    double PI = 3.14159265359;

    if (idx <= size) {
        pointings[idx] = (double)idx;
    }
}

__global__ void compute_arrdevres(double *thetas, int nring, int size, double *pointings) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    double PI = 3.14159265359;

    if (idx <= size) {
        pointings[idx] = thetas[idx/(size/nring)];//(double)idx;
    }
}

__global__ void compute_garrdevres(double *red, int nring, int size, double *pointings) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    double PI = 3.14159265359;

    if (idx <= size) {
        pointings[idx] = red[idx]+10.;//(double)idx;
    }
}

__global__ void compute_structdevres(Params params, double *pointings) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    double PI = 3.14159265359;

    if (idx <= params.npix) {
        pointings[idx] = params.red[idx]+100.;//thetas[idx/(size/nring)];//(double)idx;
    }
}

double* CUpointing_devres11(int npix) {
    printf("CUpointing_devres3:: starting kernel\n");
    clock_t start, stop;
    double cpu_time_used;
    start = clock();

    double* devres;
    cudaMalloc((void**)&devres, npix * sizeof(double));
    return devres;
}

void CUpointing_devres12(int npix, double* devres) {
    printf("CUpointing_devres3:: starting kernel\n");
    clock_t start, stop;
    double cpu_time_used;
    start = clock();

    const int threadsPerBlock = 128;
    int blocksPerGrid = (npix + threadsPerBlock - 1) / threadsPerBlock;
    compute_devres<<<blocksPerGrid, threadsPerBlock>>>(npix, devres);
    cudaDeviceSynchronize();

    cudaError_t errSync  = cudaGetLastError();
    cudaError_t errAsync = cudaDeviceSynchronize();
    if (errSync != cudaSuccess) 
    printf("Sync kernel error: %s\n", cudaGetErrorString(errSync));
    if (errAsync != cudaSuccess)
    printf("Async kernel error: %s\n", cudaGetErrorString(errAsync));
    stop = clock();
}

void CUpointing_devres13(int npix, double* devres, double* host_result) {
    printf("CUpointing_devres3:: starting kernel\n");
    clock_t start, stop;
    double cpu_time_used;
    start = clock();
    cudaMemcpy(host_result, devres, npix * sizeof(double), cudaMemcpyDeviceToHost);
    stop = clock();
    printf("Done copying from GPU: duration = %f\n", (double)(stop - start)/CLOCKS_PER_SEC);
    printf("\nHost_result: ");
    for (int i = 0; i < npix; i+=1) {
        printf("%.2f, ", host_result[i]);
    }printf("\n");
}

void CUpointing_devres3(int npix, double* host_result) {
    printf("CUpointing_devres3:: starting kernel\n");
    clock_t start, stop;
    double cpu_time_used;
    start = clock();

    double* devres;
    cudaMalloc((void**)&devres, npix * sizeof(double));
    stop = clock();
    printf("Done with allocation: duration = %f\n", (double)(stop - start)/CLOCKS_PER_SEC);

    const int threadsPerBlock = 128;
    int blocksPerGrid = (npix + threadsPerBlock - 1) / threadsPerBlock;
    compute_devres<<<blocksPerGrid, threadsPerBlock>>>(npix, devres);
    cudaDeviceSynchronize();

    cudaError_t errSync  = cudaGetLastError();
    cudaError_t errAsync = cudaDeviceSynchronize();
    if (errSync != cudaSuccess) 
    printf("Sync kernel error: %s\n", cudaGetErrorString(errSync));
    if (errAsync != cudaSuccess)
    printf("Async kernel error: %s\n", cudaGetErrorString(errAsync));
    stop = clock();
    printf("Done with kernel: duration = %f\n", (double)(stop - start)/CLOCKS_PER_SEC);

    start = clock();
    cudaMemcpy(host_result, devres, npix * sizeof(double), cudaMemcpyDeviceToHost);
    stop = clock();
    printf("Done copying from GPU: duration = %f\n", (double)(stop - start)/CLOCKS_PER_SEC);
    printf("\nHost_result: ");
    for (int i = 0; i < npix; i+=1) {
        printf("%.2f, ", host_result[i]);
    }printf("\n");
}


void CUpointing_arrdevres3(double *thetas, int nring, int npix, double* host_result) {
    printf("CUpointing_arrdevres3:: starting kernel\n");
    clock_t start, stop;
    double cpu_time_used;
    printf("nring: %d\n", nring);
    printf("npix: %d\n", npix);
    printf("\nthetas: ");
    for (int i = 0; i < nring; i+=1) {
        printf("%.2f, ", thetas[i]);
    }printf("\n");

    start = clock();
    double* devres;
    cudaMalloc((void**)&devres, npix * sizeof(double));

    double* devthetas;
    cudaMalloc((void**)&devthetas, nring * sizeof(double));
    cudaMemcpy(devthetas, thetas, nring * sizeof(double), cudaMemcpyHostToDevice);
    stop = clock();
    cudaError_t errSync  = cudaGetLastError();
    cudaError_t errAsync = cudaDeviceSynchronize();
    if (errSync != cudaSuccess) 
    printf("Sync kernel error: %s\n", cudaGetErrorString(errSync));
    if (errAsync != cudaSuccess)
    printf("Async kernel error: %s\n", cudaGetErrorString(errAsync));
    printf("Done with allocation: duration = %f\n", (double)(stop - start)/CLOCKS_PER_SEC);

    const int threadsPerBlock = 128;
    int blocksPerGrid = (npix + threadsPerBlock - 1) / threadsPerBlock;
    compute_arrdevres<<<blocksPerGrid, threadsPerBlock>>>(devthetas, nring, npix, devres);
    cudaDeviceSynchronize();

    errSync  = cudaGetLastError();
    errAsync = cudaDeviceSynchronize();
    if (errSync != cudaSuccess) 
    printf("Sync kernel error after kernel: %s\n", cudaGetErrorString(errSync));
    if (errAsync != cudaSuccess)
    printf("Async kernel error after kernel: %s\n", cudaGetErrorString(errAsync));
    stop = clock();
    printf("Done with kernel: duration = %f\n", (double)(stop - start)/CLOCKS_PER_SEC);

    start = clock();
    cudaMemcpy(host_result, devres, npix * sizeof(double), cudaMemcpyDeviceToHost);
    stop = clock();
    printf("Done copying from GPU: duration = %f\n", (double)(stop - start)/CLOCKS_PER_SEC);
    printf("\nHost_result: ");
    for (int i = 0; i < npix; i+=1) {
        printf("%.2f, ", host_result[i]);
    }
    printf("\n");
}

void CUpointing_garrdevres3(double *devred, int nring, int npix, double* host_result) {
    printf("CUpointing_garrdevres3:: starting kernel\n");
    clock_t start, stop;
    double cpu_time_used;
    printf("nring: %d\n", nring);
    printf("npix: %d\n", npix);

    start = clock();
    double* devres;
    cudaMalloc((void**)&devres, npix * sizeof(double));
    stop = clock();
    printf("Done with allocation: duration = %f\n", (double)(stop - start)/CLOCKS_PER_SEC);

    const int threadsPerBlock = 128;
    int blocksPerGrid = (npix + threadsPerBlock - 1) / threadsPerBlock;
    compute_garrdevres<<<blocksPerGrid, threadsPerBlock>>>(devred, nring, npix, devres);
    cudaDeviceSynchronize();

    cudaError_t errSync  = cudaGetLastError();
    cudaError_t errAsync = cudaDeviceSynchronize();
    if (errSync != cudaSuccess) 
    printf("Sync kernel error: %s\n", cudaGetErrorString(errSync));
    if (errAsync != cudaSuccess)
    printf("Async kernel error: %s\n", cudaGetErrorString(errAsync));
    stop = clock();
    printf("Done with kernel: duration = %f\n", (double)(stop - start)/CLOCKS_PER_SEC);

    start = clock();
    cudaMemcpy(host_result, devres, npix * sizeof(double), cudaMemcpyDeviceToHost);
    stop = clock();
    printf("Done copying from GPU: duration = %f\n", (double)(stop - start)/CLOCKS_PER_SEC);
    printf("\nHost_result: ");
    for (int i = 0; i < npix; i+=1) {
        printf("%.2f, ", host_result[i]);
    }
    printf("\n");
}

void CUpointing_structdevres3(Params params, double* host_result) {
    printf("CUpointing_structdevres3:: starting kernel\n");
    clock_t start, stop;
    double cpu_time_used;
    printf("params.nring: %d\n", params.nring);
    printf("params.npix: %d\n", params.npix);
    printf("params.red:\n");
    for (int i = 0; i < params.npix; i+=1) {
        printf("%.2f, ", params.red[i]);
    }printf("\n");
    printf("params.imd:\n");
    for (int i = 0; i < params.npix; i+=1) {
        printf("%.2f, ", params.imd[i]);
    }printf("\n");

    start = clock();

    double* devres;
    cudaMalloc((void**)&devres, params.npix * sizeof(double));

    double* devred;
    cudaMalloc((void**)&devred, params.npix * sizeof(double));
    cudaMemcpy(devred, params.red, params.npix * sizeof(double), cudaMemcpyHostToDevice);

    double* devimd;
    cudaMalloc((void**)&devimd, params.npix * sizeof(double));
    cudaMemcpy(devimd, params.imd, params.npix * sizeof(double), cudaMemcpyHostToDevice);
    stop = clock();
    printf("Done with allocation: duration = %f\n", (double)(stop - start)/CLOCKS_PER_SEC);

    start = clock();
    Params devparams;
    devparams.red = devred;
    devparams.imd = devimd;
    devparams.nring = params.nring;
    devparams.npix = params.npix;

    const int threadsPerBlock = 128;
    int blocksPerGrid = (params.npix + threadsPerBlock - 1) / threadsPerBlock;
    compute_structdevres<<<blocksPerGrid, threadsPerBlock>>>(devparams, devres);
    cudaDeviceSynchronize();

    cudaError_t errSync  = cudaGetLastError();
    cudaError_t errAsync = cudaDeviceSynchronize();
    if (errSync != cudaSuccess) 
    printf("Sync kernel error: %s\n", cudaGetErrorString(errSync));
    if (errAsync != cudaSuccess)
    printf("Async kernel error: %s\n", cudaGetErrorString(errAsync));
    stop = clock();
    printf("Done with kernel: duration = %f\n", (double)(stop - start)/CLOCKS_PER_SEC);

    start = clock();
    cudaMemcpy(host_result, devres, params.npix * sizeof(double), cudaMemcpyDeviceToHost);
    stop = clock();
    printf("Done copying from GPU: duration = %f\n", (double)(stop - start)/CLOCKS_PER_SEC);
    printf("\nHost_result: ");
    for (int i = 0; i < params.npix; i+=1) {
        printf("%.2f, ", host_result[i]);
    }
    printf("\n");
}

std::tuple<Params,double*> CUpointing_structdevres11(Params params) {
    printf("CUpointing_structdevres11:: starting kernel\n");
    clock_t start, stop;
    double cpu_time_used;
    printf("params.nring: %d\n", params.nring);
    printf("params.npix: %d\n", params.npix);
    printf("params.red:\n");
    for (int i = 0; i < params.npix; i+=1) {
        printf("%.2f, ", params.red[i]);
    }printf("\n");
    printf("params.imd:\n");
    for (int i = 0; i < params.npix; i+=1) {
        printf("%.2f, ", params.imd[i]);
    }printf("\n");

    start = clock();

    double* devred;
    cudaMalloc((void**)&devred, params.npix * sizeof(double));
    cudaMemcpy(devred, params.red, params.npix * sizeof(double), cudaMemcpyHostToDevice);

    double* devimd;
    cudaMalloc((void**)&devimd, params.npix * sizeof(double));
    cudaMemcpy(devimd, params.imd, params.npix * sizeof(double), cudaMemcpyHostToDevice);

    double* devres;
    cudaMalloc((void**)&devres, params.npix * sizeof(double));
    stop = clock();
    printf("Done with allocation: duration = %f\n", (double)(stop - start)/CLOCKS_PER_SEC);

    Params devparams;
    devparams.red = devred;
    devparams.imd = devimd;
    devparams.nring = params.nring;
    devparams.npix = params.npix;

    return std::make_tuple(devparams,devres);
}

void CUpointing_structdevres12(Params hostparams, std::tuple<Params,double*> devtup) {
    printf("CUpointing_structdevres12:: starting kernel\n");
    clock_t start, stop;
    double cpu_time_used;
    start = clock();

    const int threadsPerBlock = 128;
    int blocksPerGrid = (hostparams.npix + threadsPerBlock - 1) / threadsPerBlock;
    compute_structdevres<<<blocksPerGrid, threadsPerBlock>>>(std::get<0>(devtup), std::get<1>(devtup));
    cudaDeviceSynchronize();

    cudaError_t errSync  = cudaGetLastError();
    cudaError_t errAsync = cudaDeviceSynchronize();
    if (errSync != cudaSuccess) 
    printf("Sync kernel error: %s\n", cudaGetErrorString(errSync));
    if (errAsync != cudaSuccess)
    printf("Async kernel error: %s\n", cudaGetErrorString(errAsync));
    stop = clock();
}

void CUpointing_structdevres13(Params hostparams, double* devres, double* host_result) {
    printf("CUpointing_structdevres13:: starting kernel\n");
    clock_t start, stop;
    double cpu_time_used;
    start = clock();
    cudaMemcpy(host_result, devres, hostparams.npix * sizeof(double), cudaMemcpyDeviceToHost);
    stop = clock();
    printf("Done copying from GPU: duration = %f\n", (double)(stop - start)/CLOCKS_PER_SEC);
    printf("\nHost_result: ");
    for (int i = 0; i < hostparams.npix; i+=1) {
        printf("%.2f, ", host_result[i]);
    }printf("\n");
}

double* CUpointing_structdevres3_retptr(Params params) {
    printf("CUpointing_structdevres3:: starting kernel\n");
    clock_t start, stop;
    double cpu_time_used;
    printf("params.nring: %d\n", params.nring);
    printf("params.npix: %d\n", params.npix);
    printf("params.red:\n");
    for (int i = 0; i < params.npix; i+=1) {
        printf("%.2f, ", params.red[i]);
    }printf("\n");
    printf("params.imd:\n");
    for (int i = 0; i < params.npix; i+=1) {
        printf("%.2f, ", params.imd[i]);
    }printf("\n");

    start = clock();

    double* devres;
    cudaMalloc((void**)&devres, params.npix * sizeof(double));

    double* devred;
    cudaMalloc((void**)&devred, params.npix * sizeof(double));
    cudaMemcpy(devred, params.red, params.npix * sizeof(double), cudaMemcpyHostToDevice);

    double* devimd;
    cudaMalloc((void**)&devimd, params.npix * sizeof(double));
    cudaMemcpy(devimd, params.imd, params.npix * sizeof(double), cudaMemcpyHostToDevice);
    stop = clock();
    printf("Done with allocation: duration = %f\n", (double)(stop - start)/CLOCKS_PER_SEC);

    start = clock();
    Params devparams;
    devparams.red = devred;
    devparams.imd = devimd;
    devparams.nring = params.nring;
    devparams.npix = params.npix;

    const int threadsPerBlock = 128;
    int blocksPerGrid = (params.npix + threadsPerBlock - 1) / threadsPerBlock;
    compute_structdevres<<<blocksPerGrid, threadsPerBlock>>>(devparams, devres);
    cudaDeviceSynchronize();

    cudaError_t errSync  = cudaGetLastError();
    cudaError_t errAsync = cudaDeviceSynchronize();
    if (errSync != cudaSuccess) 
    printf("Sync kernel error: %s\n", cudaGetErrorString(errSync));
    if (errAsync != cudaSuccess)
    printf("Async kernel error: %s\n", cudaGetErrorString(errAsync));
    stop = clock();
    printf("Done with kernel: duration = %f\n", (double)(stop - start)/CLOCKS_PER_SEC);
    return devres;
}