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
#include "pointing.cuh"

void CUfloat_to_double(const float* src, double* dest, int size) {
    for (int i = 0; i < size; i++) {
        dest[i] = (double)src[i];
    }
}

__device__ double dev_power_element(double value, int exponent){
    double result = exponent > 1 ? value : 1;
    for (int i = 1; i < exponent; i++) {
        result *= value;
    }
    return result;
}

__device__ void dev_besselj0(double* x, const int start, const int size, double* result) {
    double factorial, power, term;
    for (int i = start; i < size; i++) { 
        factorial = 1.0;
        power = 1.0;
        for (int k = 1; k < 25; k++) {
            factorial *= k;
            power *= (x[i] / 2.0) * (x[i] / 2.0);
            term = power / (factorial * factorial);
            result[i] += term * term;
        }
    }
}

__device__ void sindod_m1(double* d, int start, int size, double* result){
    for (int i = start; i < size; i++) {
        result[i] = 1. + -1./6. * d[i] * (1. - 1./20. * d[i] *(1. - 1./42. * d[i]));
        // result[i] = 1. + (-1./6. * d[i]*d[i] + 1./120. * d[i]*d[i]*d[i]*d[i] - 1./5040. * d[i]*d[i]*d[i]*d[i]*d[i]*d[i]);
    }
}

__device__ void dev_norm2(double* x, double* y, const int start, const int size, double* result) {
    for (int i = start; i < size; i++) {
        result[i] = x[i] * x[i] + y[i] * y[i];
    }
}

__device__ void dev_norm(double* x, double* y, const int start, const int size, double* result) {
    for (int i = start; i < size; i++) {
        result[i] = sqrt(x[i] * x[i] + y[i] * y[i]);
    }
}

__device__ int dev_isbigger(const double* arr, const int start, int size, const double threshold) {
    for (int i = start; i < size; i++) {
        if (arr[i] > threshold) {
            return 1;
        }
    }     return 0;
}


__device__ int dev_gettriquand(double theta){
    //"""Returns the version of the pointing computation"""
    return round(cos(theta)+0.5);
}

__device__ int* dev_arange(int start, int end){
    int size = (end - start);
    int* res = (int*)malloc(size * sizeof(int));
    for (int i = 0; i < size; i++) {
        res[i] = start + i;
    }
    return res;
}


bool allGreaterThanZero(double* array, int size) {
    for (int i = 0; i < size; i++) {
        if (array[i] <= 0.0) {
            return false;
        }
    }
    return true;
}

bool allLessThanPi(double* array, int size) {
    double PI = 3.14159265359;
    for (int i = 0; i < size; i++) {
        if (array[i] >= PI) {
            return false;
        }
    }
    return true;
}

int sum(int* array, int size) {
    printf("size: %d\n", size);
    int result = 0;
    for (int i = 0; i < size; i++) {
        result += array[i];
    }
    return result;
}

int* argsort(int* array, int size) {
    int* indices = (int*)malloc(size * sizeof(int));
    if (indices == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < size; i++) {
        indices[i] = i;
    }
    for (int i = 0; i < size - 1; i++) {
        for (int j = 0; j < size - i - 1; j++) {
            if (array[indices[j]] > array[indices[j + 1]]) {
                // Swap indices
                int temp = indices[j];
                indices[j] = indices[j + 1];
                indices[j + 1] = temp;
            }
        }
    }
    return indices;
}

__global__ void compute_pointing(KernelParams kp, KernelLocals kl, double *pointings) {
    //idx is nring
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    double PI = 3.14159265359;
    if (1 == 0) {
        if (idx <= kp.nring) {
            pointings[idx] = kp.ringstarts[idx];
        }
    } else {
        if (idx <= kp.nring) {
            const int ringstart = kp.ringstarts[idx];
            const int npixring = kp.nphis[idx];
            // kl.e_r(sin(theta(iring)), 0, cos(theta(iring)));
            kl.sint = sin(kp.thetas[idx]); 
            kl.cost = cos(kp.thetas[idx]);
            for (int i = ringstart; i < ringstart+npixring; i++) {
                kl.phi[i] = kp.phi0[idx] + i * (2. * PI / npixring);
            }
            for (int i = ringstart; i < ringstart+npixring; i++) {
                kl.d[i] = kp.red[i] * kp.red[i] + kp.imd[i] * kp.imd[i];
            }
            
            if (dev_isbigger(kl.d, ringstart, ringstart+npixring, 0.001)){
                for (int i = ringstart; i < ringstart+npixring; i++) {
                    kl.a[i] = sqrt(kl.d[i]);
                    kl.sind_d[i] = sin(kl.a[i]) / kl.a[i];
                    kl.cos_a[i] = cos(kl.a[i]);
                    kl.twohav_aod[i] = (kl.cos_a[i] - 1.) / kl.d[i];
                }
                
            } else {
                sindod_m1(kl.d, ringstart, ringstart+npixring, kl.sind_d);
                for (int i = ringstart; i < ringstart+npixring; i++) {
                    kl.twohav_aod[i] = -0.5 + kl.d[i]/24. * (1. - kl.d[i]/30. * (1. - kl.d[i]/56.));
                    kl.cos_a[i] = 1. + kl.d[i] * kl.twohav_aod[i];
                }
            }
            for (int i = ringstart; i < ringstart+npixring; i++) {
                kl.e_a1[i] = kl.cost * kp.red[i];
                kl.e_a2[i] = kl.phi[i];
                kl.e_a3[i] = -kl.sint * kp.red[i];
            }

            // kl.n_prime(kl.e_r * kl.cos_a + kl.e_a * kl.sin_d);
            for (int i = ringstart; i < ringstart+npixring; i++) {
                kl.np1[i] = kl.sint * kl.cos_a[i] + kl.e_a1[i] * kl.sind_d[i];
                kl.np2[i] = 0. - kl.e_a2[i] * kl.sind_d[i];
                kl.np3[i] = kl.cost * kl.cos_a[i] + kl.e_a3[i] * kl.sind_d[i];
            }

            //theta = std::atan2(sqrt(inp.x*inp.x+inp.y*inp.y),inp.z);
            //phi = safe_atan2 (inp.y,inp.x);
            //if (phi<0.) phi += twopi;
            for (int i = ringstart; i < ringstart+npixring; i++) {
                kl.npt[i] = atan2(sqrt(kl.np1[i]*kl.np1[i] + kl.np2[i] * kl.np2[i]), kl.np3[i]);
                kl.npp[i] = atan2(kl.np2[i], kl.np1[i]);
                kl.npp[i] = (kl.npp[i] < 0.) ? (kl.npp[i] + 2.*PI) : kl.npp[i];
            }
            
            // kl.phinew = (kl.phinew >= 2.*PI) ? (kl.phinew - 2.*PI) : kl.phinew;
            for (int i = ringstart; i < ringstart+npixring; i++) {
                pointings[i] = kl.npt[i];
                pointings[i + kp.npix] = kl.npp[i] + kl.phi[i];
                pointings[i + kp.npix] = (pointings[i + kp.npix] >= 2*PI) ? (pointings[i + kp.npix] - 2.*PI) : pointings[i + kp.npix];
            }
        }
    }
}


double* CUpointing_struct(KernelParams hostkp) {
    printf("CUpointing_exec:: starting kernel\n");
    clock_t start, stop;
    double cpu_time_used;
    start = clock();

    KernelLocals kl;
    double *dev_sint;
    double *dev_cost;
    double *dev_phi;
    double *dev_sind_d, *dev_a, *dev_d;
    double *dev_cos_a, *dev_twohav_aod;
    double *dev_e_a1, *dev_e_a2, *dev_e_a3;
    double *dev_np1, *dev_np2, *dev_np3;
    double *dev_npt, *dev_npp;
    double* dev_philocs;

    cudaMalloc((void**)&dev_phi, hostkp.npix * sizeof(double));
    cudaMalloc((void**)&dev_sind_d, hostkp.npix * sizeof(double));
    cudaMalloc((void**)&dev_a, hostkp.npix * sizeof(double));
    cudaMalloc((void**)&dev_d, hostkp.npix * sizeof(double));
    cudaMalloc((void**)&dev_cos_a, hostkp.npix * sizeof(double));
    cudaMalloc((void**)&dev_twohav_aod, hostkp.npix * sizeof(double));
    cudaMalloc((void**)&dev_e_a1, hostkp.npix * sizeof(double));
    cudaMalloc((void**)&dev_e_a2, hostkp.npix * sizeof(double));
    cudaMalloc((void**)&dev_e_a3, hostkp.npix * sizeof(double));
    cudaMalloc((void**)&dev_np1, hostkp.npix * sizeof(double));
    cudaMalloc((void**)&dev_np2, hostkp.npix * sizeof(double));
    cudaMalloc((void**)&dev_np3, hostkp.npix * sizeof(double));
    cudaMalloc((void**)&dev_npt, hostkp.npix * sizeof(double));
    cudaMalloc((void**)&dev_npp, hostkp.npix * sizeof(double));
    cudaMalloc((void**)&dev_philocs, hostkp.npix * sizeof(double));

    kl.phi = dev_phi;
    kl.sind_d = dev_sind_d;
    kl.a = dev_a;
    kl.d = dev_d;
    kl.cos_a = dev_cos_a;
    kl.twohav_aod = dev_twohav_aod;
    kl.e_a1 = dev_e_a1;
    kl.e_a2 = dev_e_a2;
    kl.e_a3 = dev_e_a3;
    kl.np1 = dev_np1;
    kl.np2 = dev_np2;
    kl.np3 = dev_np3;
    kl.npt = dev_npt;
    kl.npp = dev_npp;
    kl.philocs = dev_philocs;

    printf("hostkp.thetas: %p\n", hostkp.thetas);
    printf("hostkp.phi0: %p\n", hostkp.phi0);
    printf("hostkp.nphis: %p\n", hostkp.nphis);
    printf("hostkp.ringstarts: %p\n", hostkp.ringstarts);
    printf("hostkp.red: %p\n", hostkp.red);
    printf("hostkp.imd: %p\n", hostkp.imd);

    KernelParams devkp;
    // cudaMalloc((void**)&devkp, sizeof(KernelParams));
    // cudaMemcpy(devkp, (void*)&hostkp, sizeof(KernelParams), cudaMemcpyHostToDevice);

    devkp.thetas = hostkp.thetas;
    devkp.phi0 = hostkp.phi0;
    devkp.nphis = hostkp.nphis;
    devkp.ringstarts = hostkp.ringstarts;
    devkp.red = hostkp.red;
    devkp.imd = hostkp.imd;
    devkp.nring = hostkp.nring;
    devkp.npix = hostkp.npix;


    double *dev_result;
    cudaMalloc((void**)&dev_result, 2*hostkp.npix * sizeof(double));

    const int threadsPerBlock = 128;
    int blocksPerGrid = (hostkp.nring + threadsPerBlock - 1) / threadsPerBlock;
    compute_pointing<<<blocksPerGrid, threadsPerBlock>>>(devkp, kl, dev_result);
    cudaDeviceSynchronize();

    cudaError_t errSync  = cudaGetLastError();
    cudaError_t errAsync = cudaDeviceSynchronize();
    if (errSync != cudaSuccess) 
    printf("Sync kernel error: %s\n", cudaGetErrorString(errSync));
    if (errAsync != cudaSuccess)
    printf("Async kernel error: %s\n", cudaGetErrorString(errAsync));
    stop = clock();
    printf("Done with kernel: duration = %f\n", (double)(stop - start)/CLOCKS_PER_SEC);

    cudaFree(kl.phi);
    cudaFree(kl.sind_d);
    cudaFree(kl.a);
    cudaFree(kl.d);
    cudaFree(kl.cos_a);
    cudaFree(kl.twohav_aod);
    cudaFree(kl.e_a1);
    cudaFree(kl.e_a2);
    cudaFree(kl.e_a3);
    cudaFree(kl.np1);
    cudaFree(kl.np2);
    cudaFree(kl.np3);
    cudaFree(kl.npt);
    cudaFree(kl.npp);
    cudaFree(kl.philocs);

    cudaFree(devkp.thetas);
    cudaFree(devkp.phi0);
    cudaFree(devkp.nphis);
    cudaFree(devkp.ringstarts);
    cudaFree(devkp.red);
    cudaFree(devkp.imd);
    
    return dev_result;
}