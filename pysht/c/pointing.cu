#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <cmath>
#include <chrono>
#include <time.h>

#include "pointing.h"
#include "pointing.cuh"

struct KernelParams {
    double *thetalocs;
    double *philocs;
    double *phi0;
    int *nphis;
    int *ringstarts;
    double *red;
    double *imd;
    int nrings;
    int npix;
    
};

struct KernelLocals {
    double *sind_d;
    double *dphi;
    double *thtp;
    double *e_d, *e_t, *e_tp;
    double *d;
};

struct KernelLocalsDUCC {
    double sint;
    double cost;
    double *phi;
    double *sind_d, *a, *d;
    double *cos_a, *twohav_aod;
    double *e_a1, *e_a2, *e_a3;
    double *np1, *np2, *np3;
    double *npt, *npp;
};

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

__device__ void dev_norm2(double* x, double* y, const int size, double* result) {
    for (int i = 0; i < size; i++) {
        result[i] = x[i] * x[i] + y[i] * y[i];
    }
}

__device__ void dev_norm(double* x, double* y, const int size, double* result) {
    for (int i = 0; i < size; i++) {
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

__global__ void compute_pointing_DUCC(KernelParams kp, KernelLocalsDUCC kl, double *pointings) {
    //idx is nrings
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    double PI = 3.14159265359;

    if (idx <= kp.nrings) {
        const int ringstart = kp.ringstarts[idx];
        const int npixring = kp.nphis[idx];
        // kl.e_r(sin(theta(iring)), 0, cos(theta(iring)));
        kl.sint = sin(kp.thetalocs[idx]); 
        kl.cost = cos(kp.thetalocs[idx]);
        for (int i = ringstart; i < ringstart+npixring; i++) {
            kl.phi[i] = kp.phi0[idx] + i * (2. * PI / npixring);
        }
        for (int i = ringstart; i < ringstart+npixring; i++) {
            kl.d[i] = kp.red[i] * kp.red[i] + kp.imd[i] * kp.imd[i];
        }
        // dev_norm2(kp.red, kp.imd, kp.npix, kl.d);
        
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

__global__ void compute_pointing(KernelParams kp, KernelLocals kl, double *pointings) {
    //idx is nrings
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    double PI = 3.14159265359;

    if (idx <= kp.nrings) {
        const int ringstart = kp.ringstarts[idx];
        const int npixring = kp.nphis[idx];
        int triquand = round(cos(kp.thetalocs[idx]));

        for (int i = ringstart; i < ringstart+npixring; i++) {
            kp.philocs[i] = fmod(kp.phi0[idx] + i * (2. * PI / npixring), 2.*PI);
        }
        //TODO implement this for correct offset
        // phis = phis//[pixs - self.geom.ofs[ir]]

        dev_norm(kp.red, kp.imd, kp.npix, kl.d);

        if (dev_isbigger(kl.d, ringstart, ringstart+npixring, 0.001)){
            dev_besselj0(kl.d, ringstart, ringstart+npixring, kl.sind_d);
        } else {
            sindod_m1(kl.d, ringstart, ringstart+npixring, kl.sind_d);
        }
        if (triquand == 0){ // #---'close' to equator, where cost ~ 0 
            for (int i = ringstart; i < ringstart+npixring; i++) {
                kl.dphi[i] = asin(kp.imd[i] / sqrt(1. - cos(kp.thetalocs[idx])*cos(kp.thetalocs[idx])) * kl.sind_d[i]);
                kl.thtp[i] = acos(cos(kp.thetalocs[idx]) * cos(sqrt(kl.d[i])) - kp.red[i] * kl.sind_d[i] * sqrt(1. - cos(kp.thetalocs[idx])*cos(kp.thetalocs[idx])));
            }
        } else {
            int isnorth = triquand == 1 ? 1 : 0;
            if (isnorth == 1){
                for (int i = ringstart; i < ringstart+npixring; i++) {
                    kl.e_t[i] = 2. * sin(kp.thetalocs[idx] * 0.5)*sin(kp.thetalocs[idx] * 0.5);
                }
            } else {
                for (int i = ringstart; i < ringstart+npixring; i++) {
                    kl.e_t[i] = 2. * cos(kp.thetalocs[idx] * 0.5)*cos(kp.thetalocs[idx] * 0.5);
                }
            }
            for (int i = ringstart; i < ringstart+npixring; i++) {
                kl.e_d[i] = 2. * sin(sqrt(kl.d[i]) * 0.5)*sin(sqrt(kl.d[i]) * 0.5);
                kl.e_tp[i] = kl.e_t[i] + kl.e_d[i] - kl.e_t[i] * kl.e_d[i] + (double)triquand * kp.red[i] * kl.sind_d[i] * sin(kp.thetalocs[idx]);
                kl.thtp[i] = asin(sqrt(max(kl.e_tp[i] * (2. - kl.e_tp[i]), 0.)));
            }
            if (isnorth == 1){
                for (int i = ringstart; i < ringstart+npixring; i++) {
                    kl.dphi[i] = atan2(kp.imd[i] * kl.sind_d[i], (1. - kl.e_d[i]) * sin(kp.thetalocs[idx]) + kp.red[i] * kl.sind_d[i] * (1. - kl.e_t[i])); // TODO possible x/y confusion
                }
            } else {
                for (int i = ringstart; i < ringstart+npixring; i++) {
                    kl.thtp[i] = PI - kl.thtp[i];
                    kl.dphi[i] = atan2(kp.imd[i] * kl.sind_d[i], (1. - kl.e_d[i]) * sin(kp.thetalocs[idx]) + kp.red[i] * kl.sind_d[i] * (kl.e_t[i] - 1.)); // TODO possible x/y confusion
                }
            }
        }
        for (int i = ringstart; i < ringstart+npixring; i++) {
            kl.dphi[i] = fmod(kp.philocs[i] + kl.dphi[i], 2. * PI);
        }
        for (int i = ringstart; i < ringstart+npixring; i++) {
            pointings[i] = kl.thtp[i];
            pointings[i + kp.npix] = kl.dphi[i];
            // TODO implement this (rotation of the polarization angle)
            // cot = np.cos(self.geom.theta[ir]) / np.sin(self.geom.theta[ir])
            // d = np.sqrt(t_red ** 2 + i_imd ** 2)
            // thp_phip_gamma[2, sli] = np.arctan2(i_imd, t_red ) - np.arctan2(i_imd, d * np.sin(d) * cot + t_red * np.cos(d))
        }
    }
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

void float_to_double(const float* src, double* dest, int size) {
    for (int i = 0; i < size; i++) {
        dest[i] = (double)src[i];
    }
}

void CUpointing_lenspyx(float* thetas_, float* phi0_, int* nphis, int* ringstarts, double *red, double *imd, int nrings, int npix, double *host_result) {
    // printf("nrings: %d, npix: %d\nthetas: ", nrings, npix);
    // for (int i = 0; i < nrings; i+=32) {
    //     printf("%f ", thetas_[i]);
    // }printf("\nphi0: ");
    // for (int i = 0; i < nrings; i+=32) {
    //     printf("%f ", phi0_[i]);
    // }printf("\nnphis: ");
    // for (int i = 0; i < nrings; i+=32) {
    //     printf("%d ", nphis[i]);
    // }printf("\nringstarts: ");
    // for (int i = 0; i < nrings; i+=32) {
    //     printf("%d ", ringstarts[i]);
    // }printf("\nred: ");
    // for (int i = 0; i < npix; i+=100000) {
    //     printf("%f ", red[i]);
    // }printf("\nimd: ");
    // for (int i = 0; i < npix; i+=100000) {
    //     printf("%f ", imd[i]);
    // }printf("\n");
    printf("starting allocation\n");
    double *thetas = (double*)malloc(nrings * sizeof(double));
    double *phi0 = (double*)malloc(nrings * sizeof(double));
    float_to_double(phi0_, phi0, nrings);
    float_to_double(thetas_, thetas, nrings);


    // bool condition1 = allGreaterThanZero(thetas, nrings);
    // bool condition2 = allLessThanPi(thetas, nrings);

    // printf("condition1: %d\n", condition1);
    // printf("condition2: %d\n", condition2);
    // assert(condition1 && condition2);

    KernelParams params;
    double *device_thetalocs, *device_phi0, *device_red, *device_imd, *device_philocs;
    int *device_nphis, *device_ringstarts;
    cudaMalloc((void**)&device_thetalocs, nrings * sizeof(double));
    cudaMalloc((void**)&device_phi0, nrings * sizeof(double));
    cudaMalloc((void**)&device_nphis, nrings * sizeof(int));
    cudaMalloc((void**)&device_ringstarts, nrings * sizeof(int));
    cudaMalloc((void**)&device_red, npix * sizeof(double));
    cudaMalloc((void**)&device_imd, npix * sizeof(double));
    cudaMalloc((void**)&device_philocs, npix * sizeof(double));

    cudaMemcpy(device_thetalocs, thetas, nrings * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(device_phi0, phi0, nrings * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(device_nphis, nphis, nrings * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(device_ringstarts, ringstarts, nrings * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(device_red, red, npix * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(device_imd, imd, npix * sizeof(double), cudaMemcpyHostToDevice);
    //cudaMemcpy(device_philocs, imd, npix * sizeof(double), cudaMemcpyHostToDevice); // dummy values, correct values will be set in the kernel
    
    params.thetalocs = device_thetalocs;
    params.philocs = device_philocs;
    params.phi0 = device_phi0;
    params.nphis = device_nphis;
    params.ringstarts = device_ringstarts;
    params.red = device_red;
    params.imd = device_imd;
    params.nrings = nrings;
    params.npix = npix;


    KernelLocals locals;
    double *device_sind_d, *device_dphi, *device_thtp, *device_e_d, *device_e_t, *device_e_tp, *device_d;

    cudaMalloc((void**)&device_sind_d, npix * sizeof(double));
    cudaMalloc((void**)&device_dphi, npix * sizeof(double));
    cudaMalloc((void**)&device_thtp, npix * sizeof(double));
    cudaMalloc((void**)&device_e_d, npix * sizeof(double));
    cudaMalloc((void**)&device_e_t, npix * sizeof(double));
    cudaMalloc((void**)&device_e_tp, npix * sizeof(double));
    cudaMalloc((void**)&device_d, npix * sizeof(double));

    locals.sind_d = device_sind_d;
    locals.dphi = device_dphi;
    locals.thtp = device_thtp;
    locals.e_d = device_e_d;
    locals.e_t = device_e_t;
    locals.e_tp = device_e_tp;
    locals.d = device_d;

    double *device_result;
    cudaMalloc((void**)&device_result, 2*npix * sizeof(double));

    const int threadsPerBlock = 256;
    int blocksPerGrid = (nrings + threadsPerBlock - 1) / threadsPerBlock;
    printf("Calling kernel\n ");
    compute_pointing<<<blocksPerGrid, threadsPerBlock>>>(params, locals, device_result);
    cudaDeviceSynchronize();
    printf("Done with kernel\n ");
    cudaMemcpy(host_result, device_result, 2*npix * sizeof(double), cudaMemcpyDeviceToHost);
    printf("Done grabbing\n ");

    // cudaError_t errSync  = cudaGetLastError();
    // cudaError_t errAsync = cudaDeviceSynchronize();
    // if (errSync != cudaSuccess) 
    // printf("Sync kernel error: %s\n", cudaGetErrorString(errSync));
    // if (errAsync != cudaSuccess)
    // printf("Async kernel error: %s\n", cudaGetErrorString(errAsync));

//     printf("\npointing theta: ");
//     for (int i = npix-1000; i < npix; i+=1) {
//         printf("%.2f ", host_result[i]);
//     }printf("\npointing phi: ");
//    for (int i = 2*npix-1000; i < 2*npix; i+=1) {
//         printf("%.2f ", host_result[i]);
//     }printf("\n");

    cudaFree(device_thetalocs);
    cudaFree(device_phi0);
    cudaFree(device_nphis);
    cudaFree(device_ringstarts);
    cudaFree(device_red);
    cudaFree(device_imd);
    cudaFree(device_result);
    cudaFree(device_sind_d);
    cudaFree(device_dphi);
    cudaFree(device_thtp);
    cudaFree(device_e_d);
    cudaFree(device_e_t);
    cudaFree(device_e_tp);
    cudaFree(device_d);
    free(thetas);
    free(phi0);
}


void CUpointing_DUCC(float* thetas_, float* phi0_, int* nphis, int* ringstarts, double *red, double *imd, int nrings, int npix, double *host_result) {
    printf("starting allocation\n");
    clock_t start, stop;
    double cpu_time_used;
    start = clock();

    double *thetas = (double*)malloc(nrings * sizeof(double));
    double *phi0 = (double*)malloc(nrings * sizeof(double));
    float_to_double(phi0_, phi0, nrings);
    float_to_double(thetas_, thetas, nrings);

    KernelParams params;
    double *device_thetalocs, *device_phi0, *device_red, *device_imd, *device_philocs;
    int *device_nphis, *device_ringstarts;
    cudaMalloc((void**)&device_thetalocs, nrings * sizeof(double));
    cudaMalloc((void**)&device_phi0, nrings * sizeof(double));
    cudaMalloc((void**)&device_nphis, nrings * sizeof(int));
    cudaMalloc((void**)&device_ringstarts, nrings * sizeof(int));
    cudaMalloc((void**)&device_red, npix * sizeof(double));
    cudaMalloc((void**)&device_imd, npix * sizeof(double));
    cudaMalloc((void**)&device_philocs, npix * sizeof(double));

    cudaMemcpy(device_thetalocs, thetas, nrings * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(device_phi0, phi0, nrings * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(device_nphis, nphis, nrings * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(device_ringstarts, ringstarts, nrings * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(device_red, red, npix * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(device_imd, imd, npix * sizeof(double), cudaMemcpyHostToDevice);
    //cudaMemcpy(device_philocs, imd, npix * sizeof(double), cudaMemcpyHostToDevice); // dummy values, correct values will be set in the kernel
    
    params.thetalocs = device_thetalocs;
    params.philocs = device_philocs;
    params.phi0 = device_phi0;
    params.nphis = device_nphis;
    params.ringstarts = device_ringstarts;
    params.red = device_red;
    params.imd = device_imd;
    params.nrings = nrings;
    params.npix = npix;


    KernelLocalsDUCC locals;
    double *dev_sint;
    double *dev_cost;
    double *dev_phi;
    double *dev_sind_d, *dev_a, *dev_d;
    double *dev_cos_a, *dev_twohav_aod;
    double *dev_e_a1, *dev_e_a2, *dev_e_a3;
    double *dev_np1, *dev_np2, *dev_np3;
    double *dev_npt, *dev_npp;

    cudaMalloc((void**)&dev_phi, npix * sizeof(double));
    cudaMalloc((void**)&dev_sind_d, npix * sizeof(double));
    cudaMalloc((void**)&dev_a, npix * sizeof(double));
    cudaMalloc((void**)&dev_d, npix * sizeof(double));
    cudaMalloc((void**)&dev_cos_a, npix * sizeof(double));
    cudaMalloc((void**)&dev_twohav_aod, npix * sizeof(double));
    cudaMalloc((void**)&dev_e_a1, npix * sizeof(double));
    cudaMalloc((void**)&dev_e_a2, npix * sizeof(double));
    cudaMalloc((void**)&dev_e_a3, npix * sizeof(double));
    cudaMalloc((void**)&dev_np1, npix * sizeof(double));
    cudaMalloc((void**)&dev_np2, npix * sizeof(double));
    cudaMalloc((void**)&dev_np3, npix * sizeof(double));
    cudaMalloc((void**)&dev_npt, npix * sizeof(double));
    cudaMalloc((void**)&dev_npp, npix * sizeof(double));

    locals.phi = dev_phi;
    locals.sind_d = dev_sind_d;
    locals.a = dev_a;
    locals.d = dev_d;
    locals.cos_a = dev_cos_a;
    locals.twohav_aod = dev_twohav_aod;
    locals.e_a1 = dev_e_a1;
    locals.e_a2 = dev_e_a2;
    locals.e_a3 = dev_e_a3;
    locals.np1 = dev_np1;
    locals.np2 = dev_np2;
    locals.np3 = dev_np3;
    locals.npt = dev_npt;
    locals.npp = dev_npp;
    
    double *device_result;
    cudaMalloc((void**)&device_result, 2*npix * sizeof(double));

    const int threadsPerBlock = 128;
    int blocksPerGrid = (nrings + threadsPerBlock - 1) / threadsPerBlock;
    stop = clock();
    printf("Allocation done: duration = %f\n ", (double)(stop - start)/CLOCKS_PER_SEC);
    printf("Calling kernel\n ");

    start = clock();
    compute_pointing_DUCC<<<blocksPerGrid, threadsPerBlock>>>(params, locals, device_result);
    cudaDeviceSynchronize();
    stop = clock();
    printf("Done with kernel: duration = %f\n ", (double)(stop - start)/CLOCKS_PER_SEC);

    start = clock();
    cudaMemcpy(host_result, device_result, 2*npix * sizeof(double), cudaMemcpyDeviceToHost);
    stop = clock();
    printf("Done grabbing: duration = %f\n ", (double)(stop - start)/CLOCKS_PER_SEC);

    // cudaError_t errSync  = cudaGetLastError();
    // cudaError_t errAsync = cudaDeviceSynchronize();
    // if (errSync != cudaSuccess) 
    // printf("Sync kernel error: %s\n", cudaGetErrorString(errSync));
    // if (errAsync != cudaSuccess)
    // printf("Async kernel error: %s\n", cudaGetErrorString(errAsync));

//     printf("\npointing theta: ");
//     for (int i = npix-1000; i < npix; i+=1) {
//         printf("%.2f ", host_result[i]);
//     }printf("\npointing phi: ");
//    for (int i = 2*npix-1000; i < 2*npix; i+=1) {
//         printf("%.2f ", host_result[i]);
//     }printf("\n");

    cudaFree(dev_a);
    cudaFree(dev_cos_a);
    cudaFree(dev_cost);
    cudaFree(dev_d);
    cudaFree(dev_e_a1);
    cudaFree(dev_e_a2);
    cudaFree(dev_e_a3);
    cudaFree(dev_npp);
    cudaFree(dev_np1);
    cudaFree(dev_np2);
    cudaFree(dev_np3);
    cudaFree(dev_phi);
    cudaFree(dev_sind_d);
    cudaFree(dev_sint);
    cudaFree(dev_twohav_aod);
    cudaFree(device_thetalocs);
    cudaFree(device_phi0);
    cudaFree(device_nphis);
    cudaFree(device_ringstarts);
    cudaFree(device_red);
    cudaFree(device_imd);
    cudaFree(device_result);
    free(thetas);
    free(phi0);
}