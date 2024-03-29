#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <cmath>


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
        for (int k = 1; k < 75; k++) {
            factorial *= k;
            power *= (x[i] / 2.0) * (x[i] / 2.0);
            term = power / (factorial * factorial);
            result[i] += term * term;
        }
    }
}

__device__ void sindod_m1(double* d, int start, int size, double* result){
    for (int i = start; i < size; i++) {
        // np.poly1d([ -1. / 5040., 1. / 120. -1 / 6.])(d2)
        result[i] = 1. + (-1./6. * dev_power_element(d[i],2) + 1./120. * dev_power_element(d[i],4) - 1./5040. * dev_power_element(d[i],6));
    }
}

__device__ void dev_norm(double* x, double* y, const int size, double* result) {
    for (int i = 0; i < size; i++) {
        result[i] = sqrt(x[i] * x[i] + y[i] * y[i]);
    }
}

__device__ int dev_isbigger(const double* arr, const int start, int size, const double threshold) {
    double max;
    if (arr[start] > threshold) {
        return 1;
    } else {
        max = arr[start];
        for (int i = start+1; i < size; i++) {
            if (arr[i] > max) {
                max = arr[i];
                if (max > threshold) {
                    return 1;
                }            }        }    }
    return 0;
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

        if (dev_isbigger(kl.d, ringstart, ringstart+npixring, 0.01)){
            dev_besselj0(kl.d, ringstart, ringstart+npixring, kl.sind_d);
        } else {
            sindod_m1(kl.d, ringstart, ringstart+npixring, kl.sind_d);
        }
        if (triquand == 0){ // #---'close' to equator, where cost ~ 0 
            if (cos(kp.thetalocs[idx]) > 0.8) {
                printf("wrong localization: %f\n", cos(kp.thetalocs[idx]));
            }
            for (int i = ringstart; i < ringstart+npixring; i++) {
                kl.dphi[i] = asin(kp.imd[i] / sqrt(1. - dev_power_element(cos(kp.thetalocs[idx]),2)) * kl.sind_d[i]);
                kl.thtp[i] = acos(cos(kp.thetalocs[idx]) * cos(kl.d[i]) - kp.red[i] * kl.sind_d[i] * sqrt(1. - dev_power_element(cos(kp.thetalocs[idx]),2)));
            }
        } else {
            int isnorth = triquand == 1 ? 1 : 0;
            for (int i = ringstart; i < ringstart+npixring; i++) {
                if (isnorth == 1){
                    kl.e_t[i] = 2. * dev_power_element(sin(kp.thetalocs[idx] * 0.5),2);
                } else {
                    kl.e_t[i] = 2. * dev_power_element(cos(kp.thetalocs[idx] * 0.5),2);
                }
                kl.e_d[i] = 2. * dev_power_element(sin(kl.d[i] * 0.5),2);
                kl.e_tp[i] = kl.e_t[i] + kl.e_d[i] - kl.e_t[i] * kl.e_d[i] + (double)triquand * kp.red[i] * kl.sind_d[i] * sin(kp.thetalocs[idx]);
                kl.thtp[i] = asin(sqrt(max(kl.e_tp[i] * (2. - kl.e_tp[i]), 0.)));
            }
            if (isnorth == 1){
                //assert np.max(tht) < np.pi * 0.4, ('wrong localization', np.max(tht)); //# -- for the arcsin at the end
                for (int i = ringstart; i < ringstart+npixring; i++) {
                    // kl.dphi[i] = atan2((1. - kl.e_d[i]) * sin(kp.thetalocs[idx]) + kp.red[i] * kl.sind_d[i] * (1. - kl.e_t[i]), kp.imd[i] * kl.sind_d[i]);
                    kl.dphi[i] = atan2(kp.imd[i] * kl.sind_d[i], (1. - kl.e_d[i]) * sin(kp.thetalocs[idx]) + kp.red[i] * kl.sind_d[i] * (1. - kl.e_t[i])); // TODO possible x/y confusion
                }
            } else {
                //assert np.min(tht) > np.pi * 0.4, ('wrong localization', np.min(tht)); //# -- for the arcsin at the end
                for (int i = ringstart; i < ringstart+npixring; i++) {
                    kl.thtp[i] = PI - kl.thtp[i];
                    // kl.dphi[i] = atan2((1. - kl.e_d[i]) * sin(kp.thetalocs[idx]) + kp.red[i] * kl.sind_d[i] * (kl.e_t[i] - 1.), kp.imd[i] * kl.sind_d[i]);
                    kl.dphi[i] = atan2(kp.imd[i] * kl.sind_d[i], (1. - kl.e_d[i]) * sin(kp.thetalocs[idx]) + kp.red[i] * kl.sind_d[i] * (kl.e_t[i] - 1.)); // TODO possible x/y confusion
                }
            }
        }
        for (int i = ringstart; i < ringstart+npixring; i++) {
            kl.dphi[i] = fmod(kp.philocs[i] + kl.dphi[i], 2. * PI);
        }
        // __syncthreads();
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


extern "C" void pointing(float* thetas_, float* phi0_, int* nphis, int* ringstarts, double *red, double *imd, int nrings, int npix, double *host_result) {
    printf("nrings: %d, npix: %d\nthetas: ", nrings, npix);
    for (int i = 0; i < nrings; i+=32) {
        printf("%f ", thetas_[i]);
    }printf("\nphi0: ");
    for (int i = 0; i < nrings; i+=32) {
        printf("%f ", phi0_[i]);
    }printf("\nnphis: ");
    for (int i = 0; i < nrings; i+=32) {
        printf("%d ", nphis[i]);
    }printf("\nringstarts: ");
    for (int i = 0; i < nrings; i+=32) {
        printf("%d ", ringstarts[i]);
    }printf("\nred: ");
    for (int i = 0; i < npix; i+=100000) {
        printf("%f ", red[i]);
    }printf("\nimd: ");
    for (int i = 0; i < npix; i+=100000) {
        printf("%f ", imd[i]);
    }printf("\n");

    double *thetas = (double*)malloc(nrings * sizeof(double));
    double *phi0 = (double*)malloc(nrings * sizeof(double));
    float_to_double(phi0_, phi0, nrings);
    float_to_double(thetas_, thetas, nrings);


    bool condition1 = allGreaterThanZero(thetas, nrings);
    bool condition2 = allLessThanPi(thetas, nrings);

    printf("condition1: %d\n", condition1);
    printf("condition2: %d\n", condition2);
    assert(condition1 && condition2);

    int* sorted_ringstarts = argsort(ringstarts, nrings);

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

    cudaMemset(device_sind_d, 0, npix * sizeof(double));
    cudaMemset(device_dphi, 0, npix * sizeof(double));
    cudaMemset(device_thtp, 0, npix * sizeof(double));
    cudaMemset(device_e_d, 0, npix * sizeof(double));
    cudaMemset(device_e_t, 0, npix * sizeof(double));
    cudaMemset(device_e_tp, 0, npix * sizeof(double));
    cudaMemset(device_d, 0, npix * sizeof(double));


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
    compute_pointing<<<blocksPerGrid, threadsPerBlock>>>(params, locals, device_result);
    cudaDeviceSynchronize();
    cudaMemcpy(host_result, device_result, 2*npix * sizeof(double), cudaMemcpyDeviceToHost);

    cudaError_t errSync  = cudaGetLastError();
    cudaError_t errAsync = cudaDeviceSynchronize();
    if (errSync != cudaSuccess) 
    printf("Sync kernel error: %s\n", cudaGetErrorString(errSync));
    if (errAsync != cudaSuccess)
    printf("Async kernel error: %s\n", cudaGetErrorString(errAsync));

    printf("\npointing theta: ");
    for (int i = npix-1000; i < npix; i+=1) {
        printf("%.2f ", host_result[i]);
    }printf("\npointing phi: ");
   for (int i = 2*npix-1000; i < 2*npix; i+=1) {
        printf("%.2f ", host_result[i]);
    }printf("\n");

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
    free(sorted_ringstarts);
}