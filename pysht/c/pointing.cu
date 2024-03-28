#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <cmath>


__device__ double dev_power_element(double value, int exponent){
    double result = exponent > 1 ? value : 1;
    for (int i = 1; i < exponent; i++) {
        result *= value;
    }
    return result;
}

__device__ double* dev_sin(double* arr, const int size){
    double* result = (double*)malloc(size * sizeof(double));
    for (int i = 0; i < size; i++) {
        result[i] = sin(arr[i]);
    }
    return result;
}

__device__ void maximum(const double* arr1, const double* arr2, double* result, const int size) {
    for (int i = 0; i < size; i++) {
        result[i] = arr1[i] > arr2[i] ? arr1[i] : arr2[i];
    }
}

__device__ double* dev_cos(double* arr, const int size){
    double* result = (double*)malloc(size * sizeof(double));
    for (int i = 0; i < size; i++) {
        result[i] = cos(arr[i]);
    }
    return result;
}

__device__ double* dev_asin(double* arr, const int size){
    double* result = (double*)malloc(size * sizeof(double));
    for (int i = 0; i < size; i++) {
        result[i] = asin(arr[i]);
    }
    return result;
}

__device__ double* dev_acos(double* arr, const int size){
    double* result = (double*)malloc(size * sizeof(double));
    for (int i = 0; i < size; i++) {
        result[i] = acos(arr[i]);
    }
    return result;
}

__device__ double* dev_atan2(double* arr, double* arr2, const int size){
    double* result = (double*)malloc(size * sizeof(double));
    for (int i = 0; i < size; i++) {
        result[i] = atan2(arr[1], arr2[i]);
    }
    return result;
}

__device__ int* ring2pixs(int ringstart, int nphi) {
    int* concatenated = (int*)malloc(nphi * sizeof(int));
    for (int i = 0; i < nphi; i++) {
        concatenated[i] = ringstart + i;
    }
    return concatenated;
}

__device__ double* dev_besselj0(double* x, const int size) {
    double* result = (double*)malloc(size * sizeof(double));
    double sum, factorial, power, term;
    for (int i = 0; i < size; i++) { 
        sum = 1.0;
        factorial = 1.0;
        power = 1.0;
        for (int k = 1; k < 50; k++) {
            factorial *= k;
            power *= (x[i] / 2.0) * (x[i] / 2.0);
            term = power / (factorial * factorial);
            sum += term * term;
        }
        result[i] = sum;
    }
    return result;
}

__device__ double* getelements(double* arr, int* indices, const int size_indices){
    double* result = (double*)malloc(size_indices * sizeof(double));
    for (int i = 0; i < size_indices; i++) {
        result[i] = arr[indices[i]];
    }
    return result;
}

__device__ double* sindod_m1(double* d, const int size){
    double* result = (double*)malloc(size * sizeof(double));
    double* d2 = (double*)malloc(size * sizeof(double));
    for (int i = 0; i < size; i++) {
        d2[i] = d[i] * d[i];
    }
    for (int i = 0; i < size; i++) {
        result[i] = -1./6. * dev_power_element(d2[i],2) + 1./120. * dev_power_element(d2[i],4) - 1./5040. * dev_power_element(d2[i],6);
    }
    return result;
}

__device__ double* dev_norm(double* x, double* y, const int size) {
    double* result = (double*)malloc(size * sizeof(double));
    for (int i = 0; i < size; i++) {
        result[i] = sqrt(x[i] * x[i] + y[i] * y[i]);
    }
    return result;
}

__device__ double dev_max(const double* arr, int size) {
    double max = arr[0];
    for (int i = 1; i < size; i++) {
        if (arr[i] > max) {
            max = arr[i];
        }
    }
    return max;
}

__device__ double* d2ang(double* red, double* imd, double* tht, double* phi, int triquand){
    const int npix = sizeof(red) / sizeof(red[0]);
    double *d = dev_norm(red, imd, npix);
    double* sind_d = (double*)malloc(npix * sizeof(double));
    double PI = 3.14159265359;
    double *dphi = (double*)malloc(npix * sizeof(double));
    double *thtp = (double*)malloc(npix * sizeof(double));
    if (dev_max(d, npix) > 0.01){
        sind_d = dev_besselj0(d, npix);
    } else {
        double *buffer = sindod_m1(d, npix);
        for (int i = 0; i < npix; i++) {
            sind_d[i] = 1. + buffer[i]; // # sin(d) / d avoiding division by zero or near zero, assuming small deflections
        }
    }
    if (triquand == 0){ // #---'close' to equator, where cost ~ 0
        double *cost = dev_cos(tht, npix);
        double *cosd = dev_cos(d, npix);
        for (int i = 0; i < npix; i++) {
            if (cost[i] > 0.8) {
                printf("wrong localization: %f\n", cost[i]);
            }
        }
        double *costp = (double*)malloc(npix * sizeof(double));

        
        double *bufferasin = (double*)malloc(npix * sizeof(double));
        for (int i = 0; i < npix; i++) {
            bufferasin[i] = imd[i] / sqrt(1. - dev_power_element(cost[i],2)) * sind_d[i];
        }
        dphi = dev_asin(bufferasin, npix);
        for (int i = 0; i < npix; i++) {
            costp[i] = cost[i] * cosd[i] - red[i] * sind_d[i] * sqrt(1. - dev_power_element(cost[i],2));
        }
        thtp = dev_acos(costp, npix);
    } else {
        int isnorth = triquand == 1 ? 1 : 0;
        double *sint = dev_sin(tht, npix);
        double ththalf[npix] = {0.0};
        for (int i = 0; i < npix; i++) {
            ththalf[i] = tht[i] * 0.5;
        }
        double *e_t = isnorth == 1 ? dev_sin(ththalf, npix) : dev_cos(ththalf, npix);  //# 1 -+ costh with no precision loss
        for (int i = 0; i < npix; i++) {
            e_t[i] = 2. * dev_power_element(e_t[i],2);
        }
        double dhalf[npix] = {0.0};
        for (int i = 0; i < npix; i++) {
            dhalf[i] = d[i] * 0.5;
        }
        double* e_d = dev_sin(dhalf, npix);
        for (int i = 0; i < npix; i++) {
            e_d[i] = 2. * dev_power_element(e_d[i],2);
        }
        double e_tp[npix] = {0.0};
        for (int i = 0; i < npix; i++) {
            e_tp[i] = e_t[i] + e_d[i] - e_t[i] * e_d[i] + triquand * red[i] * sind_d[i] * sint[i];  //# 1 -+ cost'
        }
        double max[npix] = {0.0};
        double zeros[npix] = {0.0};
        double factor[npix] = {0.0};
        for (int i = 0; i < npix; i++) {
            factor[i] = e_tp[i] * (2. - e_tp[i]);
        }
        maximum(zeros, factor, max, npix);
        double sintp[npix];
        for (int i = 0; i < npix; i++) {
            sintp[i] = sqrt(max[i]);
        }
        if (isnorth == 1){
            //assert np.max(tht) < np.pi * 0.4, ('wrong localization', np.max(tht)); //# -- for the arcsin at the end
            thtp = dev_asin(sintp, npix);
            double buffer_x[npix], buffer_y[npix]; 
            for (int i = 0; i < npix; i++) {
                // TODO possible x/y confusion
                buffer_x[i]  = (1. - e_d[i]) * sint[i] + red[i] * sind_d[i] * (1. - e_t[i]);
                buffer_y[i] = imd[i] * sind_d[i];
            dphi = dev_atan2(buffer_y, buffer_x, npix);
            }
        } else {
            //assert np.min(tht) > np.pi * 0.4, ('wrong localization', np.min(tht)); //# -- for the arcsin at the end
            thtp = dev_asin(sintp, npix);
            double buffer_x[npix], buffer_y[npix]; 
            for (int i = 0; i < npix; i++) {
                // TODO possible x/y confusion
                thtp[i] = PI -  thtp[i];
                buffer_x[i]  = (1. - e_d[i]) * sint[i] + red[i] * sind_d[i] * (e_t[i] - 1.);
                buffer_y[i] = imd[i] * sind_d[i];
            }
            dphi = dev_atan2(buffer_y, buffer_x, npix);
        }
    }
    double *ret = (double*)malloc(2*npix * sizeof(double));
    for (int i = 0; i < npix; i++) {
        ret[i] = thtp[i];
        ret[i + npix] = fmod(phi[i] + dphi[i], 2. * PI);
    }
    return ret;
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

__global__ void compute_pointing(double* thetas, double* phi0, int* nphis, double* ringstarts, double* red, double* imd, int nrings, double *pointings) {
    //idx is nrings
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    double PI = 3.14159265359;
    if (idx <= nrings) {
        int ringstart = ringstarts[idx];
        const int npixs = nphis[idx];
        int* pixs = ring2pixs(ringstart, npixs);
        
        if (npixs > 0) {
            double* t_red = getelements(red, pixs, npixs);
            double* i_imd = getelements(imd, pixs, npixs);

            double *phis = (double*)malloc(npixs * sizeof(double));
            for (int i = 0; i < npixs; i++) {
                phis[i] = fmod(phi0[idx] + i * (2. * PI / npixs),2. * PI);
            }
            //TODO implement this for correct offset
            // phis = phis//[pixs - self.geom.ofs[ir]]
            double *thts = (double*)malloc(npixs * sizeof(double));
            for (int i = 0; i < npixs; i++) {
                thts[i] = thetas[idx];
            }
            double *buff = d2ang(t_red, i_imd, thts, phis, dev_gettriquand(thetas[idx]));
            int *sli = dev_arange(ringstart, ringstart + npixs);
            for (int i = 0; i < npixs; i++) {
                int idx_  = sli[i];
                pointings[idx + idx_] = buff[i];
                pointings[idx + 1 + idx_ + npixs] = buff[i + npixs];
                // TODO implement this (rotation of the polarization angle)
                // cot = np.cos(self.geom.theta[ir]) / np.sin(self.geom.theta[ir])
                // d = np.sqrt(t_red ** 2 + i_imd ** 2)
                // thp_phip_gamma[2, sli] = np.arctan2(i_imd, t_red ) - np.arctan2(i_imd, d * np.sin(d) * cot + t_red * np.cos(d))
                // startpix += len(pixs)
            }
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

int* argsort(double* array, int size) {
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

extern "C" void pointing(double* thetas, double* phi0, int* nphis, double *ringstarts, double *red, double *imd, int nrings, int npix, double *host_result) {
    double *device_thetas, *device_phi0,  *device_ringstarts, *device_red, *device_imd;
    int *device_nphis;
    double *device_result;

    int startpix = 0;
    bool condition1 = allGreaterThanZero(thetas, npix);
    bool condition2 = allLessThanPi(thetas, npix);
    printf("nphis:");
    for (int i = 0; i < 2048; i++) {
        printf("%d ", nphis[i]);
    }
    printf("condition1: %d\n", condition1);
    printf("condition2: %d\n", condition2);
    printf("npix: %d\n", npix);
    // assert(condition1 && condition2);

    int* sorted_ringstarts = argsort(ringstarts, nrings);

    cudaMalloc((void**)&device_thetas, nrings * sizeof(double));
    cudaMalloc((void**)&device_phi0, nrings * sizeof(double));
    cudaMalloc((void**)&device_nphis, nrings * sizeof(int));
    cudaMalloc((void**)&device_ringstarts, nrings * sizeof(double));
    cudaMalloc((void**)&device_red, npix * sizeof(double));
    cudaMalloc((void**)&device_imd, npix * sizeof(double));

    cudaMalloc((void**)&device_result, 2*npix * sizeof(double));

    cudaMemcpy(device_thetas, thetas, nrings * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(device_phi0, phi0, nrings * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(device_nphis, nphis, nrings * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(device_ringstarts, ringstarts, nrings * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(device_red, red, npix * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(device_imd, imd, npix * sizeof(double), cudaMemcpyHostToDevice);

    // auto start = high_resolution_clock::now();
    // const int threadsPerBlock = 256;
    // const int blocksPerGrid = (lmax + threadsPerBlock - 1) / threadsPerBlock;
    compute_pointing<<<1, nrings>>>(device_thetas, device_phi0, device_nphis, device_ringstarts, device_red, device_imd, nrings, device_result);
    cudaDeviceSynchronize();
    // auto stop = high_resolution_clock::now();

    cudaMemcpy(host_result, device_result, 2*npix * sizeof(double), cudaMemcpyDeviceToHost);

    // for (int l = lmax; l <= lmax; ++l) {
        // for (int m = 0; m <= l; ++m) {

    // printf("P_%d^%d(x): ",lmax,mmax);
    // // printf("%d", nrings);
    // for (int i = 0; i < nrings; ++i) {
    //     printf("%.2f ", host_result[i]);
    // }
    // printf("\n");
        // }
    // }

    cudaFree(device_thetas);
    cudaFree(device_phi0);
    cudaFree(device_nphis);
    cudaFree(device_ringstarts);
    cudaFree(device_red);
    cudaFree(device_imd);
    cudaFree(device_result);
}