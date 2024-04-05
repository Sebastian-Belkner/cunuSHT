#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/tuple.h>

#include <stdio.h>
#include <stdlib.h>

#include <math.h>
#include <assert.h>
#include <cmath>
#include <chrono>
#include <time.h>

// #include "pointing.h"
#include "kernel_params.h"

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_device_runtime_api.h>

namespace nb = nanobind;
using namespace nb::literals;


// void float_to_double(const float* src, double* dest, int size) {
//     for (int i = 0; i < size; i++) {
//         dest[i] = (double)src[i];
//     }
// }

// std::tuple<intptr_t, intptr_t> Cpointing_ptrs(intptr_t thetas_, intptr_t phi0_, intptr_t nphis_, intptr_t ringstarts_, intptr_t synthmap_, int nring, int npix, double *host_result){
//     printf("pointing.cpp:: Cpointing_ptrs\n");
//     KernelParams kp;
//     kp.thetas = reinterpret_cast<double*>(thetas_);
//     kp.phi0 = reinterpret_cast<double*>(phi0_);
//     kp.nphis = reinterpret_cast<int*>(nphis_);
//     kp.ringstarts = reinterpret_cast<int*>(ringstarts_);
//     kp.synthmap = reinterpret_cast<double*>(synthmap_);
//     kp.nring = nring;
//     kp.npix = npix;

//     auto devres = CUpointing_struct(kp);
//     return std::make_tuple(reinterpret_cast<intptr_t>(std::get<0>(devres)),reinterpret_cast<intptr_t>(std::get<1>(devres)));
// }


// void CUfloat_to_double(const float* src, double* dest, int size) {
//     for (int i = 0; i < size; i++) {
//         dest[i] = (double)src[i];
//     }
// }

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

template <typename Scalar>
__global__ void compute_dummy(Scalar* pt, Scalar* pp, const Scalar* thetas, const Scalar* phi0, const size_t* nphis, const size_t* ringstarts, const Scalar* synthmap, const size_t nring, const size_t npix, KernelLocals kl, const size_t size) {
    //idx is nring
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    double PI = 3.14159265359;
    if (1 == 1) {
        if (idx <= nring) {
            pt[idx] = synthmap[idx];
            pp[idx] = synthmap[idx];
        }
    }
}

template <typename Scalar>
__global__ void compute_pointing_1Dto1D(Scalar* pt, Scalar* pp, const Scalar* thetas, const Scalar* phi0, const size_t* nphis, const size_t* ringstarts, const Scalar* spin1_theta, const Scalar* spin1_phi, const size_t nring, const size_t npix, KernelLocals kl, const size_t size) {
    //idx is nring
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    double PI = 3.14159265359;
    if (idx <= nring) {
        const int ringstart = ringstarts[idx];
        const int npixring = nphis[idx];
        // kl.e_r(sin(theta(iring)), 0, cos(theta(iring)));
        kl.sint = sin(thetas[idx]); 
        kl.cost = cos(thetas[idx]);
        for (int i = ringstart; i < ringstart+npixring; i++) {
            kl.phi[i] = phi0[idx] + i * (2. * PI / npixring);
        }
        for (int i = ringstart; i < ringstart+npixring; i++) {
            kl.d[i] = spin1_theta[i] * spin1_theta[i] + spin1_phi[i] * spin1_theta[i+npix];
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
            kl.e_a1[i] = kl.cost * spin1_theta[i];
            kl.e_a2[i] = kl.phi[i];
            kl.e_a3[i] = -kl.sint * spin1_phi[i];
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
            pt[i] = kl.npt[i];
            pp[i] = kl.npp[i] + kl.phi[i];
            pp[i] = (pp[i] >= 2*PI) ? (pp[i] - 2.*PI) : pp[i];
        }
    }
}

template <typename Scalar>
__global__ void compute_pointing_cparr(Scalar* pt, Scalar* pp, const Scalar* thetas, const Scalar* phi0, const size_t* nphis, const size_t* ringstarts, const Scalar* synthmap, const size_t nring, const size_t npix, KernelLocals kl, const size_t size) {
    //idx is nring
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    double PI = 3.14159265359;
    if (idx <= nring) {
        const int ringstart = ringstarts[idx];
        const int npixring = nphis[idx];
        // kl.e_r(sin(theta(iring)), 0, cos(theta(iring)));
        kl.sint = sin(thetas[idx]); 
        kl.cost = cos(thetas[idx]);
        for (int i = ringstart; i < ringstart+npixring; i++) {
            kl.phi[i] = phi0[idx] + i * (2. * PI / npixring);
        }
        for (int i = ringstart; i < ringstart+npixring; i++) {
            kl.d[i] = synthmap[i] * synthmap[i] + synthmap[i+npix] * synthmap[i+npix];
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
            kl.e_a1[i] = kl.cost * synthmap[i];
            kl.e_a2[i] = kl.phi[i];
            kl.e_a3[i] = -kl.sint * synthmap[i];
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
            pt[i] = kl.npt[i];
            pp[i] = kl.npp[i] + kl.phi[i];
            pp[i] = (pp[i] >= 2*PI) ? (pp[i] - 2.*PI) : pp[i];
        }
    }
}

template <typename Scalar>
void CUpointing_1Dto1D(
    nb::ndarray<const Scalar, nb::ndim<1>, nb::device::cuda> thetas,
    nb::ndarray<const Scalar, nb::ndim<1>, nb::device::cuda> phi0,
    nb::ndarray<const size_t, nb::ndim<1>, nb::device::cuda> nphis,
    nb::ndarray<const size_t, nb::ndim<1>, nb::device::cuda> ringstarts,
    nb::ndarray<const Scalar, nb::ndim<1>, nb::device::cuda> spin1_theta,
    nb::ndarray<const Scalar, nb::ndim<1>, nb::device::cuda> spin1_phi,
    nb::ndarray<Scalar, nb::ndim<1>, nb::device::cuda> outarr_pt,
    nb::ndarray<Scalar, nb::ndim<1>, nb::device::cuda> outarr_pp) {

    const size_t size = thetas.size();
    const size_t npix = spin1_theta.size();
    const size_t nring = ringstarts.size();
    size_t block_size = 256;
    size_t num_blocks = (size + block_size - 1) / block_size;

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
    cudaMalloc((void**)&dev_philocs, npix * sizeof(double));


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
    cudaDeviceSynchronize();
    // compute_dummy<<<num_blocks, block_size>>>(outarr_pt.data(), outarr_pp.data(), thetas.data(), phi0.data(), nphis.data(), ringstarts.data(), synthmap.data(), nring, npix, kl, size);
    compute_pointing_1Dto1D<<<num_blocks, block_size>>>(outarr_pt.data(), outarr_pp.data(), thetas.data(), phi0.data(), nphis.data(), ringstarts.data(), spin1_theta.data(), spin1_phi.data(), nring, npix, kl, size);
    cudaDeviceSynchronize();
}

template <typename Scalar>
void CUpointing_cparr(
    nb::ndarray<const Scalar, nb::ndim<1>, nb::device::cuda> thetas,
    nb::ndarray<const Scalar, nb::ndim<1>, nb::device::cuda> phi0,
    nb::ndarray<const size_t, nb::ndim<1>, nb::device::cuda> nphis,
    nb::ndarray<const size_t, nb::ndim<1>, nb::device::cuda> ringstarts,
    nb::ndarray<const Scalar, nb::ndim<1>, nb::device::cuda> synthmap,
    nb::ndarray<Scalar, nb::ndim<1>, nb::device::cuda> outarr_pt,
    nb::ndarray<Scalar, nb::ndim<1>, nb::device::cuda> outarr_pp) {

    const size_t size = thetas.size();
    const size_t npix = synthmap.size()/2;
    const size_t nring = ringstarts.size();
    size_t block_size = 256;
    size_t num_blocks = (size + block_size - 1) / block_size;

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
    cudaMalloc((void**)&dev_philocs, npix * sizeof(double));


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
    // compute_dummy<<<num_blocks, block_size>>>(outarr_pt.data(), outarr_pp.data(), thetas.data(), phi0.data(), nphis.data(), ringstarts.data(), synthmap.data(), nring, npix, kl, size);
    compute_pointing_cparr<<<num_blocks, block_size>>>(outarr_pt.data(), outarr_pp.data(), thetas.data(), phi0.data(), nphis.data(), ringstarts.data(), synthmap.data(), nring, npix, kl, size);
    cudaDeviceSynchronize();

    cudaError_t errSync  = cudaGetLastError();
    cudaError_t errAsync = cudaDeviceSynchronize();
    if (errSync != cudaSuccess) 
    printf("Sync kernel error: %s\n", cudaGetErrorString(errSync));
    if (errAsync != cudaSuccess)
    printf("Async kernel error: %s\n", cudaGetErrorString(errAsync));
}

NB_MODULE(popy, m) {
    // m.def(
    //     "Cpointing_ptrs",
    //     [](
    //         intptr_t thetas_,
    //         intptr_t phi0_,
    //         intptr_t nphis_,
    //         intptr_t ringstarts_,
    //         intptr_t synthmap_,
    //         int nring,
    //         int npix,
    //         nb::ndarray<double>&host_result
    //         ) {
    //         return Cpointing_ptrs(thetas_, phi0_, nphis_, ringstarts_, synthmap_, nring, npix, host_result.data());
    //     }
    // );
    m.def("CUpointing_1Dto1D",
        &CUpointing_1Dto1D<double>,
        "thetas"_a.noconvert(),
        "phi0"_a.noconvert(),
        "nphis"_a.noconvert(),
        "ringstarts"_a.noconvert(),
        "spin1_theta"_a.noconvert(),
        "spin1_phi"_a.noconvert(),
        "outarr_pt"_a.noconvert(),
        "outarr_pp"_a.noconvert()
    );
    m.def("CUpointing_cparr",
        &CUpointing_cparr<double>,
        "thetas"_a.noconvert(),
        "phi0"_a.noconvert(),
        "nphis"_a.noconvert(),
        "ringstarts"_a.noconvert(),
        "synthmap"_a.noconvert(),
        "outarr_pt"_a.noconvert(),
        "outarr_pp"_a.noconvert()
    );
}