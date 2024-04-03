#include <stdio.h>
#include <stdlib.h>

// #if defined(HAS_NANOBIND)
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/array.h>

#include <tuple> // For using std::tuple
// #endif

// #include "kernel_params.h"
// #include "kernel_locals.h"

#include "pointing.h"
#include "pointing.cuh"

// #if defined(HAS_NANOBIND)
namespace nb = nanobind;
using namespace nb::literals;
// #endif

void float_to_double(const float* src, double* dest, int size) {
    for (int i = 0; i < size; i++) {
        dest[i] = (double)src[i];
    }
}

intptr_t Cpointing_ptrs(intptr_t thetas_, intptr_t phi0_, intptr_t nphis_, intptr_t ringstarts_, intptr_t red_, intptr_t imd_, int nring, int npix, double *host_result){
    printf("pointing.cpp:: Cpointing_ptrs\n");
    KernelParams kp;
    kp.thetas = reinterpret_cast<double*>(thetas_);
    kp.phi0 = reinterpret_cast<double*>(phi0_);
    kp.nphis = reinterpret_cast<int*>(nphis_);
    kp.ringstarts = reinterpret_cast<int*>(ringstarts_);
    kp.red = reinterpret_cast<double*>(red_);
    kp.imd = reinterpret_cast<double*>(imd_);
    kp.nring = nring;
    kp.npix = npix;

    double* devres = CUpointing_struct(kp);
    printf("testing_cpp:: %p\n", devres);

    return reinterpret_cast<uintptr_t>(devres);
}

// #if defined(HAS_NANOBIND)
NB_MODULE(popy, m) {
    m.def(
        "Cpointing_ptrs",
        [](
            intptr_t thetas_,
            intptr_t phi0_,
            intptr_t nphis_,
            intptr_t ringstarts_,
            intptr_t red_,
            intptr_t imd_,
            int nring,
            int npix,
            nb::ndarray<double>&host_result
            ) {
            return Cpointing_ptrs(thetas_, phi0_, nphis_, ringstarts_, red_, imd_, nring, npix, host_result.data());
        }
    );
}
// #endif