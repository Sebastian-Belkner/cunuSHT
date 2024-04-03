#include <stdio.h>
#include <stdlib.h>

// #if defined(HAS_NANOBIND)
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/tuple.h>

// #endif

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

std::tuple<intptr_t, intptr_t> Cpointing_ptrs(intptr_t thetas_, intptr_t phi0_, intptr_t nphis_, intptr_t ringstarts_, intptr_t synthmap_, int nring, int npix, double *host_result){
    printf("pointing.cpp:: Cpointing_ptrs\n");
    KernelParams kp;
    kp.thetas = reinterpret_cast<double*>(thetas_);
    kp.phi0 = reinterpret_cast<double*>(phi0_);
    kp.nphis = reinterpret_cast<int*>(nphis_);
    kp.ringstarts = reinterpret_cast<int*>(ringstarts_);
    kp.synthmap = reinterpret_cast<double*>(synthmap_);
    kp.nring = nring;
    kp.npix = npix;

    auto devres = CUpointing_struct(kp);
    // printf("testing_cpp:: %p\n", devres);
    // long* arr = new long[2];
    // arr[0] = reinterpret_cast<intptr_t>(std::get<0>(devres));
    // arr[1] = reinterpret_cast<intptr_t>(std::get<1>(devres));

    // for (int i = 0; i < 2; i++) {
        // printf("arr[%d] = %p\n", i, arr[i]);
    // }
    // return arr;
    printf("pointing.cpp:: Cpointing_ptrs:: %p\n", reinterpret_cast<intptr_t>(std::get<0>(devres)));
    printf("pointing.cpp:: Cpointing_ptrs:: %p\n", reinterpret_cast<intptr_t>(std::get<1>(devres)));
    return std::make_tuple(reinterpret_cast<intptr_t>(std::get<0>(devres)),reinterpret_cast<intptr_t>(std::get<1>(devres)));
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
            intptr_t synthmap_,
            int nring,
            int npix,
            nb::ndarray<double>&host_result
            ) {
            return Cpointing_ptrs(thetas_, phi0_, nphis_, ringstarts_, synthmap_, nring, npix, host_result.data());
        }
    );
}
// #endif