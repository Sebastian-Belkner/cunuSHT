#include <stdio.h>
#include <stdlib.h>

// #if defined(HAS_NANOBIND)
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/array.h>

#include <tuple> 
// #endif

#include "test_pointing.h"
#include "test_pointing.cuh"

// #if defined(HAS_NANOBIND)
namespace nb = nanobind;
using namespace nb::literals;
// #endif

void Cpointing_devres3(int npix, double *host_result){
    CUpointing_devres3(npix, host_result);
    printf("testing_cpp:: host_result with npix=%d:\n",npix);
    for (int i = 0; i < npix; i++) {
        printf("%f, ", host_result[i]);
    }printf("\n");
}

void Cpointing_devres1(int npix, double *host_result){
    double* devres = CUpointing_devres11(npix);
    CUpointing_devres12(npix, devres);
    CUpointing_devres13(npix, devres, host_result);
    printf("testing_cpp:: host_result with npix=%d:\n",npix);
    for (int i = 0; i < npix; i++) {
        printf("%f, ", host_result[i]);
    }printf("\n");
}

void Cpointing_arrdevres3(double* thetas, int nring, int npix, double *host_result){
    CUpointing_arrdevres3(thetas, nring, npix, host_result);
    printf("testing_cpp:: host_result:\n");
    for (int i = 0; i < npix; i++) {
        printf("%f, ", host_result[i]);
    }printf("\n");
}

void Cpointing_garrdevres3(intptr_t red_ptr, int nring, int npix, double *host_result){
    printf("test_pointing.cpp:: Cpointing_garrdevres3\n");
    double* red = reinterpret_cast<double*>(red_ptr);
    CUpointing_garrdevres3(red, nring, npix, host_result);
    printf("testing_cpp:: host_result:\n");
    for (int i = 0; i < npix; i++) {
        printf("%f, ", host_result[i]);
    }printf("\n");
}

void Cpointing_structdevres3(double* red, double* imd, int nring, int npix, double *host_result){
    printf("test_pointing.cpp:: Cpointing_structdevres3\n");
    struct Params params;
    params.red = red;
    params.imd = imd;
    params.nring = nring;
    params.npix = npix;

    CUpointing_structdevres3(params, host_result);
    printf("testing_cpp:: host_result:\n");
    for (int i = 0; i < npix; i++) {
        printf("%f, ", host_result[i]);
    }printf("\n");
}

void Cpointing_structdevres1(double* red, double* imd, int nring, int npix, double *host_result){
    printf("test_pointing.cpp:: Cpointing_structdevres1\n");
    struct Params hostparams;
    hostparams.red = red;
    hostparams.imd = imd;
    hostparams.nring = nring;
    hostparams.npix = npix;

    auto devtup = CUpointing_structdevres11(hostparams);
    CUpointing_structdevres12(hostparams, devtup);
    CUpointing_structdevres13(hostparams, std::get<1>(devtup), host_result);

    printf("testing_cpp:: host_result:\n");
    for (int i = 0; i < npix; i++) {
        printf("%f, ", host_result[i]);
    }printf("\n");
}

uintptr_t Cpointing_structdevres3_retptr(double* red, double* imd, int nring, int npix){
    printf("test_pointing.cpp:: Cpointing_structdevres3_retptr\n");
    struct Params params;
    params.red = red;
    params.imd = imd;
    params.nring = nring;
    params.npix = npix;

    double* devres = CUpointing_structdevres3_retptr(params);
    printf("testing_cpp:: devres: %p\n", devres);

    return reinterpret_cast<uintptr_t>(devres);
}

// #if defined(HAS_NANOBIND)
NB_MODULE(popy, m) {
    m.def(
        "Cpointing_devres3",
        [](
            int npix,
            nb::ndarray<double>&host_result
            ) {
            Cpointing_devres3(npix, host_result.data());
        }
    );
    m.def(
        "Cpointing_devres1",
        [](
            int npix,
            nb::ndarray<double>&host_result
            ) {
            Cpointing_devres3(npix, host_result.data());
        }
    );
    m.def(
        "Cpointing_arrdevres3",
        [](
            nb::ndarray<double>& thetas,
            int nring,
            int npix,
            nb::ndarray<double>&host_result){
            Cpointing_arrdevres3(thetas.data(), nring, npix, host_result.data());
        }
    );
    m.def(
        "Cpointing_garrdevres3",
        [](
            intptr_t red_ptr,
            int nring,
            int npix,
            nb::ndarray<double>&host_result){
            Cpointing_garrdevres3(red_ptr, nring, npix, host_result.data());
        }
    );
    m.def(
        "Cpointing_structdevres3",
        [](
            nb::ndarray<double>& red,
            nb::ndarray<double>& imd,
            int nring,
            int npix,
            nb::ndarray<double>&host_result){
            Cpointing_structdevres3(red.data(), imd.data(), nring, npix, host_result.data());
        }
    );
    m.def(
        "Cpointing_structdevres1",
        [](
            nb::ndarray<double>& red,
            nb::ndarray<double>& imd,
            int nring,
            int npix,
            nb::ndarray<double>&host_result){
            Cpointing_structdevres1(red.data(), imd.data(), nring, npix, host_result.data());
        }
    );
    m.def(
        "Cpointing_structdevres3_retptr",
        [](
            nb::ndarray<double>& red,
            nb::ndarray<double>& imd,
            int nring,
            int npix
            ){
            return Cpointing_structdevres3_retptr(red.data(), imd.data(), nring, npix);
        }
    );
}
// #endif